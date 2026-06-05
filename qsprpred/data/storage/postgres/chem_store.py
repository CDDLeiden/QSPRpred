import json
from typing import Iterable, Generator, Literal, Sized, Any

import pandas as pd
import psycopg
from rdkit import Chem

from qsprpred.data.chem.identifiers import ChemIdentifier, IndexIdentifier
from qsprpred.data.chem.standardizers import ChemStandardizer
from qsprpred.data.storage.tabular.simple import ParallelizedChemStore
from qsprpred.data.storage.tabular.stored_mol import TabularMol
from qsprpred.utils.parallel import MultiprocessingJITGenerator, ParallelGenerator


class PostgresChemStore(ParallelizedChemStore):
    def __init__(
        self,
        name: str,
        connection_string: str,
        table_name: str = "molecules",
        schema: str = "public",
        smiles_col: str = "SMILES",
        id_col: str | None = None,
        autoindex_name: str | None = None,
        use_rdkit_cartridge: bool = True,
        create: bool = True,
        standardizer: ChemStandardizer | None = None,
        identifier: ChemIdentifier | None = None,
        chunk_processor: ParallelGenerator | None = None,
        chunk_size: int | None = 1000,
        n_jobs: int = 1,
    ):
        self._name = name
        self.connectionString = connection_string
        self.tableName = table_name
        self.schema = schema
        self.useRdkitCartridge = use_rdkit_cartridge
        self._smilesProp = smiles_col
        self._idProp = id_col or autoindex_name or f"{name}_ID"
        self._standardizer = standardizer
        self._identifier = identifier or IndexIdentifier()
        self._chunkSize = chunk_size
        self._nJobs = n_jobs
        self._chunkProcessor = chunk_processor or MultiprocessingJITGenerator(
            n_workers=n_jobs
        )

        if create:
            self._create_storage()

    @property
    def name(self) -> str:
        return self._name

    @property
    def idProp(self) -> str:
        return self._idProp

    @property
    def smilesProp(self) -> str:
        return self._smilesProp

    @property
    def originalSmilesProp(self) -> str:
        return "original_smiles"

    @property
    def metaFile(self) -> str:
        return f"postgresql://{self.schema}.{self.tableName}"

    @property
    def standardizer(self) -> ChemStandardizer | None:
        return self._standardizer

    @property
    def identifier(self) -> ChemIdentifier:
        return self._identifier

    @property
    def chunkSize(self) -> int:
        return self._chunkSize

    @chunkSize.setter
    def chunkSize(self, value: int | None):
        self._chunkSize = value or 1000

    @property
    def nJobs(self) -> int:
        return self._nJobs

    @nJobs.setter
    def nJobs(self, value: int | None):
        self._nJobs = value or 1

    @property
    def chunkProcessor(self) -> ParallelGenerator:
        return self._chunkProcessor

    @chunkProcessor.setter
    def chunkProcessor(self, value: ParallelGenerator):
        self._chunkProcessor = value

    def _connect(self):
        return psycopg.connect(self.connectionString, connect_timeout=10)

    def _qualified_table(self) -> str:
        return f"{self.schema}.{self.tableName}"

    def _create_storage(self):
        with self._connect() as conn:
            with conn.cursor() as cur:
                if self.useRdkitCartridge:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS rdkit;")

                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._qualified_table()} (
                        id TEXT PRIMARY KEY,
                        smiles TEXT NOT NULL,
                        mol MOL,
                        library TEXT,
                        props JSONB DEFAULT '{{}}'::jsonb
                    );
                    """
                )

                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.tableName}_library
                    ON {self._qualified_table()}(library);
                    """
                )

                cur.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.tableName}_props
                    ON {self._qualified_table()} USING GIN(props);
                    """
                )
            conn.commit()

    def _row_to_mol(self, row) -> TabularMol:
        mol_id, smiles, library, props = row
        props = props or {}
        props[self.idProp] = mol_id
        props[self.smilesProp] = smiles
        props["library"] = library
        return TabularMol(mol_id, self.name, smiles, props=props)

    def clear(self):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DELETE FROM {self._qualified_table()};")
            conn.commit()

    def save(self) -> str:
        return self.metaFile

    def reload(self):
        return self

    def addMols(
        self,
        smiles: Iterable[str],
        props: dict[str, list] | None = None,
        library: str | None = None,
        raise_on_existing: bool = True,
        **kwargs,
    ) -> list[TabularMol]:
        smiles = list(smiles)
        props = props or {}
        library = library or f"{self.name}_library"

        inserted_ids = []

        with self._connect() as conn:
            with conn.cursor() as cur:
                for idx, smi in enumerate(smiles):
                    mol_id = self.identifier(smi)

                    row_props = {}
                    for key, values in props.items():
                        row_props[key] = values[idx]

                    row_props[self.idProp] = mol_id
                    row_props[self.smilesProp] = smi

                    if raise_on_existing:
                        sql = f"""
                            INSERT INTO {self._qualified_table()}
                                (id, smiles, mol, library, props)
                            VALUES
                                (%s, %s, mol_from_smiles(%s), %s, %s::jsonb);
                        """
                    else:
                        sql = f"""
                            INSERT INTO {self._qualified_table()}
                                (id, smiles, mol, library, props)
                            VALUES
                                (%s, %s, mol_from_smiles(%s), %s, %s::jsonb)
                            ON CONFLICT (id) DO UPDATE SET
                                smiles = EXCLUDED.smiles,
                                mol = EXCLUDED.mol,
                                library = EXCLUDED.library,
                                props = EXCLUDED.props;
                        """

                    cur.execute(
                        sql,
                        (
                            mol_id,
                            smi,
                            smi,
                            library,
                            json.dumps(row_props),
                        ),
                    )
                    inserted_ids.append(mol_id)

            conn.commit()

        return [self.getMol(mol_id) for mol_id in inserted_ids]

    def addEntries(
        self,
        ids: list[str],
        props: dict[str, list],
        raise_on_existing: bool = True,
    ):
        smiles_values = props.get(self.smilesProp) or props.get("smiles")
        if smiles_values is None:
            raise ValueError(f"Missing required property {self.smilesProp}")

        with self._connect() as conn:
            with conn.cursor() as cur:
                for idx, mol_id in enumerate(ids):
                    smi = smiles_values[idx]
                    row_props = {key: values[idx] for key, values in props.items()}
                    row_props[self.idProp] = mol_id
                    row_props[self.smilesProp] = smi

                    if raise_on_existing:
                        sql = f"""
                            INSERT INTO {self._qualified_table()}
                                (id, smiles, mol, library, props)
                            VALUES
                                (%s, %s, mol_from_smiles(%s), %s, %s::jsonb);
                        """
                    else:
                        sql = f"""
                            INSERT INTO {self._qualified_table()}
                                (id, smiles, mol, library, props)
                            VALUES
                                (%s, %s, mol_from_smiles(%s), %s, %s::jsonb)
                            ON CONFLICT (id) DO UPDATE SET
                                smiles = EXCLUDED.smiles,
                                mol = EXCLUDED.mol,
                                props = EXCLUDED.props;
                        """

                    cur.execute(
                        sql,
                        (
                            mol_id,
                            smi,
                            smi,
                            self.name,
                            json.dumps(row_props),
                        ),
                    )
            conn.commit()

    def getMol(self, mol_id: str) -> TabularMol:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, smiles, library, props
                    FROM {self._qualified_table()}
                    WHERE id = %s;
                    """,
                    (mol_id,),
                )
                row = cur.fetchone()

        if row is None:
            raise ValueError(f"Molecule with ID {mol_id} not found.")

        return self._row_to_mol(row)

    def removeMol(self, mol_id: str):
        self.dropEntries([mol_id])

    def dropEntries(self, ids: Iterable[str]):
        ids = list(ids)
        if not ids:
            return

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {self._qualified_table()}
                    WHERE id = ANY(%s);
                    """,
                    (ids,),
                )
            conn.commit()

    def getMolIDs(self) -> tuple[str, ...]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id
                    FROM {self._qualified_table()}
                    ORDER BY id;
                    """
                )
                return tuple(row[0] for row in cur.fetchall())

    def getMolCount(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) FROM {self._qualified_table()};")
                return cur.fetchone()[0]

    def getDF(self) -> pd.DataFrame:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, smiles, library, props
                    FROM {self._qualified_table()}
                    ORDER BY id;
                    """
                )
                rows = cur.fetchall()

        records = []
        for mol_id, smiles, library, props in rows:
            record = props or {}
            record[self.idProp] = mol_id
            record[self.smilesProp] = smiles
            record["library"] = library
            records.append(record)

        if not records:
            return pd.DataFrame(columns=[self.idProp, self.smilesProp])

        df = pd.DataFrame(records)
        return df.set_index(self.idProp, drop=False)

    def getProperty(self, name: str, ids: list[str] | None = None) -> pd.Series:
        df = self.getDF()

        if ids is not None:
            df = df[df[self.idProp].isin(ids)]

        if name == self.idProp:
            return pd.Series(df[self.idProp].values, index=df[self.idProp], name=name)

        if name == self.smilesProp:
            return pd.Series(df[self.smilesProp].values, index=df[self.idProp], name=name)

        if name not in df.columns:
            return pd.Series(index=pd.Index([], name=self.idProp), name=name)

        return pd.Series(df[name].values, index=df[self.idProp], name=name)

    def getProperties(self) -> list[str]:
        df = self.getDF()
        return list(df.columns)

    def hasProperty(self, name: str) -> bool:
        return name in self.getProperties()

    def addProperty(self, name: str, data: Sized, ids: list[str] | None = None):
        data = list(data)

        if ids is None:
            ids = list(self.getMolIDs())

        if len(ids) != len(data):
            raise ValueError("Length of ids and data must match.")

        with self._connect() as conn:
            with conn.cursor() as cur:
                for mol_id, value in zip(ids, data):
                    cur.execute(
                        f"""
                        UPDATE {self._qualified_table()}
                        SET props = jsonb_set(
                            COALESCE(props, '{{}}'::jsonb),
                            %s,
                            %s::jsonb,
                            true
                        )
                        WHERE id = %s;
                        """,
                        ([name], json.dumps(value), mol_id),
                    )
            conn.commit()

    def removeProperty(self, name: str):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    UPDATE {self._qualified_table()}
                    SET props = props - %s;
                    """,
                    (name,),
                )
            conn.commit()

    def getSubset(
        self,
        subset: Iterable[str],
        ids: Iterable[str] | None = None,
    ):
        df = self.getDF()

        if ids is not None:
            ids = list(ids)
            df = df[df[self.idProp].isin(ids)]

        cols = [col for col in subset if col in df.columns]
        if self.idProp not in cols:
            cols.insert(0, self.idProp)
        if self.smilesProp not in cols:
            cols.append(self.smilesProp)

        return df[cols]

    def iterChunks(
        self,
        size: int | None = None,
        on_props: list | None = None,
        chunk_type: Literal["mol", "smiles", "rdkit", "df"] = "mol",
    ) -> Generator[list[TabularMol | str | Chem.Mol | pd.DataFrame], None, None]:
        size = size or self.chunkSize or 1000
        offset = 0

        while True:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT id, smiles, library, props
                        FROM {self._qualified_table()}
                        ORDER BY id
                        LIMIT %s OFFSET %s;
                        """,
                        (size, offset),
                    )
                    rows = cur.fetchall()

            if not rows:
                break

            mols = [self._row_to_mol(row) for row in rows]

            if chunk_type == "mol":
                yield mols
            elif chunk_type == "smiles":
                yield [mol.smiles for mol in mols]
            elif chunk_type == "rdkit":
                yield [Chem.MolFromSmiles(mol.smiles) for mol in mols]
            elif chunk_type == "df":
                records = []
                for mol in mols:
                    rec = mol.props or {}
                    records.append(rec)
                yield pd.DataFrame(records)
            else:
                raise ValueError(f"Unsupported chunk_type: {chunk_type}")

            offset += size

    def iterMols(self) -> Generator[TabularMol, None, None]:
        for chunk in self.iterChunks(chunk_type="mol"):
            for mol in chunk:
                yield mol

    def applyIdentifier(self, identifier: ChemIdentifier):
        self._identifier = identifier
        raise NotImplementedError(
            "applyIdentifier() is not implemented for PostgresChemStore yet."
        )

    def applyStandardizer(self, standardizer: ChemStandardizer):
        self._standardizer = standardizer
        raise NotImplementedError(
            "applyStandardizer() is not implemented for PostgresChemStore yet."
        )

    def searchOnProperty(
        self,
        prop_name: str,
        values: list[float | int | str],
        exact: bool = False,
    ):
        df = self.getDF()

        if prop_name not in df.columns:
            return df.iloc[0:0]

        if exact:
            return df[df[prop_name].isin(values)]

        mask = pd.Series(False, index=df.index)
        for value in values:
            mask = mask | df[prop_name].astype(str).str.contains(str(value), na=False)

        return df[mask]

    def searchWithSMARTS(self, patterns: list[str]):
        raise NotImplementedError(
            "searchWithSMARTS() will be implemented through RDKit Cartridge in the next phase."
        )

    def __len__(self):
        return self.getMolCount()

    def __contains__(self, item):
        return item in self.getMolIDs()

    def __getitem__(self, item):
        return self.getMol(item)