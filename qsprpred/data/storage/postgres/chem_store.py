import json
import uuid
from collections.abc import Generator, Iterable, Sized
from typing import Any, Literal

import numpy as np
import pandas as pd
from rdkit import Chem

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError as exc:  # pragma: no cover - only triggered when optional dep missing
    raise ImportError(
        "PostgresChemStore requires psycopg. Install it with: "
        "pip install 'psycopg[binary]'"
    ) from exc

from qsprpred.data.storage.interfaces.stored_mol import StoredMol
from qsprpred.data.storage.tabular.simple import ParallelizedChemStore
from qsprpred.data.storage.tabular.stored_mol import TabularMol
from qsprpred.utils.parallel import MultiprocessingJITGenerator, ParallelGenerator


class PostgresChemStore(ParallelizedChemStore):
    """PostgreSQL implementation of :class:`ChemStore`.

    This is intentionally a minimal first implementation. The public API mirrors the
    existing PandasChemStore where possible, but molecules are persisted in a
    PostgreSQL table instead of a pandas DataFrame.

    The table keeps fixed columns for the molecule identity and SMILES and stores all
    additional molecule metadata in a JSONB column named ``props``. If the RDKit
    cartridge is available, the schema can also create a generated ``mol`` column.
    """

    def __init__(
        self,
        name: str,
        connection_string: str,
        table_name: str = "molecules",
        schema: str = "public",
        smiles_col: str = "SMILES",
        id_col: str | None = None,
        autoindex_name: str | None = None,
        library: str | None = None,
        create: bool = True,
        use_rdkit_cartridge: bool = True,
        chunk_processor: ParallelGenerator | None = None,
        chunk_size: int | None = None,
        n_jobs: int = 1,
    ):
        """Create a PostgreSQL-backed ChemStore.

        Args:
            name: Logical storage name.
            connection_string: PostgreSQL connection string.
            table_name: Name of the PostgreSQL table.
            schema: PostgreSQL schema name.
            smiles_col: Public property name used for SMILES values.
            id_col: Public property name used for molecule IDs.
            autoindex_name: Alternative public property name used for molecule IDs.
            library: Default library name. If omitted, ``f"{name}_library"`` is used.
            create: Create the database schema/table on initialization.
            use_rdkit_cartridge: Try to install/use RDKit cartridge objects.
            chunk_processor: Parallel generator used by inherited ``apply`` method.
            chunk_size: Default processing chunk size.
            n_jobs: Number of parallel jobs for inherited processing utilities.
        """
        super().__init__()
        if not name:
            raise ValueError("Storage name must not be empty.")
        if not connection_string:
            raise ValueError("connection_string must not be empty.")
        self._name = name
        self.connectionString = connection_string
        self.schema = schema
        self.tableName = table_name
        self.defaultLibrary = library or f"{name}_library"
        self._smilesProp = smiles_col
        self._idProp = id_col or autoindex_name or f"{name}_ID"
        self.useRDKitCartridge = use_rdkit_cartridge
        self.nJobs = n_jobs
        self.chunkSize = chunk_size
        self.chunkProcessor = (
            MultiprocessingJITGenerator(n_workers=self.nJobs)
            if chunk_processor is None else chunk_processor
        )
        if create:
            self.createSchema()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str):
        self._name = value

    @property
    def idProp(self) -> str:
        return self._idProp

    @idProp.setter
    def idProp(self, value: str):
        self._idProp = value

    @property
    def smilesProp(self) -> str:
        return self._smilesProp

    @property
    def chunkProcessor(self) -> ParallelGenerator:
        return self._chunkProcessor

    @chunkProcessor.setter
    def chunkProcessor(self, value: ParallelGenerator):
        if not isinstance(value, ParallelGenerator):
            raise ValueError("chunk_processor must be a ParallelGenerator instance.")
        self._chunkProcessor = value

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
    def qualifiedTableName(self) -> str:
        return f'"{self.schema}"."{self.tableName}"'

    def _connect(self):
        return psycopg.connect(self.connectionString, row_factory=dict_row)

    def createSchema(self):
        """Create schema and molecule table if they do not exist."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"')
                if self.useRDKitCartridge:
                    try:
                        cur.execute("CREATE EXTENSION IF NOT EXISTS rdkit")
                        cur.execute(self._create_table_sql(with_rdkit=True))
                    except Exception:
                        conn.rollback()
                        # Fall back to a plain PostgreSQL table if the cloud provider
                        # does not permit the RDKit extension.
                        self.useRDKitCartridge = False
                        with self._connect() as fallback_conn:
                            with fallback_conn.cursor() as fallback_cur:
                                fallback_cur.execute(
                                    f'CREATE SCHEMA IF NOT EXISTS "{self.schema}"'
                                )
                                fallback_cur.execute(
                                    self._create_table_sql(with_rdkit=False)
                                )
                                self._create_indexes(fallback_cur, with_rdkit=False)
                            fallback_conn.commit()
                        return
                else:
                    cur.execute(self._create_table_sql(with_rdkit=False))
                self._create_indexes(cur, with_rdkit=self.useRDKitCartridge)
            conn.commit()

    def _create_table_sql(self, with_rdkit: bool) -> str:
        mol_column = (
            ",\n    mol MOL GENERATED ALWAYS AS "
            "(mol_from_smiles(smiles::cstring)) STORED"
            if with_rdkit else ""
        )
        return f"""
CREATE TABLE IF NOT EXISTS {self.qualifiedTableName} (
    id TEXT PRIMARY KEY,
    library TEXT NOT NULL,
    smiles TEXT NOT NULL,
    original_smiles TEXT,
    props JSONB NOT NULL DEFAULT '{{}}'::jsonb{mol_column},
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

    def _create_indexes(self, cur, with_rdkit: bool):
        cur.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{self.tableName}_library" '
            f"ON {self.qualifiedTableName}(library)"
        )
        cur.execute(
            f'CREATE INDEX IF NOT EXISTS "idx_{self.tableName}_props_gin" '
            f"ON {self.qualifiedTableName} USING GIN(props)"
        )
        if with_rdkit:
            cur.execute(
                f'CREATE INDEX IF NOT EXISTS "idx_{self.tableName}_mol_gist" '
                f"ON {self.qualifiedTableName} USING GIST(mol)"
            )

    def _normalize_json_value(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if pd.isna(value):
            return None
        return value

    def _row_to_mol(self, row: dict[str, Any]) -> TabularMol:
        props = dict(row.get("props") or {})
        props[self.idProp] = row["id"]
        props[self.smilesProp] = row["smiles"]
        if row.get("original_smiles") is not None:
            props["Original_SMILES"] = row["original_smiles"]
        props["library"] = row["library"]
        return TabularMol(row["id"], self.name, row["smiles"], props=props)

    def _rows_to_df(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        records = []
        for row in rows:
            record = dict(row.get("props") or {})
            record[self.idProp] = row["id"]
            record[self.smilesProp] = row["smiles"]
            record["library"] = row["library"]
            if row.get("original_smiles") is not None:
                record["Original_SMILES"] = row["original_smiles"]
            records.append(record)
        if not records:
            return pd.DataFrame(index=pd.Index([], name=self.idProp))
        return pd.DataFrame(records)

    def addMols(
        self,
        smiles: Iterable[str],
        props: dict[str, list] | None = None,
        library: str | None = None,
        raise_on_existing: bool = True,
        **kwargs,
    ) -> list[StoredMol]:
        """Add molecules and return the added molecules as StoredMol instances."""
        smiles = list(smiles)
        props = props or {}
        library = library or self.defaultLibrary
        for prop_name, values in props.items():
            if len(values) != len(smiles):
                raise ValueError(
                    f"Property '{prop_name}' has {len(values)} values, but "
                    f"{len(smiles)} SMILES were supplied."
                )

        ids = props.get(self.idProp) or [uuid.uuid4().hex for _ in smiles]
        records = []
        for idx, smi in enumerate(smiles):
            mol_id = str(ids[idx])
            mol_props = {
                key: self._normalize_json_value(values[idx])
                for key, values in props.items()
                if key not in {self.idProp, self.smilesProp}
            }
            records.append((mol_id, library, smi, smi, json.dumps(mol_props)))

        if not records:
            return []

        if raise_on_existing:
            insert_sql = f"""
INSERT INTO {self.qualifiedTableName}
(id, library, smiles, original_smiles, props)
VALUES (%s, %s, %s, %s, %s::jsonb)
"""
        else:
            insert_sql = f"""
INSERT INTO {self.qualifiedTableName}
(id, library, smiles, original_smiles, props)
VALUES (%s, %s, %s, %s, %s::jsonb)
ON CONFLICT (id) DO UPDATE SET
    library = EXCLUDED.library,
    smiles = EXCLUDED.smiles,
    original_smiles = EXCLUDED.original_smiles,
    props = {self.qualifiedTableName}.props || EXCLUDED.props,
    updated_at = NOW()
"""

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(insert_sql, records)
            conn.commit()
        return [self.getMol(mol_id) for mol_id in ids]

    def addEntries(
        self,
        ids: list[str],
        props: dict[str, list],
        raise_on_existing: bool = True,
        library: str | None = None,
    ):
        smiles = props.get(self.smilesProp)
        if smiles is None:
            raise ValueError(f"Property '{self.smilesProp}' is required.")
        props = dict(props)
        props[self.idProp] = ids
        return self.addMols(
            smiles,
            props=props,
            library=library,
            raise_on_existing=raise_on_existing,
        )

    def getMol(self, mol_id: str) -> StoredMol:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, library, smiles, original_smiles, props "
                    f"FROM {self.qualifiedTableName} WHERE id = %s",
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
                    f"DELETE FROM {self.qualifiedTableName} WHERE id = ANY(%s)",
                    (ids,),
                )
            conn.commit()

    def getMolIDs(self) -> tuple[str, ...]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT id FROM {self.qualifiedTableName} ORDER BY id")
                return tuple(row["id"] for row in cur.fetchall())

    def getMolCount(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT COUNT(*) AS count FROM {self.qualifiedTableName}")
                return int(cur.fetchone()["count"])

    def getProperties(self) -> list[str]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
SELECT DISTINCT jsonb_object_keys(props) AS prop
FROM {self.qualifiedTableName}
"""
                )
                props = [row["prop"] for row in cur.fetchall()]
        return list({self.idProp, self.smilesProp, "library", *props})

    def hasProperty(self, name: str) -> bool:
        return name in self.getProperties()

    def getProperty(self, name: str, ids: list[str] | None = None) -> pd.Series:
        if name == self.idProp:
            sql_name = "id"
        elif name == self.smilesProp:
            sql_name = "smiles"
        elif name == "library":
            sql_name = "library"
        else:
            sql_name = None

        params: list[Any] = []
        where = ""
        if ids is not None:
            where = "WHERE id = ANY(%s)"
            params.append(list(ids))

        select_expr = sql_name if sql_name else "props ->> %s"
        if sql_name is None:
            params.insert(0, name)
        query = f"SELECT id, {select_expr} AS value FROM {self.qualifiedTableName} {where}"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, tuple(params))
                rows = cur.fetchall()
        return pd.Series(
            [row["value"] for row in rows],
            index=pd.Index([row["id"] for row in rows], name=self.idProp),
            name=name,
        )

    def addProperty(self, name: str, data: Sized, ids: list[str] | None = None):
        data = list(data)
        ids = ids or list(self.getMolIDs())
        if len(data) != len(ids):
            raise ValueError(
                f"Property '{name}' has {len(data)} values, but {len(ids)} IDs were supplied."
            )
        if name in {self.idProp, self.smilesProp}:
            raise ValueError(f"Cannot overwrite protected property '{name}'.")
        with self._connect() as conn:
            with conn.cursor() as cur:
                for mol_id, value in zip(ids, data):
                    value = json.dumps({name: self._normalize_json_value(value)})
                    cur.execute(
                        f"""
UPDATE {self.qualifiedTableName}
SET props = props || %s::jsonb,
    updated_at = NOW()
WHERE id = %s
""",
                        (value, mol_id),
                    )
            conn.commit()

    def removeProperty(self, name: str):
        if name in {self.idProp, self.smilesProp, "library"}:
            raise ValueError(f"Cannot remove protected property '{name}'.")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
UPDATE {self.qualifiedTableName}
SET props = props - %s,
    updated_at = NOW()
""",
                    (name,),
                )
            conn.commit()

    def getDF(self) -> pd.DataFrame:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT id, library, smiles, original_smiles, props "
                    f"FROM {self.qualifiedTableName} ORDER BY id"
                )
                rows = cur.fetchall()
        return self._rows_to_df(rows)

    def getSubset(
        self,
        subset: Iterable[str],
        ids: Iterable[str] | None = None,
        name: str | None = None,
    ) -> pd.DataFrame:
        df = self.getDF()
        if ids is not None:
            df = df[df[self.idProp].isin(list(ids))]
        cols = list({self.idProp, self.smilesProp, *subset})
        return df[[col for col in cols if col in df.columns]]

    def iterChunks(
        self,
        size: int | None = None,
        on_props: Iterable[str] | None = None,
        chunk_type: Literal["mol", "smiles", "rdkit", "df"] = "mol",
    ) -> Generator[list[StoredMol | str | Chem.Mol | pd.DataFrame], None, None]:
        size = size or self.chunkSize
        on_props = list(on_props or self.getProperties())
        offset = 0
        while True:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
SELECT id, library, smiles, original_smiles, props
FROM {self.qualifiedTableName}
ORDER BY id
LIMIT %s OFFSET %s
""",
                        (size, offset),
                    )
                    rows = cur.fetchall()
            if not rows:
                break
            df = self._rows_to_df(rows)
            if chunk_type == "df":
                yield df[[col for col in {self.idProp, self.smilesProp, *on_props} if col in df.columns]]
            elif chunk_type == "smiles":
                yield list(df[self.smilesProp])
            elif chunk_type == "rdkit":
                mols = []
                for _, row in df.iterrows():
                    mol = Chem.MolFromSmiles(row[self.smilesProp])
                    for prop in on_props:
                        if prop in row and pd.notna(row[prop]):
                            mol.SetProp(prop, str(row[prop]))
                    mols.append(mol)
                yield mols
            elif chunk_type == "mol":
                yield [self._row_to_mol(row) for row in rows]
            else:
                raise ValueError(f"Unsupported chunk_type: {chunk_type}")
            offset += size

    def iterMols(self) -> Generator[StoredMol, None, None]:
        for chunk in self.iterChunks(chunk_type="mol"):
            for mol in chunk:
                yield mol

    def save(self):
        """No-op compatibility method; PostgreSQL commits are persistent."""
        return self.qualifiedTableName

    def reload(self):
        """No-op compatibility method; each call reads fresh data from PostgreSQL."""
        return None

    def clear(self):
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.qualifiedTableName}")
            conn.commit()
