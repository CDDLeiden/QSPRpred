import json
import re
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
    """PostgreSQL-backed ChemStore implementation.

    If ``run_id`` is provided, the physical PostgreSQL table is treated as a
    stable shared table. Rows belonging to different runs are separated by the
    ``run_id`` column and all read/write operations are automatically scoped to
    this run. This is useful for tests and demos where repeated executions should
    not create new physical tables.
    """

    def __init__(
            self,
            name: str,
            connection_string: str,
            table_name: str = "molecules",
            schema: str = "public",
            run_id: str | None = None,
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
        self.tableName = self._safe_sql_identifier(table_name)
        self.schema = self._safe_sql_identifier(schema)
        self.runId = run_id
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
        if self.runId is not None:
            return f"postgresql://{self.schema}.{self.tableName}?run_id={self.runId}"
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

    @staticmethod
    def _safe_sql_identifier(identifier: str) -> str:
        """Validate SQL identifiers used for schema/table/index names."""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
            raise ValueError(f"Unsafe SQL identifier: {identifier!r}")
        return identifier

    def _connect(self):
        return psycopg.connect(self.connectionString, connect_timeout=10)

    def _qualified_table(self) -> str:
        return f"{self.schema}.{self.tableName}"

    def _run_where(self, prefix: str = "WHERE") -> tuple[str, tuple]:
        if self.runId is None:
            return "", tuple()
        return f"{prefix} run_id = %s", (self.runId,)

    def _ensure_test_run_registry(self, cur):
        """Create/migrate registry tables and register the current run.

        The registry tables may already exist from older test versions. This method
        therefore creates only a minimal table first and then adds all expected
        columns with ``ADD COLUMN IF NOT EXISTS``.
        """
        if self.runId is None:
            return

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.chemstore_test_runs (
                run_id TEXT PRIMARY KEY
            );
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {self.schema}.chemstore_test_tables (
                id BIGSERIAL PRIMARY KEY
            );
            """
        )

        for column_sql in [
            "test_name TEXT",
            "started_at TIMESTAMPTZ",
            "backend TEXT",
            "table_prefix TEXT",
            "note TEXT",
        ]:
            cur.execute(
                f"""
                ALTER TABLE {self.schema}.chemstore_test_runs
                ADD COLUMN IF NOT EXISTS {column_sql};
                """
            )

        for column_sql in [
            "run_id TEXT",
            "table_name TEXT",
            "created_at TIMESTAMPTZ DEFAULT now()",
        ]:
            cur.execute(
                f"""
                ALTER TABLE {self.schema}.chemstore_test_tables
                ADD COLUMN IF NOT EXISTS {column_sql};
                """
            )

        cur.execute(
            f"""
            INSERT INTO {self.schema}.chemstore_test_runs
                (run_id, test_name, started_at, backend, table_prefix, note)
            VALUES
                (%s, %s, now(), %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                test_name = EXCLUDED.test_name,
                backend = EXCLUDED.backend,
                table_prefix = EXCLUDED.table_prefix,
                note = COALESCE(EXCLUDED.note, {self.schema}.chemstore_test_runs.note),
                started_at = COALESCE({self.schema}.chemstore_test_runs.started_at, EXCLUDED.started_at);
            """,
            (
                self.runId,
                self.name,
                "postgres",
                self.tableName,
                "PostgresChemStore run-backed table",
            ),
        )

        cur.execute(
            f"""
            INSERT INTO {self.schema}.chemstore_test_tables
                (run_id, table_name, created_at)
            SELECT %s, %s, now()
            WHERE NOT EXISTS (
                SELECT 1
                FROM {self.schema}.chemstore_test_tables
                WHERE run_id = %s AND table_name = %s
            );
            """,
            (self.runId, self.tableName, self.runId, self.tableName),
        )

    def _create_storage(self):
        with self._connect() as conn:
            with conn.cursor() as cur:
                self._ensure_test_run_registry(cur)

                if self.useRdkitCartridge:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS rdkit;")

                if self.runId is None:
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
                else:
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self._qualified_table()} (
                            run_id TEXT NOT NULL,
                            id TEXT NOT NULL,
                            smiles TEXT NOT NULL,
                            mol MOL,
                            library TEXT,
                            props JSONB DEFAULT '{{}}'::jsonb,
                            PRIMARY KEY (run_id, id)
                        );
                        """
                    )
                    # Existing stable tables created by older versions may not yet have
                    # run_id. Add it so clearRun()/scoped reads can work.
                    cur.execute(
                        f"""
                        ALTER TABLE {self._qualified_table()}
                        ADD COLUMN IF NOT EXISTS run_id TEXT;
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

                if self.runId is not None:
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS idx_{self.tableName}_run_id
                        ON {self._qualified_table()}(run_id);
                        """
                    )
            conn.commit()

    def _row_to_mol(self, row) -> TabularMol:
        mol_id, smiles, library, props = row
        props = props or {}
        props[self.idProp] = mol_id
        props[self.smilesProp] = smiles
        props["library"] = library
        if self.runId is not None:
            props["run_id"] = self.runId
        return TabularMol(mol_id, self.name, smiles, props=props)

    def clear(self):
        """Clear the whole store, or only the current run when run_id is set."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                where_sql, params = self._run_where()
                cur.execute(f"DELETE FROM {self._qualified_table()} {where_sql};",
                            params)
            conn.commit()

    def clearRun(self, run_id: str | None = None):
        """Delete rows belonging to one run_id."""
        run_id = run_id or self.runId

        if not run_id:
            raise ValueError("clearRun() requires run_id or self.runId.")

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    DELETE FROM {self._qualified_table()}
                    WHERE run_id = %s;
                    """,
                    (run_id,),
                )
            conn.commit()

    def save(self) -> str:
        return self.metaFile

    def reload(self):
        return self

    def _insert_sql(self, raise_on_existing: bool) -> str:
        if self.runId is None:
            if raise_on_existing:
                return f"""
                    INSERT INTO {self._qualified_table()}
                        (id, smiles, mol, library, props)
                    VALUES
                        (%s, %s, mol_from_smiles(%s), %s, %s::jsonb);
                """
            return f"""
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

        if raise_on_existing:
            return f"""
                INSERT INTO {self._qualified_table()}
                    (run_id, id, smiles, mol, library, props)
                VALUES
                    (%s, %s, %s, mol_from_smiles(%s), %s, %s::jsonb);
            """
        return f"""
            INSERT INTO {self._qualified_table()}
                (run_id, id, smiles, mol, library, props)
            VALUES
                (%s, %s, %s, mol_from_smiles(%s), %s, %s::jsonb)
            ON CONFLICT (run_id, id) DO UPDATE SET
                smiles = EXCLUDED.smiles,
                mol = EXCLUDED.mol,
                library = EXCLUDED.library,
                props = EXCLUDED.props;
        """

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
        sql = self._insert_sql(raise_on_existing)

        with self._connect() as conn:
            with conn.cursor() as cur:
                for idx, smi in enumerate(smiles):
                    mol_id = self.identifier(smi)

                    row_props = {key: values[idx] for key, values in props.items()}
                    row_props[self.idProp] = mol_id
                    row_props[self.smilesProp] = smi
                    if self.runId is not None:
                        row_props["run_id"] = self.runId

                    if self.runId is None:
                        params = (mol_id, smi, smi, library, json.dumps(row_props))
                    else:
                        params = (
                            self.runId,
                            mol_id,
                            smi,
                            smi,
                            library,
                            json.dumps(row_props),
                        )

                    cur.execute(sql, params)
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

        sql = self._insert_sql(raise_on_existing)

        with self._connect() as conn:
            with conn.cursor() as cur:
                for idx, mol_id in enumerate(ids):
                    smi = smiles_values[idx]
                    row_props = {key: values[idx] for key, values in props.items()}
                    row_props[self.idProp] = mol_id
                    row_props[self.smilesProp] = smi
                    if self.runId is not None:
                        row_props["run_id"] = self.runId

                    if self.runId is None:
                        params = (mol_id, smi, smi, self.name, json.dumps(row_props))
                    else:
                        params = (
                            self.runId,
                            mol_id,
                            smi,
                            smi,
                            self.name,
                            json.dumps(row_props),
                        )

                    cur.execute(sql, params)
            conn.commit()

    def getMol(self, mol_id: str) -> TabularMol:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if self.runId is None:
                    cur.execute(
                        f"""
                        SELECT id, smiles, library, props
                        FROM {self._qualified_table()}
                        WHERE id = %s;
                        """,
                        (mol_id,),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT id, smiles, library, props
                        FROM {self._qualified_table()}
                        WHERE run_id = %s AND id = %s;
                        """,
                        (self.runId, mol_id),
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
                if self.runId is None:
                    cur.execute(
                        f"""
                        DELETE FROM {self._qualified_table()}
                        WHERE id = ANY(%s);
                        """,
                        (ids,),
                    )
                else:
                    cur.execute(
                        f"""
                        DELETE FROM {self._qualified_table()}
                        WHERE run_id = %s AND id = ANY(%s);
                        """,
                        (self.runId, ids),
                    )
            conn.commit()

    def getMolIDs(self) -> tuple[str, ...]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if self.runId is None:
                    cur.execute(
                        f"""
                        SELECT id
                        FROM {self._qualified_table()}
                        ORDER BY id;
                        """
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT id
                        FROM {self._qualified_table()}
                        WHERE run_id = %s
                        ORDER BY id;
                        """,
                        (self.runId,),
                    )
                return tuple(row[0] for row in cur.fetchall())

    def getMolCount(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if self.runId is None:
                    cur.execute(f"SELECT COUNT(*) FROM {self._qualified_table()};")
                else:
                    cur.execute(
                        f"""
                        SELECT COUNT(*)
                        FROM {self._qualified_table()}
                        WHERE run_id = %s;
                        """,
                        (self.runId,),
                    )
                return cur.fetchone()[0]

    def getDF(self) -> pd.DataFrame:
        with self._connect() as conn:
            with conn.cursor() as cur:
                if self.runId is None:
                    cur.execute(
                        f"""
                        SELECT id, smiles, library, props
                        FROM {self._qualified_table()}
                        ORDER BY id;
                        """
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT id, smiles, library, props
                        FROM {self._qualified_table()}
                        WHERE run_id = %s
                        ORDER BY id;
                        """,
                        (self.runId,),
                    )
                rows = cur.fetchall()

        records = []
        for mol_id, smiles, library, props in rows:
            record = props or {}
            record[self.idProp] = mol_id
            record[self.smilesProp] = smiles
            record["library"] = library
            if self.runId is not None:
                record["run_id"] = self.runId
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
            return pd.Series(df[self.smilesProp].values, index=df[self.idProp],
                             name=name)

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
                    if self.runId is None:
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
                    else:
                        cur.execute(
                            f"""
                            UPDATE {self._qualified_table()}
                            SET props = jsonb_set(
                                COALESCE(props, '{{}}'::jsonb),
                                %s,
                                %s::jsonb,
                                true
                            )
                            WHERE run_id = %s AND id = %s;
                            """,
                            ([name], json.dumps(value), self.runId, mol_id),
                        )
            conn.commit()

    def removeProperty(self, name: str):
        with self._connect() as conn:
            with conn.cursor() as cur:
                if self.runId is None:
                    cur.execute(
                        f"""
                        UPDATE {self._qualified_table()}
                        SET props = props - %s;
                        """,
                        (name,),
                    )
                else:
                    cur.execute(
                        f"""
                        UPDATE {self._qualified_table()}
                        SET props = props - %s
                        WHERE run_id = %s;
                        """,
                        (name, self.runId),
                    )
            conn.commit()

    def getSubset(
            self,
            subset: Iterable[str],
            ids: Iterable[str] | None = None,
    ):
        df = self.getDF()

        if ids is None:
            ids = list(df[self.idProp])
        else:
            ids = list(ids)

        return self._clone_with_ids(ids, f"{self.tableName}_subset")

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
                    if self.runId is None:
                        cur.execute(
                            f"""
                            SELECT id, smiles, library, props
                            FROM {self._qualified_table()}
                            ORDER BY id
                            LIMIT %s OFFSET %s;
                            """,
                            (size, offset),
                        )
                    else:
                        cur.execute(
                            f"""
                            SELECT id, smiles, library, props
                            FROM {self._qualified_table()}
                            WHERE run_id = %s
                            ORDER BY id
                            LIMIT %s OFFSET %s;
                            """,
                            (self.runId, size, offset),
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

    def _clone_with_ids(self, ids: Iterable[str],
                        name: str | None = None) -> "PostgresChemStore":
        ids = list(ids)
        subset_name = self._safe_sql_identifier(name or f"{self.tableName}_subset")

        subset_store = PostgresChemStore(
            name=f"{self.name}_subset",
            connection_string=self.connectionString,
            table_name=subset_name,
            schema=self.schema,
            run_id=self.runId,
            smiles_col=self.smilesProp,
            id_col=self.idProp,
            use_rdkit_cartridge=self.useRdkitCartridge,
            create=True,
            standardizer=self.standardizer,
            identifier=self.identifier,
            chunk_size=self.chunkSize,
            n_jobs=self.nJobs,
        )
        subset_store.clear()

        if not ids:
            return subset_store

        with self._connect() as conn:
            with conn.cursor() as cur:
                if self.runId is None:
                    cur.execute(
                        f"""
                        INSERT INTO {subset_store._qualified_table()}
                            (id, smiles, mol, library, props)
                        SELECT id, smiles, mol, library, props
                        FROM {self._qualified_table()}
                        WHERE id = ANY(%s)
                        ON CONFLICT (id) DO UPDATE SET
                            smiles = EXCLUDED.smiles,
                            mol = EXCLUDED.mol,
                            library = EXCLUDED.library,
                            props = EXCLUDED.props;
                        """,
                        (ids,),
                    )
                else:
                    cur.execute(
                        f"""
                        INSERT INTO {subset_store._qualified_table()}
                            (run_id, id, smiles, mol, library, props)
                        SELECT run_id, id, smiles, mol, library, props
                        FROM {self._qualified_table()}
                        WHERE run_id = %s AND id = ANY(%s)
                        ON CONFLICT (run_id, id) DO UPDATE SET
                            smiles = EXCLUDED.smiles,
                            mol = EXCLUDED.mol,
                            library = EXCLUDED.library,
                            props = EXCLUDED.props;
                        """,
                        (self.runId, ids),
                    )
            conn.commit()

        return subset_store

    def searchOnProperty(
            self,
            prop_name: str,
            values: list[float | int | str],
            exact: bool = False,
    ):
        df = self.getDF()

        if prop_name not in df.columns:
            return self._clone_with_ids([], f"{self.tableName}_property_searched")

        if exact:
            matched = df[df[prop_name].isin(values)]
        else:
            mask = pd.Series(False, index=df.index)
            for value in values:
                mask = mask | df[prop_name].astype(str).str.contains(str(value),
                                                                     na=False)
            matched = df[mask]

        ids = list(matched[self.idProp])
        return self._clone_with_ids(ids, f"{self.tableName}_property_searched")

    def searchWithSMARTS(
            self,
            patterns: list[str],
            operator: Literal["or", "and"] = "or",
            use_chirality: bool = False,
            name: str | None = None,
            match_function: Any | None = None,
    ) -> "PostgresChemStore":
        if not self.useRdkitCartridge:
            raise RuntimeError(
                "searchWithSMARTS() requires PostgreSQL with RDKit Cartridge enabled."
            )

        if not patterns:
            raise ValueError("At least one SMARTS pattern must be provided.")

        if operator not in {"or", "and"}:
            raise ValueError("operator must be either 'or' or 'and'.")

        if match_function is not None:
            raise ValueError(
                "match_function is not supported by PostgresChemStore; "
                "SMARTS matching is executed inside PostgreSQL through RDKit Cartridge."
            )

        sql_operator = " OR " if operator == "or" else " AND "
        conditions = sql_operator.join(
            ["mol @> qmol_from_smarts(%s)" for _ in patterns]
        )

        with self._connect() as conn:
            with conn.cursor() as cur:
                for pattern in patterns:
                    cur.execute("SELECT is_valid_smarts(%s);", (pattern,))
                    is_valid = cur.fetchone()[0]
                    if not is_valid:
                        raise ValueError(f"Invalid SMARTS pattern: {pattern!r}")

                if self.runId is None:
                    cur.execute(
                        f"""
                        SELECT id
                        FROM {self._qualified_table()}
                        WHERE {conditions}
                        ORDER BY id;
                        """,
                        tuple(patterns),
                    )
                else:
                    cur.execute(
                        f"""
                        SELECT id
                        FROM {self._qualified_table()}
                        WHERE run_id = %s AND ({conditions})
                        ORDER BY id;
                        """,
                        tuple([self.runId, *patterns]),
                    )
                ids = [row[0] for row in cur.fetchall()]

        subset_name = name or f"{self.tableName}_smarts_searched"
        return self._clone_with_ids(ids, subset_name)

    def __len__(self):
        return self.getMolCount()

    def __contains__(self, item):
        return item in self.getMolIDs()

    def __getitem__(self, item):
        return self.getMol(item)

    def getSummary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "name": [self.name],
            "table_name": [self.tableName],
            "schema": [self.schema],
            "run_id": [self.runId],
            "smiles_col": [self.smilesProp],
            "id_col": [self.idProp],
            "use_rdkit_cartridge": [self.useRdkitCartridge],
            "mol_count": [self.getMolCount()],
        })
