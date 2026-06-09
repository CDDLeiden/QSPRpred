# PostgreSQL / RDKit ChemStore backend

This branch adds an experimental PostgreSQL-backed implementation of the QSPRPred
`ChemStore` interface. The implementation stores molecules in PostgreSQL and can use
the RDKit PostgreSQL Cartridge for server-side chemical substructure search.

## 1. Create a Neon PostgreSQL database

1. Go to Neon and create a new project.
2. Choose a recent PostgreSQL version.
3. Open the project dashboard and copy the connection string.
4. The connection string should look like this:

```bash
postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require
```

The password is part of the connection string. Do not commit the real connection
string into Git.

## 2. Configure local environment

Create a local `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env`:

```env
QSPR_POSTGRES_DSN=postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require
QSPR_POSTGRES_SCHEMA=public
QSPR_POSTGRES_TABLE=test_molecules
QSPR_USE_RDKIT=true
```

The `.env` file must stay local. Keep `.env.example` in Git, but never commit `.env`.

Recommended `.gitignore` entries:

```gitignore
.env
.env.*
!.env.example
```

## 3. Install dependencies

Activate the project virtual environment first:

```bash
source .venv/bin/activate
```

Install the PostgreSQL driver and `.env` loader:

```bash
pip install "psycopg[binary]" python-dotenv
```

## 4. Enable and test RDKit Cartridge

The RDKit PostgreSQL Cartridge is enabled as a PostgreSQL extension:

```sql
CREATE EXTENSION IF NOT EXISTS rdkit;
```

The helper script checks this automatically:

```bash
python scripts/test_rdkit.py
```

Expected output:

```text
Checking RDKit extension...
RDKit extension installed.
RDKit test OK:
('CCO',)
```

## 5. Initialize the storage table

```bash
python scripts/init_postgres_chemstore.py
```

This creates a table similar to:

```sql
CREATE TABLE IF NOT EXISTS molecules (
    id TEXT PRIMARY KEY,
    smiles TEXT NOT NULL,
    mol MOL,
    library TEXT,
    props JSONB DEFAULT '{}'::jsonb
);
```

## 6. Run the manual integration test

```bash
python scripts/test_postgres_chemstore.py
```

This test verifies:

- connection to PostgreSQL,
- table creation,
- insertion through `addMols()`,
- retrieval through `getMol()`, `getMolIDs()`, `getMolCount()`,
- property export through `getProperty()` and `getDF()`,
- chunking through `iterChunks()`,
- deletion through `removeMol()`,
- RDKit Cartridge SMARTS search through `searchWithSMARTS()`.

## 7. Run pytest integration tests

The pytest tests are skipped automatically if `QSPR_POSTGRES_DSN` is not configured.

```bash
pytest testing/storage/test_postgres_chemstore.py
```

## 8. SMARTS search

`PostgresChemStore.searchWithSMARTS()` uses the RDKit Cartridge substructure search
operator `@>`:

```sql
SELECT id
FROM molecules
WHERE mol @> mol_from_smarts('c1ccccc1');
```

For multiple patterns, the method supports:

```python
store.searchWithSMARTS(["N", "c1ccccc1"], operator="or")
store.searchWithSMARTS(["N", "c1ccccc1"], operator="and")
```

The method returns a new `PostgresChemStore` backed by a materialized subset table,
so the result can be filtered further.

## 9. Current limitations

- `use_chirality` is accepted for API compatibility, but the current SQL path does
  not expose a separate chirality flag.
- `match_function` is not supported because matching is executed inside PostgreSQL.
- Subset results are materialized into physical PostgreSQL tables.
- The implementation is still experimental and should be cleaned up before merging.
