# Storage tests with Pandas/PostgreSQL backends

This patch keeps the original Pandas storage tests and adds a PostgreSQL/RDKit backend test class.

## Default Pandas tests

```bash
python -m pytest qsprpred/data/storage/tests.py
```

If the optional `papyrus_structure_pipeline` package is not installed, the original Pandas tests are skipped during collection instead of failing on import.

## PostgreSQL tests

Requires `.env` with:

```env
QSPR_POSTGRES_DSN=postgresql://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require
QSPR_POSTGRES_SCHEMA=public
QSPR_USE_RDKIT=true
```

Run:

```bash
python -m pytest qsprpred/data/storage/tests.py -k PostgresTabularStorageTest
```

The PostgreSQL test class creates a unique table per test method and registers test metadata in:

- `chemstore_test_runs`
- `chemstore_test_tables`

The generated test tables are intentionally not dropped so that test runs can be inspected later.
