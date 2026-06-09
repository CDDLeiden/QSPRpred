# PostgreSQL ChemStore test-run registry

This patch replaces long per-run PostgreSQL test table names with short stable tables:

- `cst_addmols`
- `cst_molprocess`
- `cst_subset`
- `cst_search`

Each PostgreSQL test run is registered in `chemstore_test_runs`. Rows written by the `PostgresChemStore` test backend include `run_id`, so multiple runs can coexist in the same stable tables.

## Run Pandas tests

```bash
python -m pytest qsprpred/data/storage/tests.py
```

## Run PostgreSQL storage tests

```bash
python -m pytest qsprpred/data/storage/tests.py -k PostgresTabularStorageTest
```

## Inspect runs

```sql
SELECT *
FROM chemstore_test_runs
ORDER BY started_at DESC;
```

```sql
SELECT *
FROM chemstore_test_tables
ORDER BY created_at DESC;
```

## Delete old long-name legacy tables

Dry run:

```bash
python scripts/cleanup_legacy_postgres_test_tables.py
```

Actually drop them:

```bash
python scripts/cleanup_legacy_postgres_test_tables.py --apply
```
