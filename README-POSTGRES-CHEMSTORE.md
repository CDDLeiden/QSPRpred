# Phase 1: PostgresChemStore

Files to copy into the project:

- `qsprpred/data/storage/postgres/__init__.py`
- `qsprpred/data/storage/postgres/chem_store.py`
- `qsprpred/data/storage/postgres/schema.sql`

Install dependency:

```bash
pip install "psycopg[binary]"
```

Minimal usage:

```python
from qsprpred.data.storage.postgres import PostgresChemStore

store = PostgresChemStore(
    name="chem_store",
    connection_string="postgresql://USER:PASSWORD@HOST:5432/DB?sslmode=require",
    use_rdkit_cartridge=True,
)

mols = store.addMols(
    ["CCO", "c1ccccc1"],
    props={"compound_name": ["ethanol", "benzene"]},
    raise_on_existing=False,
)

print(store.getMolCount())
print(store.getMolIDs())
print(store.getDF())
```

Note: `use_rdkit_cartridge=True` tries to run `CREATE EXTENSION IF NOT EXISTS rdkit` and create a generated `mol` column. If the cloud PostgreSQL provider does not allow this extension, the implementation falls back to a plain PostgreSQL table with JSONB properties.
