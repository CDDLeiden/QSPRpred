import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

dsn = os.getenv("QSPR_POSTGRES_DSN")
schema = os.getenv("QSPR_POSTGRES_SCHEMA", "public")
table = os.getenv("QSPR_POSTGRES_TABLE", "test_molecules")

sql = f"""
CREATE EXTENSION IF NOT EXISTS rdkit;

CREATE TABLE IF NOT EXISTS {schema}.{table} (
    id TEXT PRIMARY KEY,
    smiles TEXT NOT NULL,
    mol MOL,
    library TEXT,
    props JSONB DEFAULT '{{}}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_{table}_library
ON {schema}.{table}(library);

CREATE INDEX IF NOT EXISTS idx_{table}_props
ON {schema}.{table} USING GIN(props);
"""

with psycopg.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

print(f"Initialized {schema}.{table}")