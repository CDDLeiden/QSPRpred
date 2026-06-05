import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

dsn = os.getenv("QSPR_POSTGRES_DSN")

if not dsn:
    raise RuntimeError("QSPR_POSTGRES_DSN is not set")

print("Connecting to PostgreSQL...")

with psycopg.connect(dsn) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT current_database();")
        print("Database:", cur.fetchone()[0])

        cur.execute("SELECT version();")
        print("Version:", cur.fetchone()[0])

print("Connection OK")