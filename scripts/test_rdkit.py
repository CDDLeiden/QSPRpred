import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

dsn = os.getenv("QSPR_POSTGRES_DSN")

with psycopg.connect(dsn) as conn:
    with conn.cursor() as cur:

        print("Checking RDKit extension...")

        try:
            cur.execute("""
                CREATE EXTENSION IF NOT EXISTS rdkit;
            """)
            conn.commit()

            print("RDKit extension installed.")

            cur.execute("""
                SELECT mol_from_smiles('CCO');
            """)

            result = cur.fetchone()

            print("RDKit test OK:")
            print(result)

        except Exception as e:
            print("RDKit test FAILED:")
            print(e)