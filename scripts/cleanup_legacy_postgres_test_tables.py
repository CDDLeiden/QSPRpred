"""Clean legacy PostgreSQL ChemStore test tables created with long per-run names.

The newer test layout uses short stable tables (for example cst_addmols,
cst_search, cst_subset) plus a run registry in chemstore_test_runs. This script
removes old tables such as:

    chemstore_postgrestabularstoragetest_testaddmols_202606...
    chemstore_postgrestabularstoragetest_testsearch_202606...

It is safe by default: without --apply it only prints the DROP statements.
"""

from __future__ import annotations

import argparse
import os
from dotenv import load_dotenv
import psycopg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Actually drop the tables.")
    parser.add_argument(
        "--schema",
        default=None,
        help="PostgreSQL schema. Defaults to QSPR_POSTGRES_SCHEMA or public.",
    )
    args = parser.parse_args()

    load_dotenv()
    dsn = os.getenv("QSPR_POSTGRES_DSN")
    if not dsn:
        raise RuntimeError("QSPR_POSTGRES_DSN is not set in the environment or .env file.")

    schema = args.schema or os.getenv("QSPR_POSTGRES_SCHEMA", "public")

    patterns = [
        "chemstore_postgrestabularstoragetest_%",
        "chemstore_postgres_tabular_storage_test_%",
        "qsprmodel_training_%_molecules",
    ]

    with psycopg.connect(dsn, connect_timeout=10) as conn:
        with conn.cursor() as cur:
            like_sql = " OR ".join(["tablename LIKE %s" for _ in patterns])
            cur.execute(
                f"""
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = %s
                  AND ({like_sql})
                ORDER BY tablename;
                """,
                (schema, *patterns),
            )
            tables = [row[0] for row in cur.fetchall()]

            if not tables:
                print("No legacy PostgreSQL ChemStore test tables found.")
                return

            print("Legacy tables found:")
            for table in tables:
                print(f"  {schema}.{table}")

            if not args.apply:
                print("\nDry run only. To drop these tables, run:")
                print("  python scripts/cleanup_legacy_postgres_test_tables.py --apply")
                return

            for table in tables:
                cur.execute(f'DROP TABLE IF EXISTS "{schema}"."{table}" CASCADE;')
            conn.commit()

            print(f"Dropped {len(tables)} legacy table(s).")


if __name__ == "__main__":
    main()
