import os

from dotenv import load_dotenv

from qsprpred.extra.data.storage.postgres import PostgresChemStore

load_dotenv()

dsn = os.getenv("QSPR_POSTGRES_DSN")
schema = os.getenv("QSPR_POSTGRES_SCHEMA", "public")
table_name = os.getenv("QSPR_POSTGRES_TABLE", "test_molecules")
use_rdkit = os.getenv("QSPR_USE_RDKIT", "true").lower() == "true"

if not dsn:
    raise RuntimeError("Missing QSPR_POSTGRES_DSN in .env")

print("Creating PostgresChemStore...")

store = PostgresChemStore(
    name="test_chemstore",
    connection_string=dsn,
    table_name=table_name,
    schema=schema,
    use_rdkit_cartridge=use_rdkit,
    create=True,
)

print("Store created.")

print("Clearing table...")
store.clear()
print("Table cleared.")

print("Adding molecules...")
mols = store.addMols(
    smiles=["CCO", "CCN", "c1ccccc1", "Cc1ccccc1"],
    props={
        "chem_name": ["ethanol", "ethylamine", "benzene", "toluene"],
        "source": ["test", "test", "test", "test"],
        "score": [1.1, 2.2, 3.3, 4.4],
    },
    raise_on_existing=False,
)
print("Molecules added.")

print("Inserted molecules:")
for mol in mols:
    print("-", mol)

print("Getting molecule count...")
print(store.getMolCount())

print("Getting molecule IDs...")
ids = store.getMolIDs()
print(ids)

print("Getting property chem_name...")
print(store.getProperty("chem_name"))

print("Getting DataFrame...")
print(store.getDF())

print("Testing chunks...")
for chunk in store.iterChunks(size=2, chunk_type="df"):
    print(chunk)

if use_rdkit:
    print("Testing SMARTS search: aromatic ring c1ccccc1...")
    aromatic = store.searchWithSMARTS(
        ["c1ccccc1"],
        name="test_molecules_manual_aromatic",
    )
    print(aromatic.getDF())
    print("Aromatic count:", aromatic.getMolCount())

    print("Testing SMARTS search: N OR aromatic ring...")
    nitrogen_or_aromatic = store.searchWithSMARTS(
        ["N", "c1ccccc1"],
        operator="or",
        name="test_molecules_manual_n_or_aromatic",
    )
    print(nitrogen_or_aromatic.getDF())
    print("N OR aromatic count:", nitrogen_or_aromatic.getMolCount())

print("Loading first molecule...")
first_id = ids[0]
print(store.getMol(first_id))

print("Removing first molecule...")
store.removeMol(first_id)
print("Removed.")

print("Final count:")
print(store.getMolCount())

print("Test finished OK.")
