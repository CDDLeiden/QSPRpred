# PostgreSQL RDKit ChemStore for QSPRPred

## Overview

This branch adds an experimental PostgreSQL-backed implementation of the QSPRPred `ChemStore` interface.

The backend stores molecules in PostgreSQL and supports server-side chemical structure operations through the RDKit PostgreSQL Cartridge.

Current implementation includes:

* PostgreSQL ChemStore backend
* RDKit Cartridge integration
* SMARTS substructure search
* QSPRTable integration
* QSPRModel integration
* RandomForest training workflow
* Stable PostgreSQL test tables
* Test-run registry
* Neon PostgreSQL support

---

# Architecture

The backend stores molecules in PostgreSQL tables instead of Pandas DataFrames.

Main components:

```text
PostgresChemStore
        ↓
MoleculeTable
        ↓
QSPRTable
        ↓
Descriptors (MorganFP, etc.)
        ↓
Machine Learning Models
```

RDKit Cartridge enables server-side chemical operations:

```text
SMILES
   ↓
RDKit MOL
   ↓
SMARTS search
```

---

# Neon PostgreSQL Setup

Create a Neon account:

https://neon.tech

Create a PostgreSQL project and obtain the connection string:

```text
postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require
```

Do not commit real credentials.

---

# Environment Configuration

Create:

```bash
cp .env.example .env
```

Example:

```env
QSPR_POSTGRES_DSN=postgresql://USER:PASSWORD@HOST/DATABASE?sslmode=require
QSPR_POSTGRES_SCHEMA=public
QSPR_POSTGRES_TABLE=molecules
QSPR_USE_RDKIT=true
```

Recommended `.gitignore`:

```gitignore
.env
.env.*
!.env.example
```

---

# Required Packages

Activate virtual environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install "psycopg[binary]" python-dotenv
```

---

# RDKit PostgreSQL Cartridge

Enable RDKit:

```sql
CREATE EXTENSION IF NOT EXISTS rdkit;
```

Verify installation:

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

---

# PostgreSQL ChemStore

Initialize storage:

```bash
python scripts/init_postgres_chemstore.py
```

Typical schema:

```sql
CREATE TABLE molecules (
    id TEXT PRIMARY KEY,
    smiles TEXT NOT NULL,
    mol MOL,
    library TEXT,
    props JSONB DEFAULT '{}'::jsonb
);
```

---

# Test Run Registry

Stable PostgreSQL-backed tests use a run registry.

Tables:

```text
chemstore_test_runs
chemstore_test_tables
```

Every execution receives a unique:

```text
run_id
```

allowing multiple runs to coexist safely.

Stable test tables:

```text
cst_addmols
cst_molprocess
cst_search
cst_subset
cst_qsprmodel_training
cst_demo_qspr
```

Rows are linked to test runs through:

```text
run_id
```

---

# Running Tests

## Manual ChemStore Test

```bash
python scripts/test_postgres_chemstore.py
```

Validates:

* connection
* table creation
* addMols()
* getMol()
* getMolCount()
* getMolIDs()
* getProperty()
* getDF()
* iterChunks()
* removeMol()
* SMARTS search

---

## MoleculeTable Integration

```bash
python -m pytest testing/storage/test_molecule_table_postgres.py
```

---

## Model Integration

```bash
python -m pytest testing/storage/test_model_postgres_storage.py
```

---

## QSPRTable Workflow

```bash
python -m pytest testing/storage/test_qsprtable_postgres_workflow.py
```

---

## QSPRModel Training Workflow

```bash
python -m pytest testing/storage/test_qsprmodel_training_postgres.py
```

---

## Original Pandas Storage Tests

```bash
python -m pytest qsprpred/data/storage/tests.py
```

---

## PostgreSQL Storage Tests

```bash
python -m pytest qsprpred/data/storage/tests.py -k PostgresTabularStorageTest -v
```

---

# Demo Workflow

Run complete end-to-end demonstration:

```bash
python scripts/demo_postgres_qspr_workflow.py
```

Workflow:

```text
PostgresChemStore
      ↓
QSPRTable
      ↓
MorganFP
      ↓
RandomForest
      ↓
Predictions
```

The script:

* stores molecules in PostgreSQL
* computes descriptors
* trains a RandomForest model
* generates predictions
* records execution metadata

---

# Inspecting Test Runs

Recent runs:

```sql
SELECT *
FROM chemstore_test_runs
ORDER BY started_at DESC;
```

Registered tables:

```sql
SELECT *
FROM chemstore_test_tables
ORDER BY created_at DESC;
```

---

# Cleanup Legacy Test Tables

Preview:

```bash
python scripts/cleanup_legacy_postgres_test_tables.py
```

Apply cleanup:

```bash
python scripts/cleanup_legacy_postgres_test_tables.py --apply
```

---

# Current Status

Implemented:

* PostgreSQL ChemStore
* RDKit Cartridge integration
* SMARTS search
* MoleculeTable support
* QSPRTable support
* QSPRModel support
* Test-run registry
* Stable PostgreSQL test tables
* Demo workflow

Current maturity:

```text
Research Ready
Beta Ready
```

Recommended next steps:

* larger benchmark datasets
* performance profiling
* descriptor caching
* production deployment evaluation
* full integration into the main QSPRPred branch

---

# Typical Validation Sequence

```bash
python scripts/test_connection.py
python scripts/test_rdkit.py
python scripts/test_postgres_chemstore.py

python -m pytest testing/storage/test_molecule_table_postgres.py
python -m pytest testing/storage/test_model_postgres_storage.py
python -m pytest testing/storage/test_qsprtable_postgres_workflow.py
python -m pytest testing/storage/test_qsprmodel_training_postgres.py

python -m pytest qsprpred/data/storage/tests.py -k PostgresTabularStorageTest -v

python scripts/demo_postgres_qspr_workflow.py
```

If all commands complete successfully, the PostgreSQL backend is fully operational.
