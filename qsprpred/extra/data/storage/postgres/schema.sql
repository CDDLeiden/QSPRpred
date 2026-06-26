-- PostgreSQL schema for PostgresChemStore.
-- RDKit cartridge is optional, but recommended for chemical indexing/search.

CREATE EXTENSION IF NOT EXISTS rdkit;

CREATE TABLE IF NOT EXISTS molecules (
    id TEXT PRIMARY KEY,
    library TEXT NOT NULL,
    smiles TEXT NOT NULL,
    original_smiles TEXT,
    props JSONB NOT NULL DEFAULT '{}'::jsonb,
    mol MOL GENERATED ALWAYS AS (mol_from_smiles(smiles::cstring)) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_molecules_library
ON molecules(library);

CREATE INDEX IF NOT EXISTS idx_molecules_props_gin
ON molecules USING GIN(props);

CREATE INDEX IF NOT EXISTS idx_molecules_mol_gist
ON molecules USING GIST(mol);
