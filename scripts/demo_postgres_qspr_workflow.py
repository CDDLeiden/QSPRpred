import os
import uuid

import pandas as pd
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor

from qsprpred.data.descriptors.fingerprints import MorganFP
from qsprpred.data.tables.qspr import QSPRTable
from qsprpred.extra.data.storage.postgres import PostgresChemStore
from qsprpred.tasks import TargetTasks, TargetSpec

load_dotenv()


def main():
    dsn = os.getenv("QSPR_POSTGRES_DSN")
    schema = os.getenv("QSPR_POSTGRES_SCHEMA", "public")

    if not dsn:
        raise RuntimeError("Missing QSPR_POSTGRES_DSN in .env")

    run_id = f"demo_qspr_{uuid.uuid4().hex[:8]}"
    table_name = "cst_demo_qspr"

    print(f"Run ID: {run_id}")
    print(f"Using table: {schema}.{table_name}")

    store = PostgresChemStore(
        name=run_id,
        connection_string=dsn,
        table_name=table_name,
        schema=schema,
        use_rdkit_cartridge=True,
        create=True,
        run_id=run_id,
    )

    store.clearRun(run_id)

    df = pd.DataFrame(
        {
            "SMILES": [
                "CCO",
                "CCN",
                "CCC",
                "CCCC",
                "CCCO",
                "CCCN",
                "c1ccccc1",
                "Cc1ccccc1",
                "CC(=O)O",
                "CC(C)O",
            ],
            "activity": [1.0, 1.2, 1.5, 2.0, 2.2, 2.4, 3.0, 3.3, 1.8, 2.1],
        }
    )

    dataset = QSPRTable.fromDF(
        name="demo_qspr_postgres",
        df=df,
        path=".",
        smiles_col="SMILES",
        target_props=[
            TargetSpec(
                name="activity",
                task=TargetTasks.REGRESSION,
            )
        ],
        storage=store,
    )

    dataset.prepareDataset(
        feature_calculators=[
            MorganFP(radius=2, nBits=128),
        ],
        recalculate_features=True,
    )

    model = RandomForestRegressor(
        n_estimators=50,
        random_state=42,
    )

    model.fit(dataset.X, dataset.y.values.ravel())
    preds = model.predict(dataset.X)

    print("\nPredictions:")
    for smiles, y_true, y_pred in zip(df["SMILES"], df["activity"], preds):
        print(f"{smiles:12s} true={y_true:.2f} predicted={y_pred:.2f}")

    print("\nStored molecules:")
    print(store.getDF()[["SMILES", "activity"]])

    print("\nDone.")


if __name__ == "__main__":
    main()
