import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from Backend11 import Base, Condition, Medicine
from database import SessionLocal, engine

Base.metadata.create_all(bind=engine)

df = pd.read_csv("dummy_data.csv")


def guess_min_severity(severity):
    severity = severity.strip().lower()
    if severity in ["high", "severe"]:
        return "high"
    if severity in ["medium", "moderate"]:
        return "medium"
    return "low"


def seed_from_dummy_data():
    db: Session = SessionLocal()
    try:
        print(f"Inserting rows from dummy_data.csv: {len(df)}")
        inserted_conditions = set()
        inserted_meds = 0
        skipped_meds = 0

        for _, row in df.iterrows():
            condition_name = row["condition"].strip()
            medicine_name = row["medicine"].strip()

            condition = db.query(Condition).filter(Condition.name == condition_name).first()
            if not condition:
                condition = Condition(name=condition_name)
                db.add(condition)
                db.commit()
                db.refresh(condition)
                inserted_conditions.add(condition_name)

            existing_med = db.query(Medicine).filter(
                Medicine.name == medicine_name,
                Medicine.condition_id == condition.id,
            ).first()

            if existing_med:
                skipped_meds += 1
                continue

            db.add(
                Medicine(
                    name=medicine_name,
                    base_dosage="500mg",
                    is_generic=False,
                    cost=round(10 + 90 * np.random.rand(), 2),
                    min_severity=guess_min_severity(row["severity"]),
                    condition_id=condition.id,
                )
            )
            inserted_meds += 1

        db.commit()
        print(
            f"Completed. Conditions added: {len(inserted_conditions)} | "
            f"Medicines inserted: {inserted_meds} | Skipped: {skipped_meds}"
        )
    finally:
        db.close()


if __name__ == "__main__":
    seed_from_dummy_data()
