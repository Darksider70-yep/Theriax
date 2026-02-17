# seed_conditions_medicines.py
from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from Backend11 import Condition, Medicine

Base.metadata.create_all(bind=engine)

conditions_data = [
    "Cold",
    "Fever",
    "Headache",
    "Allergy",
    "Flu",
]

medicines_data = [
    {"name": "Medicinex", "dosage": "10mg", "cost": 12.5, "is_generic": False, "min_severity": "high", "condition": "Fever"},
    {"name": "ColdAway", "dosage": "5mg", "cost": 4.0, "is_generic": True, "min_severity": "low", "condition": "Cold"},
    {"name": "PainLess", "dosage": "20mg", "cost": 9.9, "is_generic": False, "min_severity": "medium", "condition": "Headache"},
    {"name": "HealFast", "dosage": "7.5mg", "cost": 6.0, "is_generic": True, "min_severity": "low", "condition": "Flu"},
    {"name": "CureAll", "dosage": "15mg", "cost": 8.0, "is_generic": True, "min_severity": "medium", "condition": "Allergy"},
]


def seed_conditions_and_medicines():
    db: Session = SessionLocal()
    try:
        condition_map = {}
        for name in conditions_data:
            condition = db.query(Condition).filter_by(name=name).first()
            if not condition:
                condition = Condition(name=name)
                db.add(condition)
                db.commit()
                db.refresh(condition)
            condition_map[name] = condition

        for med in medicines_data:
            existing = db.query(Medicine).filter_by(name=med["name"]).first()
            if existing:
                continue

            db.add(
                Medicine(
                    name=med["name"],
                    base_dosage=med["dosage"],
                    cost=med["cost"],
                    is_generic=med["is_generic"],
                    min_severity=med["min_severity"],
                    condition_id=condition_map[med["condition"]].id,
                )
            )

        db.commit()
        print("Seeded conditions and medicines.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_conditions_and_medicines()
