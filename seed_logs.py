# seed_logs.py
import random
from datetime import datetime

from sqlalchemy.orm import Session

from database import Base, SessionLocal, engine
from Backend11 import AIPredictionLog

Base.metadata.create_all(bind=engine)

medicines = [
    {"name": "Medicinex", "dosage": "10mg", "cost": 12.5, "is_generic": False},
    {"name": "ColdAway", "dosage": "5mg", "cost": 4.0, "is_generic": True},
    {"name": "PainLess", "dosage": "20mg", "cost": 9.9, "is_generic": False},
    {"name": "HealFast", "dosage": "7.5mg", "cost": 6.0, "is_generic": True},
    {"name": "CureAll", "dosage": "15mg", "cost": 8.0, "is_generic": True},
]

conditions = ["Cold", "Fever", "Headache"]
severities = ["low", "medium", "high"]


def seed_logs():
    db: Session = SessionLocal()
    try:
        for _ in range(20):
            med = random.choice(medicines)
            db.add(
                AIPredictionLog(
                    symptoms=random.choice(["cough", "fever", "sore throat"]),
                    age=random.randint(18, 65),
                    weight=round(random.uniform(50, 90), 1),
                    severity=random.choice(severities),
                    condition=random.choice(conditions),
                    predicted_medicine=med["name"],
                    predicted_dosage=med["dosage"],
                    predicted_cost=med["cost"],
                    is_generic=med["is_generic"],
                    confidence=round(random.uniform(0.7, 0.99), 2),
                    timestamp=datetime.utcnow(),
                )
            )

        db.commit()
        print("Seeded 20 dummy logs into AIPredictionLog.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_logs()
