from app.core.database import engine, Base
from app.models.sql import User, PatientReport

print("Resetting database schema...")
try:
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    print("Database reset successful. All tables recreated.")
except Exception as e:
    print(f"Error resetting database: {e}")
