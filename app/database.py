import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

# Convert standard PostgreSQL URL to AsyncPG format
db_url = os.getenv("DATABASE_URL", "postgresql://plant_user:plant_password@db:5432/plant_tracker")
async_db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(async_db_url, echo=bool(os.getenv("DEBUG", "False").lower() == "true"))
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()

async def get_db():
    """Dependency for getting async DB session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()