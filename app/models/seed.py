from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Date
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base
from models.image import Image


def round_to_nearest_fraction(value: float) -> float:
    """Round a decimal to the nearest standard fraction of an inch (1/8, 1/4, 1/2)"""
    # Convert to eighths (smallest fraction we support)
    eighths = round(value * 8) / 8
    return round(eighths, 3)  # Round to 3 decimal places to avoid floating point issues


class Seed(Base):
    __tablename__ = "seeds"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    variety = Column(String(100))
    brand = Column(String(100))

    # Key fields as specified
    _seed_depth = Column("seed_depth", Float)  # Planting depth in inches
    spacing = Column(Float)  # Spacing between plants in inches

    # Additional useful fields
    notes = Column(Text)

    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    images = relationship(
        "Image", back_populates="seed", lazy="joined", cascade="all, delete-orphan"
    )
    plantings = relationship("Planting", back_populates="seed", lazy="select")

    @property
    def seed_depth(self) -> Float:
        """Get the seed depth"""
        return self._seed_depth

    @seed_depth.setter
    def seed_depth(self, value: float):
        """Set the seed depth, automatically rounding to nearest standard fraction"""
        if value is not None:
            self._seed_depth = round_to_nearest_fraction(value)
        else:
            self._seed_depth = None
