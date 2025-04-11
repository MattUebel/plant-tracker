from sqlalchemy import Column, Integer, String, Text, DateTime, Date, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base


class TransplantEvent(Base):
    __tablename__ = "transplant_events"

    id = Column(Integer, primary_key=True, index=True)

    # Foreign key to planting
    planting_id = Column(
        Integer, ForeignKey("plantings.id", ondelete="CASCADE"), nullable=False
    )

    # TransplantEvent specific fields
    date = Column(Date, nullable=False)  # When the transplant occurred
    location = Column(String(255))  # Where the plant was moved to
    container = Column(String(255))  # Container type used
    description = Column(Text)  # Additional notes about the transplant

    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationship back to planting
    planting = relationship("Planting", back_populates="transplant_events")
