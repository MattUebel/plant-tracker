from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Float,
    Date,
    ForeignKey,
    JSON,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base
from models.image import Image


class Planting(Base):
    __tablename__ = "plantings"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)

    # Relationships
    seed_id = Column(
        Integer, ForeignKey("seeds.id", ondelete="SET NULL"), nullable=True
    )
    seed = relationship("Seed", back_populates="plantings")

    # Planting specific fields
    expected_germination_time = Column(
        Integer
    )  # Expected days to germinate, from seed data
    actual_germination_time = Column(Integer)  # Actual days to germinate, as observed
    expected_maturity_time = Column(
        Integer
    )  # Expected days to maturity, from seed data
    actual_maturity_time = Column(Integer)  # Actual days to maturity, as observed
    harvest_events = Column(
        JSON
    )  # Array of {date, quantity, units, notes} for harvest events
    seeds_planted = Column(Integer)  # Number of seeds planted
    successful_plants = Column(Integer)  # Number of plants that successfully grew
    planting_date = Column(Date)  # When the seeds were planted
    notes = Column(Text)  # Additional notes

    # Metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships - make sure this matches the foreign_keys used in the Image model
    images = relationship(
        "Image",
        back_populates="planting",
        foreign_keys="[Image.planting_id]",
        lazy="joined",
        cascade="all, delete-orphan",
    )

    # Relationship to transplant events - new proper relationship
    transplant_events = relationship(
        "TransplantEvent",
        back_populates="planting",
        lazy="joined",
        cascade="all, delete-orphan",
        order_by="TransplantEvent.date",
    )
