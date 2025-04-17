from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from database import Base


class Image(Base):
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    entity_type = Column(
        String(50), nullable=False
    )  # Type of entity (e.g., 'seed', 'plant')
    entity_id = Column(Integer, nullable=False)  # ID of the related entity
    filename = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=True)
    mime_type = Column(String(100), nullable=True)
    file_size = Column(Integer, nullable=True)  # Size in bytes
    structured_data = Column(JSON, nullable=True)  # Structured data extracted from OCR
    seed_id = Column(Integer, ForeignKey("seeds.id", ondelete="CASCADE"), nullable=True)
    planting_id = Column(
        Integer, ForeignKey("plantings.id", ondelete="CASCADE"), nullable=True
    )

    # Standard metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

    # Relationships - ensure back_populates attributes match the relationship names in the related models
    seed = relationship("Seed", back_populates="images", foreign_keys=[seed_id])
    planting = relationship(
        "Planting", back_populates="images", foreign_keys=[planting_id]
    )
