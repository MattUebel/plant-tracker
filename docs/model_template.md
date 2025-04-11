# Model Template Guidelines

This document outlines the structure and conventions to follow when creating models for the Plant Tracker application.

## Basic Structure

All models should follow this basic structure:

```python
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Date
from sqlalchemy.sql import func
from database import Base

class ModelName(Base):
    __tablename__ = "table_name"
    
    # Primary Key
    id = Column(Integer, primary_key=True, index=True)
    
    # Core Fields
    name = Column(String(100), nullable=False)  # All models should have a name
    
    # Model-specific fields...
    
    # Optional relationships
    # related_items = relationship("RelatedModel", back_populates="this_model")
    
    # Metadata - ALWAYS include these
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
```

## Naming Conventions

1. **Model Class Names**: Use PascalCase singular nouns (e.g., `SeedPacket`, not `seed_packets` or `SeedPackets`)
2. **Table Names**: Use snake_case plural nouns (e.g., `seed_packets`, not `seedpacket` or `seed_packet`)
3. **Column Names**: Use snake_case (e.g., `germination_rate`, not `germinationRate`)

## Field Types and Usage

### Common Field Types

- `String`: For short text up to 255 characters. Always specify a length, e.g., `String(100)`.
- `Text`: For longer text content with no length limits.
- `Integer`: For whole numbers.
- `Float`: For decimal numbers.
- `Boolean`: For true/false values.
- `Date`: For date values without time.
- `DateTime`: For date and time values.
- `Enum`: For fields with a fixed set of possible values.

### Examples from SeedPacket Model

The `SeedPacket` model demonstrates these principles:

```python
class SeedPacket(Base):
    __tablename__ = "seed_packets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    variety = Column(String(100))
    brand = Column(String(100))
    
    # Key fields with appropriate types
    germination_rate = Column(Float)
    germ_date = Column(Date)
    maturity = Column(Integer)
    growth = Column(String(100))
    seed_depth = Column(Float)
    spacing = Column(Float)
    
    # Additional fields
    quantity = Column(Integer)
    purchase_date = Column(Date)
    expiration_date = Column(Date)
    notes = Column(Text)
    
    # Standard metadata
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
```

## Best Practices

1. Always include comments for fields that may not be self-explanatory
2. Use `nullable=False` for required fields
3. Add appropriate indices for fields that will be frequently queried
4. Include appropriate relationship definitions when models are related
5. Always include the `created_at` and `updated_at` metadata fields for tracking
6. Use type annotations in your model methods for better code readability

## Database Migrations

After creating or updating a model, remember to generate and apply a migration:

```bash
./migrate.sh migrate "describe_your_changes"
./migrate.sh upgrade
```