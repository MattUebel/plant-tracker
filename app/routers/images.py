from fastapi import APIRouter, Depends, HTTPException, Request, File, UploadFile, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import get_db
from models.seed import Seed
from models.image import Image
from utils.image_processor import ImageProcessor
from typing import List, Optional
import os
import json

# Fix the import path to match the Docker container structure
import sys
# Import the templates from proper location for Docker
from utils.templates import templates

router = APIRouter()
image_processor = ImageProcessor()

# Create upload directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

@router.get("/{entity_type}/{entity_id}/images", response_class=HTMLResponse)
async def list_entity_images(
    entity_type: str, 
    entity_id: int, 
    request: Request, 
    db: AsyncSession = Depends(get_db)
):
    """List all images for a specific entity"""
    # Verify entity exists
    entity = await get_entity(db, entity_type, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"{entity_type.title()} not found")
    
    # Get all images for this entity
    result = await db.execute(
        select(Image)
        .where(Image.entity_type == entity_type)
        .where(Image.entity_id == entity_id)
        .order_by(Image.created_at.desc())
    )
    images = result.scalars().all()
    
    return templates.TemplateResponse(
        "images/list.html",
        {"request": request, "entity": entity, "entity_type": entity_type, "entity_id": entity_id, "images": images}
    )

@router.get("/{entity_type}/{entity_id}/images/upload", response_class=HTMLResponse)
async def upload_image_form(
    entity_type: str, 
    entity_id: int, 
    request: Request, 
    db: AsyncSession = Depends(get_db)
):
    """Display form to upload new image"""
    # Verify entity exists
    entity = await get_entity(db, entity_type, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"{entity_type.title()} not found")
    
    return templates.TemplateResponse(
        "images/upload.html",
        {"request": request, "entity": entity, "entity_type": entity_type, "entity_id": entity_id}
    )

@router.post("/{entity_type}/{entity_id}/images")
async def upload_image(
    entity_type: str, 
    entity_id: int, 
    request: Request, 
    image: UploadFile = File(...),
    process_ocr: bool = Form(False),
    db: AsyncSession = Depends(get_db)
):
    """Upload a new image and optionally process OCR"""
    # Verify entity exists
    entity = await get_entity(db, entity_type, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"{entity_type.title()} not found")
    
    # Save the image
    image_data = await image_processor.save_image(image, entity_type, entity_id)
    
    # Create new image record
    new_image = Image(
        entity_type=entity_type,
        entity_id=entity_id,
        filename=image_data["filename"],
        file_path=image_data["file_path"],
        original_filename=image_data["original_filename"],
        mime_type=image_data["mime_type"],
        file_size=image_data["file_size"]
    )
    
    # Process OCR if requested (and if entity_type is 'seed')
    ocr_result = None
    if process_ocr and entity_type == 'seed':
        ocr_text, structured_data = await image_processor.process_ocr(image_data["file_path"])
        new_image.ocr_text = ocr_text
        new_image.structured_data = structured_data  # Store structured data in new column
        ocr_result = {
            "ocr_text": ocr_text,
            "structured_data": structured_data
        }
        
        # Update seed with extracted data if available
        if structured_data:
            await update_seed_from_ocr(db, entity_id, structured_data)
    
    # Add to database
    db.add(new_image)
    await db.commit()
    await db.refresh(new_image)
    
    # Return the response
    if process_ocr and entity_type == 'seed':
        return templates.TemplateResponse(
            "images/ocr_result.html",
            {
                "request": request, 
                "entity": entity, 
                "entity_type": entity_type,
                "entity_id": entity_id,
                "image": new_image, 
                "ocr_result": ocr_result
            }
        )
    else:
        return templates.TemplateResponse(
            "redirect.html",
            {"request": request, "url": f"/{entity_type}s/{entity_id}/images"}
        )

@router.get("/{entity_type}/{entity_id}/images/{image_id}", response_class=HTMLResponse)
async def image_detail(
    entity_type: str,
    entity_id: int,
    image_id: int, 
    request: Request, 
    db: AsyncSession = Depends(get_db)
):
    """Show details of a specific image"""
    # Verify entity exists
    entity = await get_entity(db, entity_type, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"{entity_type.title()} not found")
    
    # Get the image
    result = await db.execute(
        select(Image)
        .where(Image.id == image_id)
        .where(Image.entity_type == entity_type)
        .where(Image.entity_id == entity_id)
    )
    image = result.scalars().first()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return templates.TemplateResponse(
        "images/detail.html",
        {
            "request": request, 
            "entity": entity, 
            "entity_type": entity_type,
            "entity_id": entity_id,
            "image": image
        }
    )

@router.post("/{entity_type}/{entity_id}/images/{image_id}/process-ocr")
async def process_image_ocr(
    entity_type: str,
    entity_id: int,
    image_id: int, 
    request: Request, 
    db: AsyncSession = Depends(get_db)
):
    """Process OCR for an existing image"""
    # Only allow OCR processing for seed images
    if entity_type != 'seed':
        raise HTTPException(status_code=400, detail="OCR processing is only available for seed images")
        
    # Verify entity exists
    entity = await get_entity(db, entity_type, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"{entity_type.title()} not found")
    
    # Get the image
    result = await db.execute(
        select(Image)
        .where(Image.id == image_id)
        .where(Image.entity_type == entity_type)
        .where(Image.entity_id == entity_id)
    )
    image = result.scalars().first()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Process OCR
    ocr_text, structured_data = await image_processor.process_ocr(image.file_path)
    image.ocr_text = ocr_text
    image.structured_data = structured_data  # Store structured data in new column
    
    # Update seed with extracted data if available
    if structured_data and entity_type == 'seed':
        await update_seed_from_ocr(db, entity_id, structured_data)
    
    # Save changes
    await db.commit()
    
    # Return OCR results
    ocr_result = {
        "ocr_text": ocr_text,
        "structured_data": structured_data
    }
    
    return templates.TemplateResponse(
        "images/ocr_result.html",
        {
            "request": request, 
            "entity": entity, 
            "entity_type": entity_type,
            "entity_id": entity_id,
            "image": image, 
            "ocr_result": ocr_result
        }
    )

@router.post("/{entity_type}/{entity_id}/images/{image_id}/delete")
async def delete_image(
    entity_type: str,
    entity_id: int,
    image_id: int, 
    request: Request, 
    db: AsyncSession = Depends(get_db)
):
    """Delete an image"""
    # Verify entity exists
    entity = await get_entity(db, entity_type, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"{entity_type.title()} not found")
    
    # Get the image
    result = await db.execute(
        select(Image)
        .where(Image.id == image_id)
        .where(Image.entity_type == entity_type)
        .where(Image.entity_id == entity_id)
    )
    image = result.scalars().first()
    
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    
    # Delete the image file
    try:
        if os.path.exists(image.file_path):
            os.remove(image.file_path)
    except Exception as e:
        # Log the error but continue with database deletion
        print(f"Error deleting image file: {str(e)}")
    
    # Delete from database
    await db.delete(image)
    await db.commit()
    
    return templates.TemplateResponse(
        "redirect.html",
        {"request": request, "url": f"/{entity_type}s/{entity_id}/images"}
    )

@router.get("/{entity_type}/{entity_id}/images/{image_id}/apply-ocr-data")
async def apply_ocr_data_form(
    entity_type: str,
    entity_id: int,
    image_id: int, 
    request: Request, 
    db: AsyncSession = Depends(get_db)
):
    """Display form to review and apply OCR data"""
    # Only allow OCR processing for seed images
    if entity_type != 'seed':
        raise HTTPException(status_code=400, detail="OCR processing is only available for seed images")
        
    # Verify entity exists
    entity = await get_entity(db, entity_type, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"{entity_type.title()} not found")
    
    # Get the image
    result = await db.execute(
        select(Image)
        .where(Image.id == image_id)
        .where(Image.entity_type == entity_type)
        .where(Image.entity_id == entity_id)
    )
    image = result.scalars().first()
    
    if not image or not image.ocr_text:
        raise HTTPException(status_code=404, detail="Image or OCR data not found")
    
    # Extract structured data
    structured_data = await image_processor.extract_structured_data(image.ocr_text)
    
    if not structured_data:
        raise HTTPException(status_code=422, detail="Could not extract structured data")
    
    return templates.TemplateResponse(
        "images/apply_ocr.html",
        {
            "request": request, 
            "entity": entity, 
            "entity_type": entity_type,
            "entity_id": entity_id,
            "image": image, 
            "structured_data": structured_data
        }
    )

@router.post("/{entity_type}/{entity_id}/images/{image_id}/apply-ocr-data")
async def apply_ocr_data(
    entity_type: str,
    entity_id: int,
    image_id: int, 
    request: Request, 
    ocr_data: str = Form(...),
    db: AsyncSession = Depends(get_db)
):
    """Apply OCR data to the entity record"""
    # Only allow OCR processing for seed images
    if entity_type != 'seed':
        raise HTTPException(status_code=400, detail="OCR processing is only available for seed images")
        
    # Verify entity exists
    entity = await get_entity(db, entity_type, entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"{entity_type.title()} not found")
    
    # Parse OCR data
    try:
        structured_data = json.loads(ocr_data)
        
        # Update seed with extracted data
        if entity_type == 'seed':
            await update_seed_from_ocr(db, entity_id, structured_data)
            
            # Save changes
            await db.commit()
        
        return templates.TemplateResponse(
            "redirect.html",
            {"request": request, "url": f"/{entity_type}s/{entity_id}"}
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")

# Helper functions
async def get_entity(db: AsyncSession, entity_type: str, entity_id: int):
    """Get an entity by type and ID"""
    if entity_type == 'seed':
        result = await db.execute(select(Seed).where(Seed.id == entity_id))
        return result.scalars().first()
    else:
        # Add support for other entity types here
        return None

async def update_seed_from_ocr(db: AsyncSession, seed_id: int, structured_data: dict):
    """Update seed entity with OCR structured data"""
    # Get the seed
    result = await db.execute(select(Seed).where(Seed.id == seed_id))
    seed = result.scalars().first()
    
    if not seed:
        return
    
    # Only update fields that are present in the structured data
    if "name" in structured_data and structured_data["name"]:
        seed.name = structured_data["name"]
    
    if "variety" in structured_data and structured_data["variety"]:
        seed.variety = structured_data["variety"]
    
    if "germination_rate" in structured_data and structured_data["germination_rate"]:
        try:
            seed.germination_rate = float(structured_data["germination_rate"])
        except (ValueError, TypeError):
            pass
    
    if "maturity" in structured_data and structured_data["maturity"]:
        try:
            seed.maturity = int(structured_data["maturity"])
        except (ValueError, TypeError):
            pass
    
    if "growth" in structured_data and structured_data["growth"]:
        seed.growth = structured_data["growth"]
    
    if "seed_depth" in structured_data and structured_data["seed_depth"]:
        try:
            seed.seed_depth = float(structured_data["seed_depth"])
        except (ValueError, TypeError):
            pass
    
    if "spacing" in structured_data and structured_data["spacing"]:
        try:
            seed.spacing = float(structured_data["spacing"])
        except (ValueError, TypeError):
            pass
    
    if "notes" in structured_data and structured_data["notes"]:
        # Append to existing notes if present
        existing_notes = seed.notes or ""
        if existing_notes:
            seed.notes = existing_notes + "\n\n--- OCR EXTRACTED NOTES ---\n" + structured_data["notes"]
        else:
            seed.notes = structured_data["notes"]