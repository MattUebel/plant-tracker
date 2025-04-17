from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    File,
    UploadFile,
    Form,
    Query,
)
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import get_db
from models.seed import Seed
from models.image import Image  # Ensure we're using the Image model only
from utils.image_processor import image_processor
from routers.seeds import update_seed_from_structured_data  # Import shared function
from typing import List, Optional
import os
import json

# Fix the import path to match the Docker container structure
import sys

# Import the templates from proper location for Docker
from utils.templates import templates

router = APIRouter(
    tags=["seed_images"],
)

# Create upload directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)


@router.get("/{seed_id}/images", response_class=HTMLResponse)
async def list_seed_images(
    seed_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """List all images for a specific seed"""
    # First check if the seed exists
    result = await db.execute(select(Seed).where(Seed.id == seed_id))
    seed = result.scalars().first()

    if not seed:
        raise HTTPException(status_code=404, detail="Seed not found")

    # Get all images for this seed
    result = await db.execute(
        select(Image)
        .where(Image.entity_type == "seed")
        .where(Image.entity_id == seed_id)
    )
    images = result.scalars().all()

    return templates.TemplateResponse(
        "seed_images/list.html", {"request": request, "seed": seed, "images": images}
    )


@router.get("/{seed_id}/images/upload", response_class=HTMLResponse)
async def upload_seed_image_form(
    seed_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Display form to upload an image for a seed"""
    # Get the seed
    result = await db.execute(select(Seed).where(Seed.id == seed_id))
    seed = result.scalars().first()

    if not seed:
        raise HTTPException(status_code=404, detail="Seed not found")

    # Get the configured vision API provider for display
    vision_api_provider = os.getenv("VISION_API_PROVIDER", "claude").capitalize()

    return templates.TemplateResponse(
        "seeds/upload_image.html",
        {
            "request": request,
            "seed": seed,
            "title": f"Add Image - {seed.name}",
            "preview_mode": True,
            "vision_api_provider": vision_api_provider,
        },
    )


@router.post("/{seed_id}/images", response_class=HTMLResponse)
async def upload_seed_image(
    seed_id: int,
    request: Request,
    image: UploadFile = File(...),
    provider: str = Form(None),
    process_ocr: bool = Form(False),
    preview_mode: bool = Form(False),
    db: AsyncSession = Depends(get_db),
):
    """Upload an image for a seed"""
    # Get the seed
    result = await db.execute(select(Seed).where(Seed.id == seed_id))
    seed = result.scalars().first()

    if not seed:
        raise HTTPException(status_code=404, detail="Seed not found")

    try:
        # Save the image
        image_data = await image_processor.save_image(image, "seed", seed_id)

        # Create new image record with proper seed relationship
        new_image = Image(
            entity_type="seed",
            entity_id=seed_id,
            filename=image_data["filename"],
            file_path=image_data["file_path"],
            original_filename=image_data["original_filename"],
            mime_type=image_data["mime_type"],
            file_size=image_data["file_size"],
            seed_id=seed_id,  # Set the seed_id for the relationship
        )

        # Process image with configured Vision API if requested
        if process_ocr:
            # Override provider based on user selection
            if provider:
                image_processor.vision_api_provider = provider.lower()
            # Only keep structured_data; ignore raw OCR
            _, structured_data = await image_processor.process_image_with_vision_api(
                image_data["file_path"]
            )
            new_image.structured_data = structured_data

        # Save to database
        db.add(new_image)
        await db.commit()

        # If preview mode is enabled and we have OCR data, show the preview
        if preview_mode and process_ocr and structured_data:
            # Create a copy of the seed to show how it would look with OCR data
            preview_seed = Seed(
                name=seed.name,
                variety=seed.variety,
                brand=seed.brand,
                germination_rate=seed.germination_rate,
                maturity=seed.maturity,
                growth=seed.growth,
                seed_depth=seed.seed_depth,
                spacing=seed.spacing,
                quantity=seed.quantity,
                notes=seed.notes,
            )

            # Apply OCR data to the preview seed
            preview_seed = await update_seed_from_structured_data(
                preview_seed, structured_data, overwrite_existing=True
            )

            # Get the vision API provider for display
            vision_api_provider = os.getenv(
                "VISION_API_PROVIDER", "claude"
            ).capitalize()

            # Return the preview template
            return templates.TemplateResponse(
                "seed_images/preview_changes.html",
                {
                    "request": request,
                    "seed": seed,
                    "preview_seed": preview_seed,
                    "image": new_image,
                    "structured_data": structured_data,
                    "vision_api_provider": vision_api_provider,
                },
            )

        # Redirect back to seed detail page
        return templates.TemplateResponse(
            "redirect.html", {"request": request, "url": f"/seeds/{seed_id}"}
        )

    except Exception as e:
        await db.rollback()
        print(f"Error uploading seed image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")


@router.get("/{seed_id}/images/{image_id}", response_class=HTMLResponse)
async def seed_image_detail(
    seed_id: int, image_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Show details of a specific seed image"""
    # Check if the seed exists
    result = await db.execute(select(Seed).where(Seed.id == seed_id))
    seed = result.scalars().first()

    if not seed:
        raise HTTPException(status_code=404, detail="Seed not found")

    # Get the image
    result = await db.execute(
        select(Image)
        .where(Image.id == image_id)
        .where(Image.entity_type == "seed")
        .where(Image.entity_id == seed_id)
    )
    image = result.scalars().first()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Get the vision API provider for display
    vision_api_provider = os.getenv("VISION_API_PROVIDER", "claude").capitalize()

    return templates.TemplateResponse(
        "seed_images/detail.html",
        {
            "request": request,
            "seed": seed,
            "image": image,
            "vision_api_provider": vision_api_provider,
        },
    )


@router.post("/{seed_id}/images/{image_id}/process-ocr")
async def process_image_ocr(
    seed_id: int, image_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Process OCR for an existing image"""
    # Check if the seed exists
    result = await db.execute(select(Seed).where(Seed.id == seed_id))
    seed = result.scalars().first()

    if not seed:
        raise HTTPException(status_code=404, detail="Seed not found")

    # Get the image
    result = await db.execute(
        select(Image)
        .where(Image.id == image_id)
        .where(Image.entity_type == "seed")
        .where(Image.entity_id == seed_id)
    )
    image = result.scalars().first()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Determine provider from form or default
    form = await request.form()
    selected = form.get("provider")
    if selected:
        image_processor.vision_api_provider = selected.lower()
    # Only keep structured_data, no raw OCR
    _, structured_data = await image_processor.process_image_with_vision_api(
        image.file_path
    )
    image.structured_data = structured_data

    # Create a preview seed without modifying the original
    preview_seed = Seed(
        id=seed.id,
        name=seed.name,
        variety=seed.variety,
        brand=seed.brand,
        germination_rate=seed.germination_rate,
        maturity=seed.maturity,
        growth=seed.growth,
        seed_depth=seed.seed_depth,
        spacing=seed.spacing,
        quantity=seed.quantity,
        notes=seed.notes,
    )

    # Update preview seed with extracted data
    preview_seed = await update_seed_from_structured_data(
        preview_seed, structured_data, overwrite_existing=True
    )

    # Save just the OCR data to the image
    await db.commit()

    # Get the vision API provider for display
    vision_api_provider = os.getenv("VISION_API_PROVIDER", "claude").capitalize()

    # Return OCR results with preview
    return templates.TemplateResponse(
        "seed_images/preview_changes.html",
        {
            "request": request,
            "seed": seed,
            "preview_seed": preview_seed,
            "image": image,
            "structured_data": structured_data,
            "vision_api_provider": vision_api_provider,
        },
    )


@router.post("/{seed_id}/images/{image_id}/delete")
async def delete_seed_image(
    seed_id: int, image_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Delete a seed image"""
    # Check if the seed exists
    result = await db.execute(select(Seed).where(Seed.id == seed_id))
    seed = result.scalars().first()

    if not seed:
        raise HTTPException(status_code=404, detail="Seed not found")

    # Get the image
    result = await db.execute(
        select(Image)
        .where(Image.id == image_id)
        .where(Image.entity_type == "seed")
        .where(Image.entity_id == seed_id)
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

    # Delete the image record
    await db.delete(image)
    await db.commit()

    # Redirect back to seed images list
    return templates.TemplateResponse(
        "redirect.html", {"request": request, "url": f"/seeds/{seed_id}/images"}
    )


@router.get("/{seed_id}/images/{image_id}/apply-ocr-data")
async def apply_ocr_data_form(
    seed_id: int, image_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Display form to review and apply OCR data to seed"""
    # Check if the seed exists
    result = await db.execute(select(Seed).where(Seed.id == seed_id))
    seed = result.scalars().first()

    if not seed:
        raise HTTPException(status_code=404, detail="Seed not found")

    # Get the image
    result = await db.execute(
        select(Image)
        .where(Image.id == image_id)
        .where(Image.entity_type == "seed")
        .where(Image.entity_id == seed_id)
    )
    image = result.scalars().first()

    if not image or not image.structured_data:
        raise HTTPException(status_code=404, detail="Image or OCR data not found")

    # Get structured data directly from the image
    structured_data = image.structured_data

    if not structured_data:
        raise HTTPException(status_code=422, detail="Could not extract structured data")

    # Get the vision API provider for display
    vision_api_provider = os.getenv("VISION_API_PROVIDER", "claude").capitalize()

    return templates.TemplateResponse(
        "seed_images/apply_ocr.html",
        {
            "request": request,
            "seed": seed,
            "image": image,
            "structured_data": structured_data,
            "vision_api_provider": vision_api_provider,
        },
    )


@router.post("/{seed_id}/images/{image_id}/apply-ocr-data")
async def apply_ocr_data(
    seed_id: int,
    image_id: int,
    request: Request,
    ocr_data: str = Form(...),
    overwrite_existing: bool = Form(False),
    db: AsyncSession = Depends(get_db),
):
    """Apply OCR data to the seed record"""
    # Check if the seed exists
    result = await db.execute(select(Seed).where(Seed.id == seed_id))
    seed = result.scalars().first()

    if not seed:
        raise HTTPException(status_code=404, detail="Seed not found")

    # Parse OCR data
    try:
        structured_data = json.loads(ocr_data)

        # Update seed with extracted data
        seed = await update_seed_from_structured_data(
            seed, structured_data, overwrite_existing=overwrite_existing
        )

        # Save changes
        await db.commit()

        return templates.TemplateResponse(
            "redirect.html", {"request": request, "url": f"/seeds/{seed_id}"}
        )
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON data")


# Note: Removed the reprocess-ocr endpoint since we're using a more robust Gemini approach
