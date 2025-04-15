from fastapi import APIRouter, Depends, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from database import get_db
from models.seed import Seed
from models.image import Image
from utils.image_processor import image_processor
from utils.templates import templates
import sys
import os
import datetime
from typing import Optional, List

# Add the current directory to the Python path
sys.path.insert(0, os.getcwd())

router = APIRouter(
    prefix="/seeds",
    tags=["seeds"],
)


@router.get("/", response_class=HTMLResponse)
async def get_seeds(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        # Using async-compatible query format with unique() to handle collections
        # Also loading plantings relationship for the plantings count column
        result = await db.execute(
            select(Seed)
            .options(selectinload(Seed.plantings), selectinload(Seed.images))
            .order_by(Seed.name)
        )
        seeds = result.scalars().unique().all()
        return templates.TemplateResponse(
            "seeds/list.html", {"request": request, "seeds": seeds}
        )
    except Exception as e:
        # Add some debug logging to help diagnose issues
        print(f"Error in get_seeds: {str(e)}")
        raise


@router.get("/api/{seed_id}")
async def get_seed_data(seed_id: int, db: AsyncSession = Depends(get_db)):
    """API endpoint to get seed data for auto-populating planting forms"""
    # Use async-compatible query format
    result = await db.execute(select(Seed).where(Seed.id == seed_id))
    seed = result.scalars().first()

    if not seed:
        raise HTTPException(status_code=404, detail="Seed not found")

    # Return seed data as JSON
    return {
        "id": seed.id,
        "name": seed.name,
        "variety": seed.variety,
        "germination": seed.germination_rate,
        "maturity": seed.maturity,
        "seed_depth": seed.seed_depth,
        "spacing": seed.spacing,
    }


@router.get("/new", response_class=HTMLResponse)
async def new_seed(request: Request):
    """Show the new seed form"""
    return templates.TemplateResponse("seeds/new.html", {"request": request})


@router.post("/", response_class=HTMLResponse)
async def create_seed(
    request: Request,
    name: str = Form(None),
    variety: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    maturity_days: Optional[int] = Form(None),
    germination_days: Optional[int] = Form(None),
    planting_depth: Optional[float] = Form(None),
    spacing: Optional[float] = Form(None),
    growing_notes: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
):
    try:
        # Check for required fields
        if not name:
            return templates.TemplateResponse(
                "seeds/new.html",
                {
                    "request": request,
                    "error": "Seed name is required",
                },
                status_code=422,
            )

        # Create a new seed
        new_seed = Seed(
            name=name,
            variety=variety,
            maturity=maturity_days,  # Using correct field name from the model
            seed_depth=planting_depth,  # Using correct field name from the model
            spacing=spacing,
        )

        # Use async-compatible SQLAlchemy operations
        db.add(new_seed)
        await db.commit()
        await db.refresh(new_seed)

        # Save uploaded image if provided
        if image and image.filename:
            # Save the image using image_processor
            image_data = await image_processor.save_image(image, "seed", new_seed.id)

            # Create the image record using Image model
            new_image = Image(
                entity_type="seed",
                entity_id=new_seed.id,
                seed_id=new_seed.id,  # Set direct relationship
                filename=image_data["filename"],
                file_path=image_data["file_path"],
                original_filename=image_data["original_filename"],
                mime_type=image_data["mime_type"],
                file_size=image_data["file_size"],
            )
            db.add(new_image)
            await db.commit()

        return RedirectResponse(url=f"/seeds/{new_seed.id}", status_code=303)

    except Exception as e:
        print(f"Error in create_seed: {str(e)}")
        await db.rollback()
        return templates.TemplateResponse(
            "seeds/new.html",
            {
                "request": request,
                "error": f"Error creating seed: {str(e)}",
            },
            status_code=500,
        )


@router.get("/{seed_id}", response_class=HTMLResponse)
async def view_seed(request: Request, seed_id: int, db: AsyncSession = Depends(get_db)):
    # Use async-compatible query format
    result = await db.execute(select(Seed).where(Seed.id == seed_id))
    seed = result.scalars().first()

    if seed is None:
        raise HTTPException(status_code=404, detail="Seed not found")

    return templates.TemplateResponse(
        "seeds/detail.html", {"request": request, "seed": seed}
    )


@router.post("/{seed_id}/delete")
async def delete_seed(seed_id: int, db: AsyncSession = Depends(get_db)):
    try:
        # Use async-compatible query format
        result = await db.execute(select(Seed).where(Seed.id == seed_id))
        seed = result.scalars().first()

        if seed is None:
            raise HTTPException(status_code=404, detail="Seed not found")

        # Delete related images first to avoid constraint violations
        # Use Image model instead of SeedImage
        image_result = await db.execute(select(Image).where(Image.seed_id == seed_id))
        images = image_result.scalars().all()

        for image in images:
            # Delete the file if it exists
            if os.path.exists(image.file_path):
                os.unlink(image.file_path)

            await db.delete(image)

        await db.delete(seed)
        await db.commit()

        return RedirectResponse(url="/seeds", status_code=303)
    except Exception as e:
        print(f"Error deleting seed: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting seed: {str(e)}")


@router.get("/{seed_id}/edit", response_class=HTMLResponse)
async def edit_seed_form(
    request: Request, seed_id: int, db: AsyncSession = Depends(get_db)
):
    # Use async-compatible query format
    result = await db.execute(select(Seed).where(Seed.id == seed_id))
    seed = result.scalars().first()

    if seed is None:
        raise HTTPException(status_code=404, detail="Seed not found")

    return templates.TemplateResponse(
        "seeds/edit.html", {"request": request, "seed": seed}
    )


@router.post("/{seed_id}/edit")
async def update_seed(
    seed_id: int,
    name: str = Form(...),
    variety: Optional[str] = Form(None),
    source: Optional[str] = Form(None),
    maturity_days: Optional[int] = Form(None),
    germination_days: Optional[int] = Form(None),
    planting_depth: Optional[float] = Form(None),
    spacing: Optional[float] = Form(None),
    growing_notes: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db),
):
    try:
        # Use async-compatible query format
        result = await db.execute(select(Seed).where(Seed.id == seed_id))
        seed = result.scalars().first()

        if seed is None:
            raise HTTPException(status_code=404, detail="Seed not found")

        # Update fields
        seed.name = name
        seed.variety = variety
        # No source field in the model, skip it
        seed.maturity = maturity_days  # Use correct field name maturity
        # Handle germination_days separately if needed
        seed.seed_depth = planting_depth  # Use correct field name seed_depth
        seed.spacing = spacing

        # Save uploaded image if provided
        if image and image.filename:
            # Save the image using image_processor
            image_data = await image_processor.save_image(image, "seed", seed_id)

            # Create the image record using Image model
            new_image = Image(
                entity_type="seed",
                entity_id=seed_id,
                seed_id=seed_id,  # Set direct relationship
                filename=image_data["filename"],
                file_path=image_data["file_path"],
                original_filename=image_data["original_filename"],
                mime_type=image_data["mime_type"],
                file_size=image_data["file_size"],
            )
            db.add(new_image)

        await db.commit()

        return RedirectResponse(url=f"/seeds/{seed_id}", status_code=303)
    except Exception as e:
        print(f"Error updating seed: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error updating seed: {str(e)}")


@router.post("/{seed_id}/delete-image/{image_id}")
async def delete_seed_image(
    seed_id: int, image_id: int, db: AsyncSession = Depends(get_db)
):
    try:
        # Use async-compatible query format with Image model
        result = await db.execute(
            select(Image)
            .where(Image.id == image_id)
            .where(Image.seed_id == seed_id)
            .where(Image.entity_type == "seed")
        )
        image = result.scalars().first()

        if image is None:
            raise HTTPException(status_code=404, detail="Image not found")

        # Delete the file if it exists
        if os.path.exists(image.file_path):
            os.unlink(image.file_path)

        await db.delete(image)
        await db.commit()

        return RedirectResponse(url=f"/seeds/{seed_id}", status_code=303)
    except Exception as e:
        print(f"Error deleting seed image: {str(e)}")
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error deleting image: {str(e)}")


@router.post("/{seed_id}/add-image", response_class=HTMLResponse)
async def add_seed_image(
    seed_id: int,
    request: Request,
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Add an image to a seed from the detail page"""
    try:
        # Get the seed
        result = await db.execute(select(Seed).where(Seed.id == seed_id))
        seed = result.scalars().first()
        if not seed:
            raise HTTPException(status_code=404, detail="Seed not found")
        # Save the uploaded image
        image_data = await image_processor.save_image(image, "seed", seed_id)
        new_image = Image(
            entity_type="seed",
            entity_id=seed_id,
            seed_id=seed_id,
            filename=image_data["filename"],
            file_path=image_data["file_path"],
            original_filename=image_data["original_filename"],
            mime_type=image_data["mime_type"],
            file_size=image_data["file_size"],
        )
        db.add(new_image)
        await db.commit()
        # Redirect to seed details
        return templates.TemplateResponse(
            "redirect.html", {"request": request, "url": f"/seeds/{seed_id}"}
        )
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Error adding image: {str(e)}")


async def update_seed_from_structured_data(
    seed: Seed, structured_data: dict, overwrite_existing: bool = False
) -> Seed:
    """Update seed entity with structured data extracted from OCR"""

    def should_update(field_name: str) -> bool:
        """Determine if we should update a field based on overwrite setting"""
        current_value = getattr(seed, field_name, None)
        return overwrite_existing or not current_value

    # Update fields if they exist in structured data and meet update criteria
    if "name" in structured_data and structured_data["name"] and should_update("name"):
        seed.name = structured_data["name"]

    if (
        "variety" in structured_data
        and structured_data["variety"]
        and should_update("variety")
    ):
        seed.variety = structured_data["variety"]

    if (
        "brand" in structured_data
        and structured_data["brand"]
        and should_update("brand")
    ):
        seed.brand = structured_data["brand"]

    if (
        "germination_rate" in structured_data
        and structured_data["germination_rate"]
        and should_update("germination_rate")
    ):
        try:
            seed.germination_rate = float(structured_data["germination_rate"])
        except (ValueError, TypeError):
            pass

    if (
        "maturity" in structured_data
        and structured_data["maturity"]
        and should_update("maturity")
    ):
        try:
            seed.maturity = int(structured_data["maturity"])
        except (ValueError, TypeError):
            pass

    if (
        "seed_depth" in structured_data
        and structured_data["seed_depth"]
        and should_update("seed_depth")
    ):
        try:
            seed.seed_depth = float(structured_data["seed_depth"])
        except (ValueError, TypeError):
            pass

    if (
        "spacing" in structured_data
        and structured_data["spacing"]
        and should_update("spacing")
    ):
        try:
            seed.spacing = float(structured_data["spacing"])
        except (ValueError, TypeError):
            pass

    if "notes" in structured_data and structured_data["notes"]:
        # For notes, we append the new notes if there are existing ones
        existing_notes = seed.notes or ""
        if existing_notes and not overwrite_existing:
            seed.notes = (
                existing_notes
                + "\n\n--- OCR EXTRACTED NOTES ---\n"
                + structured_data["notes"]
            )
        else:
            seed.notes = structured_data["notes"]

    return seed
