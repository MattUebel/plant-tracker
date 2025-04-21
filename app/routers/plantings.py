from fastapi import APIRouter, Depends, HTTPException, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from database import get_db
from models.seed import Seed
from models.planting import Planting
from models.image import Image
from models.transplant_event import TransplantEvent
from utils.image_processor import image_processor

# Fix the import path for templates to work in Docker - use relative import
from utils.templates import templates

# Fix the import path to match the Docker container structure
import sys
import os

import datetime
import json
from datetime import date
from typing import Optional, Dict, Any, List, Tuple

router = APIRouter(
    prefix="/plantings",
    tags=["plantings"],
)


async def process_planting_image(
    image: UploadFile, planting_id: int, db: AsyncSession = None
) -> Image:
    """
    Process a planting image and associate it with the planting.

    Returns:
        Image object
    """
    # Get the planting
    result = await db.execute(select(Planting).where(Planting.id == planting_id))
    planting = result.scalars().first()

    if not planting:
        raise HTTPException(status_code=404, detail="Planting not found")

    # Save the uploaded image
    image_data = await image_processor.save_image(image, "planting", planting.id)

    # Create new image record with proper planting relationship
    new_image = Image(
        entity_type="planting",
        entity_id=planting.id,
        filename=image_data["filename"],
        file_path=image_data["file_path"],
        original_filename=image_data["original_filename"],
        mime_type=image_data["mime_type"],
        file_size=image_data["file_size"],
        planting_id=planting.id,
    )

    # Save image
    db.add(new_image)
    await db.commit()
    await db.refresh(new_image)

    return new_image


@router.get("/", response_class=HTMLResponse)
async def list_plantings(request: Request, db: AsyncSession = Depends(get_db)):
    """List all plantings"""
    result = await db.execute(
        select(Planting)
        .options(
            selectinload(Planting.images),
            selectinload(Planting.seed),
            selectinload(Planting.transplant_events),
        )
        .order_by(Planting.created_at.desc())
    )
    plantings = result.scalars().unique().all()

    return templates.TemplateResponse(
        "plantings/list.html",
        {"request": request, "plantings": plantings, "title": "Plantings"},
    )


@router.get("/new", response_class=HTMLResponse)
async def new_planting_form(request: Request, db: AsyncSession = Depends(get_db)):
    """Display form to add a new planting"""
    # Get all seeds for the dropdown
    result = await db.execute(
        select(Seed).options(selectinload(Seed.images)).order_by(Seed.name)
    )
    seeds = result.scalars().unique().all()

    return templates.TemplateResponse(
        "plantings/new.html",
        {"request": request, "title": "Start a New Planting", "seeds": seeds},
    )


@router.post("/", response_class=HTMLResponse)
async def create_planting(
    request: Request,
    name: str = Form(...),
    seed_id: str = Form(default=""),
    seeds_planted: str = Form(default=""),
    planting_date: Optional[str] = Form(default=None),
    notes: Optional[str] = Form(default=""),
    image: Optional[UploadFile] = File(default=None),
    db: AsyncSession = Depends(get_db),
):
    """Create new planting"""
    # Convert empty strings to None for integer fields
    parsed_seed_id = int(seed_id) if seed_id.strip() else None
    parsed_seeds_planted = int(seeds_planted) if seeds_planted.strip() else None

    # Parse the planting date if provided
    parsed_planting_date = None
    if planting_date:
        try:
            parsed_planting_date = date.fromisoformat(planting_date)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid date format. Please use YYYY-MM-DD."
            )

    # Create new planting
    new_planting = Planting(
        name=name,
        seed_id=parsed_seed_id,
        seeds_planted=parsed_seeds_planted,
        planting_date=parsed_planting_date,
        notes=notes,
    )

    # Add to database
    db.add(new_planting)
    await db.commit()
    await db.refresh(new_planting)

    # Handle image upload if provided
    if image and image.filename:
        await process_planting_image(image, new_planting.id, db=db)

    # Redirect to planting details
    return templates.TemplateResponse(
        "redirect.html", {"request": request, "url": f"/plantings/{new_planting.id}"}
    )


@router.get("/{planting_id}", response_class=HTMLResponse)
async def planting_detail(
    planting_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Show details of a specific planting"""
    result = await db.execute(
        select(Planting)
        .options(
            selectinload(Planting.images),
            selectinload(Planting.seed),
            selectinload(Planting.transplant_events),
        )
        .where(Planting.id == planting_id)
    )
    planting = result.scalars().unique().first()

    if not planting:
        raise HTTPException(status_code=404, detail="Planting not found")

    return templates.TemplateResponse(
        "plantings/detail.html",
        {"request": request, "planting": planting, "title": planting.name},
    )


@router.get("/{planting_id}/edit", response_class=HTMLResponse)
async def edit_planting_form(
    planting_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Display form to edit planting"""
    # Get the planting
    result = await db.execute(
        select(Planting)
        .options(selectinload(Planting.seed))
        .where(Planting.id == planting_id)
    )
    planting = result.scalars().unique().first()

    if not planting:
        raise HTTPException(status_code=404, detail="Planting not found")

    # Get all seeds for the dropdown
    result = await db.execute(
        select(Seed).options(selectinload(Seed.images)).order_by(Seed.name)
    )
    seeds = result.scalars().unique().all()

    return templates.TemplateResponse(
        "plantings/edit.html",
        {
            "request": request,
            "planting": planting,
            "seeds": seeds,
            "title": f"Edit {planting.name}",
        },
    )


@router.post("/{planting_id}/update", response_class=HTMLResponse)
async def update_planting(
    planting_id: int,
    request: Request,
    name: str = Form(...),
    seed_id: str = Form(default=""),
    actual_germination_time: str = Form(default=""),
    actual_maturity_time: str = Form(default=""),
    seeds_planted: str = Form(default=""),
    successful_plants: str = Form(default=""),
    planting_date: Optional[str] = Form(default=None),
    notes: Optional[str] = Form(default=""),
    db: AsyncSession = Depends(get_db),
):
    """Update planting (transplant event fields removed)"""
    # Get the planting to update
    result = await db.execute(select(Planting).where(Planting.id == planting_id))
    planting = result.scalars().first()

    if not planting:
        raise HTTPException(status_code=404, detail="Planting not found")

    # Convert empty strings to None for integer fields
    parsed_seed_id = int(seed_id) if seed_id.strip() else None
    parsed_actual_germination_time = (
        int(actual_germination_time) if actual_germination_time.strip() else None
    )
    parsed_actual_maturity_time = (
        int(actual_maturity_time) if actual_maturity_time.strip() else None
    )
    parsed_seeds_planted = int(seeds_planted) if seeds_planted.strip() else None
    parsed_successful_plants = (
        int(successful_plants) if successful_plants.strip() else None
    )

    # Parse the planting date if provided
    if planting_date:
        try:
            planting.planting_date = date.fromisoformat(planting_date)
        except ValueError:
            raise HTTPException(
                status_code=400, detail="Invalid date format. Please use YYYY-MM-DD."
            )
    else:
        planting.planting_date = None

    # Update fields
    planting.name = name
    planting.seed_id = parsed_seed_id
    planting.actual_germination_time = parsed_actual_germination_time
    planting.actual_maturity_time = parsed_actual_maturity_time
    planting.seeds_planted = parsed_seeds_planted
    planting.successful_plants = parsed_successful_plants
    planting.notes = notes

    # Save changes
    await db.commit()

    # Redirect to planting details
    return templates.TemplateResponse(
        "redirect.html", {"request": request, "url": f"/plantings/{planting_id}"}
    )


@router.post("/{planting_id}/add-image", response_class=HTMLResponse)
async def add_planting_image(
    planting_id: int,
    request: Request,
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Add an image to a planting"""
    try:
        # Process and save the image
        await process_planting_image(image, planting_id, db=db)

        # Redirect to planting details
        return templates.TemplateResponse(
            "redirect.html", {"request": request, "url": f"/plantings/{planting_id}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding image: {str(e)}")


@router.post("/{planting_id}/delete")
async def delete_planting(
    planting_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Delete a planting and all associated image files"""
    # Get the planting to delete
    result = await db.execute(select(Planting).where(Planting.id == planting_id))
    planting = result.scalars().first()

    if not planting:
        raise HTTPException(status_code=404, detail="Planting not found")

    # Get all associated images before deleting the planting
    associated_images = planting.images.copy() if planting.images else []

    # Collect image filenames for deletion
    image_filenames = [img.filename for img in associated_images if img.filename]

    try:
        # Delete the planting from database (this will cascade delete image records)
        await db.delete(planting)
        await db.commit()

        # Now delete the physical image files
        deleted_count = 0
        for filename in image_filenames:
            if image_processor.delete_image_file(filename):
                deleted_count += 1

        if image_filenames and deleted_count:
            print(
                f"Deleted {deleted_count} of {len(image_filenames)} image files for planting {planting_id}"
            )
    except Exception as e:
        await db.rollback()
        print(f"Error deleting planting {planting_id}: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting planting: {str(e)}"
        )

    # Redirect to plantings list
    return templates.TemplateResponse(
        "redirect.html", {"request": request, "url": "/plantings"}
    )


@router.post("/{planting_id}/remove-transplant/{event_id}")
async def remove_transplant_event(
    planting_id: int,
    event_id: int,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Remove a transplant event from a planting"""
    # Get the transplant event
    result = await db.execute(
        select(TransplantEvent)
        .where(TransplantEvent.id == event_id)
        .where(TransplantEvent.planting_id == planting_id)
    )
    event = result.scalars().first()

    if not event:
        raise HTTPException(status_code=404, detail="Transplant event not found")

    # Delete the event
    await db.delete(event)
    await db.commit()

    # Redirect back to planting details
    return templates.TemplateResponse(
        "redirect.html", {"request": request, "url": f"/plantings/{planting_id}"}
    )


@router.post("/{planting_id}/record-germination")
async def record_germination(
    planting_id: int,
    request: Request,
    germination_date: str = Form(...),
    notes: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Record when seeds have germinated"""
    # Get the planting
    result = await db.execute(select(Planting).where(Planting.id == planting_id))
    planting = result.scalars().first()

    if not planting:
        raise HTTPException(status_code=404, detail="Planting not found")

    # Parse the germination date
    try:
        parsed_germination_date = date.fromisoformat(germination_date)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Please use YYYY-MM-DD."
        )

    # Calculate actual germination time if planting date exists
    if planting.planting_date:
        days_diff = (parsed_germination_date - planting.planting_date).days
        planting.actual_germination_time = days_diff if days_diff >= 0 else None

    # Add a note about germination if provided
    if notes:
        if not planting.notes:
            planting.notes = ""

        planting.notes += f"\n\n[{datetime.datetime.now().strftime('%Y-%m-%d')}] Germination noted: {notes}"

    # Save changes
    await db.commit()

    # Redirect to planting details
    return templates.TemplateResponse(
        "redirect.html", {"request": request, "url": f"/plantings/{planting_id}"}
    )


@router.post("/{planting_id}/add-transplant")
async def add_transplant_event(
    planting_id: int,
    request: Request,
    transplant_date: str = Form(...),
    location: str = Form(...),
    container: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Add a more detailed transplant event to a planting"""
    # Get the planting
    result = await db.execute(select(Planting).where(Planting.id == planting_id))
    planting = result.scalars().first()

    if not planting:
        raise HTTPException(status_code=404, detail="Planting not found")

    # Parse the transplant date
    try:
        parsed_transplant_date = date.fromisoformat(transplant_date)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Please use YYYY-MM-DD."
        )

    # Build the full description for display
    full_description = f"Moved to: {location}"
    if container:
        full_description += f" in {container}"
    if description:
        full_description += f" - {description}"

    # Create a new TransplantEvent object
    new_event = TransplantEvent(
        planting_id=planting_id,
        date=parsed_transplant_date,
        location=location,
        container=container,
        description=description,
    )

    # Add to database
    db.add(new_event)
    await db.commit()

    # Redirect to planting details
    return templates.TemplateResponse(
        "redirect.html", {"request": request, "url": f"/plantings/{planting_id}"}
    )


@router.post("/{planting_id}/record-harvest")
async def record_harvest(
    planting_id: int,
    request: Request,
    harvest_date: str = Form(...),
    quantity: Optional[str] = Form(None),
    units: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Record when plants were harvested"""
    # Get the planting
    result = await db.execute(select(Planting).where(Planting.id == planting_id))
    planting = result.scalars().first()

    if not planting:
        raise HTTPException(status_code=404, detail="Planting not found")

    # Parse the harvest date
    try:
        parsed_harvest_date = date.fromisoformat(harvest_date)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Please use YYYY-MM-DD."
        )

    # Calculate actual maturity time if planting date exists
    if planting.planting_date:
        days_diff = (parsed_harvest_date - planting.planting_date).days
        planting.actual_maturity_time = days_diff if days_diff >= 0 else None

    # Initialize harvest_events if None
    if not hasattr(planting, "harvest_events") or not planting.harvest_events:
        planting.harvest_events = []

    # Create harvest event
    harvest_event = {
        "date": parsed_harvest_date.isoformat(),
        "quantity": quantity,
        "units": units,
        "notes": notes,
    }

    # Add to harvest events
    planting.harvest_events.append(harvest_event)

    # Add a note about harvest if provided
    harvest_note = f"[{datetime.datetime.now().strftime('%Y-%m-%d')}] Harvest recorded"
    if quantity and units:
        harvest_note += f": {quantity} {units}"
    if notes:
        harvest_note += f" - {notes}"

    if not planting.notes:
        planting.notes = ""
    planting.notes += f"\n\n{harvest_note}"

    # Save changes
    await db.commit()

    # Redirect to planting details
    return templates.TemplateResponse(
        "redirect.html", {"request": request, "url": f"/plantings/{planting_id}"}
    )


@router.post("/{planting_id}/debug-transplant")
async def debug_transplant(
    planting_id: int, request: Request, db: AsyncSession = Depends(get_db)
):
    """Debug route to log form data"""
    form_data = await request.form()
    form_dict = {key: value for key, value in form_data.items()}

    # Print to console with clear markers
    print("=" * 50)
    print("DEBUG - FORM DATA RECEIVED:")
    print(json.dumps(form_dict, indent=2))
    print("=" * 50)

    # Also write to a debug file
    with open("/app/debug_form_data.txt", "w") as f:
        f.write(f"Form data received at {datetime.datetime.now()}:\n")
        f.write(json.dumps(form_dict, indent=2))

    # Return the form data for debugging
    return HTMLResponse(
        f"""
    <html>
        <head>
            <title>Debug Form Data</title>
            <style>
                body {{ font-family: Arial, sans-serif; padding: 20px; }}
                pre {{ background: #f0f0f0; padding: 15px; border-radius: 5px; }}
                .back {{ margin-top: 20px; }}
            </style>
        </head>
        <body>
            <h1>Form Data Received:</h1>
            <pre>{json.dumps(form_dict, indent=2)}</pre>
            <div class="back">
                <a href="/plantings/{planting_id}">Back to Planting Details</a>
            </div>
        </body>
    </html>
    """
    )
