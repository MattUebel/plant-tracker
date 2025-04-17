"""
Streamlined router for seed packet processing and seed creation.
This router provides a simplified flow for extracting data from seed packet images.
"""

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from database import get_db
from models.seed import Seed
from models.image import Image
from utils.seed_packet_processor import seed_packet_processor
from utils.templates import templates
from typing import Optional
import os
import json
from PIL import Image as PILImage, UnidentifiedImageError

router = APIRouter(
    prefix="/seed-packets",
    tags=["seed-packets"],
)


@router.get("/upload", response_class=HTMLResponse)
async def upload_seed_packet_form(request: Request):
    """Display a form to upload and process a seed packet image."""
    return templates.TemplateResponse(
        "seed_packets/upload.html",
        {
            "request": request,
            "title": "Upload Seed Packet",
        },
    )


@router.post("/process", response_class=HTMLResponse)
async def process_seed_packet(
    request: Request,
    provider: str = Form("gemini"),
    seed_packet: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Process a seed packet image to extract structured data and display preview.
    This is a streamlined process that always shows the review form first before creating a seed.
    """
    try:
        # Validate the uploaded file is a valid image
        contents = await seed_packet.read()
        if not contents or len(contents) == 0:
            return templates.TemplateResponse(
                "seed_packets/upload.html",
                {
                    "request": request,
                    "error": "No image file uploaded or file is empty.",
                },
                status_code=400,
            )
        # Try to open with PIL to verify it's a valid image
        try:
            from io import BytesIO

            PILImage.open(BytesIO(contents)).verify()
        except (UnidentifiedImageError, Exception):
            return templates.TemplateResponse(
                "seed_packets/upload.html",
                {
                    "request": request,
                    "error": "The uploaded file is not a valid image. Please try again.",
                },
                status_code=400,
            )
        # Reset file pointer for downstream processing
        await seed_packet.seek(0)

        # Read the uploaded file
        image_data = await seed_packet.read()

        # Process the seed packet using our centralized processor
        structured_data, file_path = await seed_packet_processor.process_seed_packet(
            image_data, filename=seed_packet.filename, provider=provider
        )

        # Debug the extracted data
        print(f"Extracted structured data: {structured_data}")
        print(f"File path: {file_path}")

        # Create a preview seed for display with proper property access
        preview_seed = {
            "name": structured_data.get("name", "Unknown Seed"),
            "variety": structured_data.get("variety", ""),
            "brand": structured_data.get("brand", ""),
            "germination_rate": structured_data.get("germination_rate", ""),
            "maturity": structured_data.get("maturity", ""),
            "seed_depth": structured_data.get("seed_depth", ""),
            "spacing": structured_data.get("spacing", ""),
            "notes": structured_data.get("notes", ""),
        }

        # Format the file_path to be a relative path for the template
        relative_file_path = file_path.split("/")[-1] if file_path else ""

        # Always show the preview for review and manual adjustment
        return templates.TemplateResponse(
            "seed_packets/preview.html",
            {
                "request": request,
                "provider": provider,
                "preview_seed": preview_seed,
                "structured_data": structured_data,
                "structured_data_json": json.dumps(structured_data),
                "file_path": relative_file_path,  # Use just the filename for image display
                "full_file_path": file_path,  # Keep the full path for form submission
                "original_filename": seed_packet.filename,
                "mime_type": seed_packet.content_type,
                "file_size": os.path.getsize(file_path) if file_path else 0,
                "display_error": not file_path or not os.path.exists(file_path),
            },
        )

    except Exception as e:
        # Handle errors gracefully
        error_message = str(e)
        print(f"Error processing seed packet: {error_message}")
        return templates.TemplateResponse(
            "seed_packets/upload.html",
            {
                "request": request,
                "title": "Upload Seed Packet",
                "error": f"Error processing seed packet: {error_message}",
                "retry_image_data": (
                    image_data.hex() if "image_data" in locals() else None
                ),
                "original_filename": seed_packet.filename,
                "mime_type": seed_packet.content_type,
            },
            status_code=500,
        )


@router.post("/retry-extraction", response_class=HTMLResponse)
async def retry_extraction(
    request: Request,
    image_data_hex: str = Form(...),
    original_filename: str = Form(...),
    mime_type: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """Retry structured data extraction using the original uploaded image data (hex-encoded)."""
    try:
        image_data = bytes.fromhex(image_data_hex)
        structured_data, file_path = await seed_packet_processor.process_seed_packet(
            image_data, filename=original_filename
        )
        preview_seed = {
            "name": structured_data.get("name", "Unknown Seed"),
            "variety": structured_data.get("variety", ""),
            "brand": structured_data.get("brand", ""),
            "germination_rate": structured_data.get("germination_rate", ""),
            "maturity": structured_data.get("maturity", ""),
            "seed_depth": structured_data.get("seed_depth", ""),
            "spacing": structured_data.get("spacing", ""),
            "notes": structured_data.get("notes", ""),
        }
        relative_file_path = file_path.split("/")[-1] if file_path else ""
        return templates.TemplateResponse(
            "seed_packets/preview.html",
            {
                "request": request,
                "preview_seed": preview_seed,
                "structured_data": structured_data,
                "structured_data_json": json.dumps(structured_data),
                "file_path": relative_file_path,
                "full_file_path": file_path,
                "original_filename": original_filename,
                "mime_type": mime_type,
                "file_size": os.path.getsize(file_path) if file_path else 0,
                "display_error": not file_path or not os.path.exists(file_path),
            },
        )
    except Exception as e:
        error_message = str(e)
        print(f"Error retrying extraction: {error_message}")
        return templates.TemplateResponse(
            "seed_packets/upload.html",
            {
                "request": request,
                "title": "Upload Seed Packet",
                "error": f"Retry failed: {error_message}",
            },
            status_code=500,
        )


@router.post("/create-from-preview")
async def create_seed_from_preview(
    request: Request,
    file_path: str = Form(...),
    original_filename: str = Form(...),
    mime_type: str = Form(...),
    ocr_text: str = Form(...),
    structured_data_json: str = Form(...),
    name: str = Form(...),
    variety: Optional[str] = Form(None),
    brand: Optional[str] = Form(None),
    seed_depth: Optional[str] = Form(None),
    spacing: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Create a seed record from the preview data with user modifications."""
    try:
        # Parse numeric fields that might be empty strings
        parsed_seed_depth = (
            float(seed_depth) if seed_depth and seed_depth.strip() else None
        )
        parsed_spacing = float(spacing) if spacing and spacing.strip() else None

        # Create new seed with only valid fields
        new_seed = Seed(
            name=name,
            variety=variety,
            brand=brand,
            seed_depth=parsed_seed_depth,
            spacing=parsed_spacing,
            notes=notes,
        )
        db.add(new_seed)
        await db.commit()
        await db.refresh(new_seed)

        print(f"Created new seed with ID: {new_seed.id}")

        # If the file path exists, create an image record
        if file_path and os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            filename = os.path.basename(file_path)

            print(
                f"Creating image record with filename: {filename}, file_path: {file_path}"
            )

            # Create image record linked to the seed
            new_image = Image(
                entity_type="seed",
                entity_id=new_seed.id,
                seed_id=new_seed.id,
                filename=filename,
                file_path=file_path,
                original_filename=original_filename,
                mime_type=mime_type,
                file_size=file_size,
                ocr_text=ocr_text,
                structured_data=structured_data_json,
            )

            db.add(new_image)
            await db.commit()
            print(f"Created new image record for seed {new_seed.id}")
        else:
            print(f"Warning: file path does not exist: {file_path}")

        # Redirect to the seed detail page
        return RedirectResponse(url=f"/seeds/{new_seed.id}", status_code=303)

    except Exception as e:
        # Handle errors gracefully
        error_message = str(e)
        print(f"Error creating seed from preview: {error_message}")
        return templates.TemplateResponse(
            "seed_packets/error.html",
            {
                "request": request,
                "title": "Error",
                "error": f"Error creating seed: {error_message}",
            },
            status_code=500,
        )
