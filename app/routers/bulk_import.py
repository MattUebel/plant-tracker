"""
Router for handling bulk import operations.
"""

import os
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from database import get_db
from models.seed import Seed
from models.image import Image
from utils.seed_packet_processor import seed_packet_processor
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/bulk-import",
    tags=["bulk-import"],
)


@router.post("/process-image")
async def process_bulk_image(
    request: Request,
    image_file: UploadFile = File(...),
    provider: str = "gemini",  # Default provider, can be overridden if needed
    db: AsyncSession = Depends(get_db),
):
    """
    Receives an image, processes it for seed data, and creates Seed/Image records directly.
    NOTE: This endpoint is currently unsecured.
    """
    log_prefix = "[/bulk-import/process-image]"
    logger.info(f"{log_prefix} Received request for file: {image_file.filename}")
    try:
        # Read image data
        image_data = await image_file.read()
        if not image_data:
            logger.warning(
                f"{log_prefix} Empty image file received: {image_file.filename}"
            )
            raise HTTPException(status_code=400, detail="Empty image file received.")

        logger.info(
            f"{log_prefix} Processing {image_file.filename} using provider: {provider}"
        )
        # Process using the existing seed packet processor
        structured_data, file_path = await seed_packet_processor.process_seed_packet(
            image_data, filename=image_file.filename, provider=provider
        )

        if not structured_data:
            logger.error(
                f"{log_prefix} Failed to extract structured data for {image_file.filename}"
            )
            # Return a specific response for failed extraction, but not an error that stops the script
            return {
                "message": "Skipped: Failed to extract any structured data.",
                "filename": image_file.filename,
                "status": "skipped_extraction_failed",
            }

        # --- Validation Step ---
        extracted_name = structured_data.get("name")
        extracted_variety = structured_data.get("variety")

        # Check if name or variety is missing or effectively empty (like 'Unknown Seed')
        if (
            not extracted_name
            or extracted_name == "Unknown Seed"
            or not extracted_variety
        ):
            reason = (
                "Missing name"
                if (not extracted_name or extracted_name == "Unknown Seed")
                else "Missing variety"
            )
            logger.warning(
                f"{log_prefix} Skipping {image_file.filename} due to missing essential data: {reason}. Data: {json.dumps(structured_data)}"
            )
            return {
                "message": f"Skipped: {reason}.",
                "filename": image_file.filename,
                "status": "skipped_missing_data",
                "extracted_data": structured_data,  # Optionally return data for debugging
            }

        # --- Direct Creation Logic (proceed only if validation passes) ---
        try:
            # Create new seed with only valid fields from processed structured data
            new_seed = Seed(
                name=extracted_name,  # Use validated name
                variety=extracted_variety,  # Use validated variety
                brand=structured_data.get("brand"),
                seed_depth=structured_data.get("seed_depth"),
                spacing=structured_data.get("spacing"),
                notes=structured_data.get("notes"),
                # Add other fields from structured_data if they exist and are processed
                # germination_rate=structured_data.get("germination_rate"), # Example
                # maturity=structured_data.get("maturity"), # Example
            )
            db.add(new_seed)
            await db.flush()  # Flush to get the new_seed.id

            # Create associated image record
            new_image = Image(
                entity_type="seed",
                entity_id=new_seed.id,
                seed_id=new_seed.id,  # Explicitly set seed_id
                filename=os.path.basename(file_path),
                file_path=file_path,
                original_filename=image_file.filename,
                mime_type=image_file.content_type,
                file_size=len(image_data),
                structured_data=structured_data,  # Store the processed structured data
            )
            db.add(new_image)

            await db.commit()
            await db.refresh(new_seed)
            await db.refresh(new_image)

            logger.info(
                f"{log_prefix} Successfully created Seed ID: {new_seed.id} and Image ID: {new_image.id} for {image_file.filename}"
            )
            return {
                "message": "Image processed and seed created successfully.",
                "seed_id": new_seed.id,
                "image_id": new_image.id,
                "filename": image_file.filename,
                "status": "success",  # Added status for client script
            }

        except Exception as db_exc:
            await db.rollback()
            logger.error(
                f"{log_prefix} Database error creating seed/image for {image_file.filename}: {db_exc}",
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=f"Database error: {db_exc}")

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions
        logger.error(
            f"{log_prefix} HTTPException for {image_file.filename}: {http_exc.detail}"
        )
        raise http_exc
    except Exception as e:
        logger.error(
            f"{log_prefix} Error processing bulk image {image_file.filename}: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Internal server error processing image: {e}"
        )
