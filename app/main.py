import os
import sys
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# Add current directory to Python path to ensure imports work in Docker
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import templates from local utils
from utils.templates import templates
from dotenv import load_dotenv
from routers.seeds import router as seeds_router
from routers.images import router as images_router
from routers.seed_images import router as seed_images_router
from routers.plantings import router as plantings_router
from routers.seed_packets import router as seed_packets_router  # Add new router

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="Plant Tracker",
    description="An application for tracking plants",
    version="0.1.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include routers with appropriate tags but use prefix that matches template URLs
# Note: removed prefixes to match the URL paths used in templates
app.include_router(seeds_router, tags=["seeds"])
app.include_router(images_router, prefix="/images", tags=["images"])
app.include_router(seed_images_router, tags=["seed_images"])
app.include_router(plantings_router, tags=["plantings"])
app.include_router(seed_packets_router, tags=["seed_packets"])  # Add new router


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "title": "Plant Tracker"}
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app", host="0.0.0.0", port=int(os.getenv("APP_PORT", 8000)), reload=True
    )
