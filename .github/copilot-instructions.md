- @azure Rule - Use Azure Best Practices: When generating code for Azure, running terminal commands for Azure, or performing operations related to Azure, invoke your `azure_development-get_best_practices` tool if available.

- This is a Plant Tracker application for managing seeds and plantings in a garden. It runs in docker containers. Always use the CLI tool to run "docker compose" commands to interact with the app or review the logs.

- The application uses a FastAPI backend with SQLAlchemy ORM and PostgreSQL database.

- Key features include:
  - Seed catalog management with image upload and OCR extraction
  - Planting tracking with timelines and transplant events
  - Image gallery with multiple image support for both seeds and plantings

- File structure:
  - `/app`: Main application code
  - `/app/models`: SQLAlchemy database models
  - `/app/routers`: FastAPI route handlers
  - `/app/templates`: HTML templates using Jinja2
  - `/app/utils`: Utility classes and functions
  - `/uploads`: Storage for uploaded images
  - `/scripts`: Utility scripts for installation and maintenance
    - `install_service.sh`: Script to install the app as a systemd service on Raspberry Pi OS
    - `migrate.sh`: Database migration script
    - `psql.sh`: PostgreSQL database management script

- Code conventions:
  - Use async/await patterns with SQLAlchemy
  - Keep routes in their respective router files
  - Use proper error handling with try/except blocks
  - Follow Bootstrap conventions for frontend components

- Deployment options:
  - Standard Docker deployment using docker compose
  - Raspberry Pi deployment as a systemd service using the provided installation script
  - The Raspberry Pi installation requires an API key for OCR functionality