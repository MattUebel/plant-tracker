# GitHub Copilot Instructions for Plant Tracker App

## General Workflow & Preferences

*   **Iterative Approach:** Please follow an iterative process. When asked to implement a feature or fix a bug:
    1.  **Gather Information:** Use tools (`semantic_search`, `read_file`, `grep_search`, etc.) to understand the current state and relevant code. Ask clarifying questions if needed.
    2.  **Plan (If Complex):** For non-trivial changes, propose 1-2 concise implementation plans before coding.
    3.  **Implement:** Make the necessary code changes using `insert_edit_into_file`. Ensure changes are holistic, updating related configurations (like `.env.example`), utility functions, or API endpoints as needed.
    4.  **Verify:** After editing files, always use `get_errors` to check for syntax/linting issues. If applicable, suggest relevant `docker compose` commands to test the changes.
*   **Holistic Changes:** Ensure modifications are comprehensive and don't negatively impact other parts of the application. Update related configurations (e.g., `.env.example` when `.env` structure changes) and documentation if necessary.
*   **Use Provided Documentation:** If specific documentation files (`#file:path/to/doc.md`) are referenced in the prompt, please consult them to guide the implementation.
*   **Preserve Defaults:** When adding options (like command-line arguments), ensure the application's default behavior (often configured via `.env`) remains unchanged unless explicitly overridden.
*   **Docker Interaction:** Always use the CLI tool to run `docker compose` commands (e.g., `docker compose build app`, `docker compose up -d`, `docker compose logs app`) to interact with the application containers. Do not just print the commands.

## Application Overview

- This is a Plant Tracker application for managing seeds and plantings in a garden.
- It runs in Docker containers.
- The application uses a FastAPI backend with SQLAlchemy ORM (async) and a PostgreSQL database.

## Key Features

- Seed catalog management with image upload and structured data extraction using vision models (Claude, Gemini, Mistral).
- Planting tracking with timelines and transplant events.
- Image gallery with multiple image support for both seeds and plantings.

## Vision Providers

- The app supports multiple vision API providers (Claude, Gemini, Mistral) for extracting data from seed packet images.
- Configuration is primarily managed via `.env` variables (`VISION_API_PROVIDER`, `*_API_KEY`, `*_MODEL`).
- All providers are currently implemented using a single API call approach with multimodal models.
- The bulk import script (`scripts/bulk_import_seeds.py`) allows overriding the default provider and model via command-line arguments (`--provider`, `--model`).

## File Structure

- `/app`: Main application code
  - `/app/models`: SQLAlchemy database models
  - `/app/routers`: FastAPI route handlers
  - `/app/templates`: HTML templates using Jinja2
  - `/app/utils`: Utility classes and functions (including `image_processor.py`)
- `/uploads`: Storage for uploaded images (managed by the application)
- `/scripts`: Utility scripts
  - `bulk_import_seeds.py`: Client script for bulk image processing.
  - `install_service.sh`: Script for Raspberry Pi systemd service installation.
  - `migrate.sh`: Alembic database migration helper script.
  - `psql.sh`: PostgreSQL database management script.
- `/docs`: Markdown documents explaining API usage or guides.

## Code Conventions

- Use async/await patterns with SQLAlchemy and FastAPI.
- Keep routes in their respective router files within `/app/routers`.
- Use proper error handling with try/except blocks and appropriate logging.
- Follow Bootstrap 5 conventions for frontend components in Jinja2 templates.

## Deployment Options

- Standard Docker deployment using `compose.yml`.
- Raspberry Pi deployment as a systemd service using `scripts/install_service.sh`.

## Database Migrations

Use the `scripts/migrate.sh` helper script (which runs Alembic inside the Docker container) to manage database schema changes:

- **Generate:** `./scripts/migrate.sh migrate "Your migration message"`
- **Apply:** `./scripts/migrate.sh upgrade head`
- **Revert:** `./scripts/migrate.sh downgrade -1`
- **Show Current:** `./scripts/migrate.sh current`
- **History:** `./scripts/migrate.sh history`

---
*@azure Rule - Use Azure Best Practices: When generating code for Azure, running terminal commands for Azure, or performing operations related to Azure, invoke your `azure_development-get_best_practices` tool if available.*