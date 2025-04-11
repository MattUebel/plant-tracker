# 🌱 Plant Tracker

A modern web application for gardeners to track and manage seeds, plantings, and growing progress.

## 🌟 Features

- **Seed Catalog Management**: Store information about seed varieties, brands, and growing requirements
- **Image Recognition**: Upload seed packet images with automatic information extraction
- **Planting Tracker**: Record and track plantings from seed to harvest
- **Transplant Timeline**: Document transplant events with locations and dates
- **Photo Gallery**: Attach and view multiple images for both seeds and plantings

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Frontend**: Bootstrap 5, Jinja2 templates
- **Deployment**: Docker and docker-compose
- **Image Processing**: OCR and structured data extraction via API

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Git

### Installation

#### Standard Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/[your-username]/plant-tracker.git
   cd plant-tracker
   ```

2. Start the application:
   ```bash
   docker compose up -d
   ```

3. Access the application in your web browser:
   ```
   http://localhost:8000
   ```

#### Raspberry Pi Installation (as a systemd service)

You can run Plant Tracker as a systemd service on a Raspberry Pi, which allows it to start automatically on boot and be managed as a system service.

Prerequisites:
- Raspberry Pi running Raspbian OS
- Git
- Internet connection
- Google Gemini API key for OCR functionality (get one from https://makersuite.google.com/)

To install:

1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/plant-tracker.git
   ```

2. Run the installation script:
   ```bash
   cd plant-tracker
   sudo ./scripts/install_service.sh
   ```
   
3. Follow the prompts to configure your installation, including adding your Gemini API key.

4. Once installed, you can manage the service with the following commands:
   ```bash
   plant-tracker start     # Start the service
   plant-tracker stop      # Stop the service
   plant-tracker restart   # Restart the service
   plant-tracker status    # Check service status
   plant-tracker logs      # View service logs
   plant-tracker update    # Update to latest version
   plant-tracker config    # Edit configuration file
   plant-tracker backup    # Create a database backup
   plant-tracker restore   # Restore from a backup
   ```

5. Access the application in your web browser at http://[raspberry-pi-ip]:8000

## 📸 Screenshots

*Coming soon*

## 📁 Project Structure

```
├── app/                  # Main application code
│   ├── models/           # SQLAlchemy database models
│   ├── routers/          # FastAPI route handlers
│   ├── templates/        # Jinja2 HTML templates
│   └── utils/            # Utility functions and classes
├── docs/                 # Documentation files
├── scripts/              # Utility scripts
│   ├── install_service.sh # Raspberry Pi systemd installation
│   ├── migrate.sh        # Database migration helper
│   └── psql.sh           # PostgreSQL management helper
├── uploads/              # Storage for uploaded images
├── compose.yml           # Docker Compose configuration
└── Dockerfile            # Container definition
```

## 🔧 Development

To run the application in development mode:

```bash
docker compose up
```

### Database Migrations

To create a new database migration after model changes:

```bash
./scripts/migrate.sh migrate "Description of changes"
./scripts/migrate.sh upgrade
```

Or using Docker Compose directly:

```bash
docker compose exec app alembic revision --autogenerate -m "Description of changes"
docker compose exec app alembic upgrade head
```

### Database Management

The `psql.sh` script provides convenient shortcuts for database operations:

```bash
./scripts/psql.sh shell     # Open a PostgreSQL shell
./scripts/psql.sh exec      # Execute SQL commands
./scripts/psql.sh dump      # Create a database backup
./scripts/psql.sh restore   # Restore from a backup
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- FastAPI for the amazing web framework
- SQLAlchemy for the ORM system
- Bootstrap for the frontend components
