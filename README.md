# 🌱 Plant Tracker

A modern web application for gardeners to track and manage seeds, plantings, and growing progress.

## 🌟 Features

- **Seed Catalog Management**: Store information about seed varieties, brands, and growing requirements
- **Image Recognition**: Upload seed packet images with automatic information extraction via Google Gemini AI
- **OCR Technology**: Extract text and structured data from seed packets automatically
- **Planting Tracker**: Record and track plantings from seed to harvest
- **Transplant Timeline**: Document transplant events with locations and dates
- **Photo Gallery**: Attach and view multiple images for both seeds and plantings
- **Responsive Design**: Mobile-friendly interface works on phones, tablets, and desktop computers

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Frontend**: Bootstrap 5, Jinja2 templates
- **Deployment**: Docker and docker-compose
- **Image Processing**: OCR and structured data extraction via Google Gemini API
- **Asynchronous**: Built with async/await patterns for responsive performance

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- Git
- For OCR functionality: Google Gemini API key (get one from https://makersuite.google.com/)

### Installation

#### Standard Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MattUebel/plant-tracker.git
   cd plant-tracker
   ```

2. Create a `.env` file in the project root with your configuration:
   ```bash
   # Database configuration
   POSTGRES_PASSWORD=your_password_here
   POSTGRES_USER=postgres
   POSTGRES_DB=plant_tracker
   DB_PORT=5433

   # Application settings
   SECRET_KEY=your_secret_key_here
   APP_PORT=8000
   DEBUG=false
   ENVIRONMENT=development

   # API keys for vision processing
   GEMINI_API_KEY=your_api_key_here
   GEMINI_MODEL=gemini-2.5-pro

   # Optional: Alternative OCR provider
   # ANTHROPIC_API_KEY=your_api_key_here
   # CLAUDE_MODEL=claude-3-7-sonnet-20250219

   # Default OCR provider
   VISION_API_PROVIDER=gemini
   ```

3. Start the application:
   ```bash
   docker compose up -d
   ```

4. Access the application in your web browser:
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
   git clone https://github.com/MattUebel/plant-tracker.git
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

## 🔍 Key Features Explained

### Seed Packet OCR

The application uses Google's Gemini API to extract information from seed packet images:

1. Upload a seed packet image
2. The system automatically extracts text using OCR
3. AI processes the extracted text to identify key information like plant name, variety, planting depth, spacing, etc.
4. You can review and edit the extracted information before saving

### Planting Management

Track your plants from seed to harvest:

1. Create plantings from your seed catalog
2. Record sowing dates, locations, and conditions
3. Document transplant events when seedlings are moved
4. Add multiple photos to track growth over time
5. Record notes and observations throughout the growing season

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

For live log viewing:

```bash
docker compose logs -f
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

## 💡 Troubleshooting

### Common Issues

- **Database Connection Errors**: Ensure the database container is running with `docker compose ps`
- **Image Upload Issues**: Check that the uploads directory has proper permissions
- **OCR Not Working**: Verify your API key is correctly set in the .env file

### Logs

View application logs for more detailed error information:

```bash
docker compose logs app
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- FastAPI for the amazing web framework
- SQLAlchemy for the ORM system
- Bootstrap for the frontend components
- Google Gemini API for image recognition capabilities
