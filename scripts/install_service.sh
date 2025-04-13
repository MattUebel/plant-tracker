#!/bin/bash
# install_service.sh - Plant Tracker Installation Script for Raspberry Pi
# This script installs Plant Tracker as a systemd service on Raspberry Pi OS

set -e

# Configuration variables - adjust these as needed
REPO_URL="https://github.com/MattUebel/plant-tracker.git"
INSTALL_DIR="/opt/plant-tracker"
SERVICE_NAME="plant-tracker"
ENV_FILE="${INSTALL_DIR}/.env"
POSTGRES_PASSWORD=$(tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c 16)
APP_SECRET_KEY=$(tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c 32)

# Display a colorful banner
show_banner() {
    echo -e "\e[32m"
    echo "==============================================="
    echo "  🌱  Plant Tracker Installation Script  🌱"
    echo "==============================================="
    echo -e "\e[0m"
}

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then 
        echo "Please run as root (sudo)"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    echo -e "\n\e[34m📦 Installing dependencies...\e[0m"
    
    # Update package list
    apt-get update
    
    # Install required packages
    apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        git
        
    # Install Docker if not already installed
    if ! command -v docker &> /dev/null; then
        echo "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sh get-docker.sh
        usermod -aG docker pi
        rm get-docker.sh
    else
        echo "Docker already installed."
    fi
    
    # Install Docker Compose V2 plugin if not already installed
    if ! docker compose version &> /dev/null; then
        echo "Installing Docker Compose V2 plugin..."
        
        # For Debian/Ubuntu/Raspbian
        if command -v apt-get &> /dev/null; then
            apt-get install -y docker-compose-plugin
        # For other systems, try direct installation
        else
            mkdir -p /usr/local/lib/docker/cli-plugins
            curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" -o /usr/local/lib/docker/cli-plugins/docker-compose
            chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
        fi
        
        echo "Docker Compose V2 plugin installed."
    else
        echo "Docker Compose V2 already installed."
    fi
}

# Clone or update the repository
clone_repository() {
    echo -e "\n\e[34m📥 Setting up repository...\e[0m"
    
    if [ -d "$INSTALL_DIR" ]; then
        echo "Installation directory already exists."
        if [ -d "${INSTALL_DIR}/.git" ]; then
            echo "Checking for updates..."
            cd "$INSTALL_DIR"
            
            # Stash any local changes
            git stash -q || true
            
            # Check if there are updates
            git fetch
            LOCAL=$(git rev-parse HEAD)
            REMOTE=$(git rev-parse @{u})
            
            if [ "$LOCAL" != "$REMOTE" ]; then
                echo "Updates available. Pulling changes..."
                
                # Stop the service before updating
                if systemctl is-active --quiet "$SERVICE_NAME"; then
                    echo "Stopping service to update..."
                    systemctl stop "$SERVICE_NAME"
                fi
                
                git pull
                echo "Code updated."
                
                # Will restart later in the script
            else
                echo "Repository is up to date."
            fi
        else
            echo "Directory exists but is not a git repository. Backing up and cloning fresh..."
            mv "$INSTALL_DIR" "${INSTALL_DIR}.backup.$(date +%Y%m%d%H%M%S)"
            git clone "$REPO_URL" "$INSTALL_DIR"
        fi
    else
        echo "Cloning repository to $INSTALL_DIR..."
        git clone "$REPO_URL" "$INSTALL_DIR"
    fi
    
    # Enter the directory
    cd "$INSTALL_DIR"
}

# Create or verify environment file
create_env_file() {
    echo -e "\n\e[34m🔧 Setting up environment file...\e[0m"
    
    if [ -f "$ENV_FILE" ]; then
        echo ".env file exists. Checking configuration..."
        
        # Source the environment file to get variables
        source "$ENV_FILE"
        
        # Check if required configuration exists
        if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_api_key_here" ]; then
            echo -e "\e[31mError: GEMINI_API_KEY is not set in $ENV_FILE\e[0m"
            echo "The Plant Tracker requires a Gemini API key for OCR functionality."
            echo -e "Please get an API key from \e[36mhttps://makersuite.google.com/\e[0m"
            echo -e "Then edit $ENV_FILE and set the GEMINI_API_KEY value.\n"
            
            read -p "Would you like to continue installation without OCR functionality? (y/N): " CONTINUE
            
            if [[ ! $CONTINUE =~ ^[Yy]$ ]]; then
                echo "Installation aborted. Please update the .env file and run the script again."
                exit 1
            fi
            
            echo "Continuing installation without OCR functionality..."
        else
            echo "Required API keys are present in .env file."
        fi
    else
        echo ".env file doesn't exist. Creating template..."
        
        cat > "$ENV_FILE" << EOL
# Database configuration
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_USER=postgres
POSTGRES_DB=plant_tracker
DB_PORT=5433

# Application settings
SECRET_KEY=${APP_SECRET_KEY}
APP_PORT=8000
DEBUG=false
ENVIRONMENT=production

# API keys for vision processing (MUST BE UPDATED BEFORE RUNNING THE APP!)
# Get Gemini API key from https://makersuite.google.com/
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-pro

# Optional: Anthropic API key (alternative OCR provider)
# Get API key from https://console.anthropic.com/
ANTHROPIC_API_KEY=
CLAUDE_MODEL=claude-3-7-sonnet-20250219

# Default OCR provider
VISION_API_PROVIDER=gemini
EOL
        echo ".env file template created at $ENV_FILE"
        echo -e "\e[33m"
        echo "======================= IMPORTANT NOTICE ========================"
        echo "You MUST edit $ENV_FILE and add your Gemini API key"
        echo "before continuing with the installation."
        echo "Get an API key from: https://makersuite.google.com/"
        echo "================================================================="
        echo -e "\e[0m"
        
        read -p "Would you like to edit the .env file now? (Y/n): " EDIT_ENV
        
        if [[ ! $EDIT_ENV =~ ^[Nn]$ ]]; then
            # Try to open with user's preferred editor
            ${EDITOR:-nano} "$ENV_FILE"
        else
            echo "Please edit the .env file before continuing."
            exit 1
        fi
        
        echo -e "\n\e[34mVerifying API key configuration...\e[0m"
        source "$ENV_FILE"
        
        if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_api_key_here" ]; then
            echo -e "\e[31mError: GEMINI_API_KEY is still not configured in $ENV_FILE\e[0m"
            echo "Installation aborted. Please update the .env file and run the script again."
            exit 1
        fi
        
        echo -e "\e[32mAPI key configuration verified.\e[0m"
    fi
}

# Create systemd service file
create_service() {
    echo -e "\n\e[34m🔧 Creating systemd service...\e[0m"
    
    # Check if service file exists and has different content
    local SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
    local NEEDS_UPDATE=1
    
    if [ -f "$SERVICE_PATH" ]; then
        echo "Service file exists. Checking if it needs updates..."
        
        # Create a temporary file with our desired content
        local TEMP_SERVICE=$(mktemp)
        cat > "$TEMP_SERVICE" << EOL
[Unit]
Description=Plant Tracker Application
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${ENV_FILE}
ExecStartPre=/usr/bin/docker compose pull
ExecStart=/usr/bin/docker compose up
ExecStop=/usr/bin/docker compose down
Restart=always
RestartSec=10s

[Install]
WantedBy=multi-user.target
EOL

        # Compare existing with new
        if diff -q "$SERVICE_PATH" "$TEMP_SERVICE" >/dev/null; then
            echo "Service file is up-to-date."
            NEEDS_UPDATE=0
        else
            echo "Service file needs updating."
        fi
        
        rm "$TEMP_SERVICE"
    fi
    
    if [ $NEEDS_UPDATE -eq 1 ]; then
        # Create or update the service file
        cat > "$SERVICE_PATH" << EOL
[Unit]
Description=Plant Tracker Application
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=${ENV_FILE}
ExecStartPre=/usr/bin/docker compose pull
ExecStart=/usr/bin/docker compose up
ExecStop=/usr/bin/docker compose down
Restart=always
RestartSec=10s

[Install]
WantedBy=multi-user.target
EOL
        echo "Systemd service file created/updated."
        
        # Reload systemd to recognize the new/changed service
        systemctl daemon-reload
    fi
}

# Setup uploads directory with proper permissions
setup_uploads_directory() {
    echo -e "\n\e[34m📁 Setting up uploads directory...\e[0m"
    
    mkdir -p "${INSTALL_DIR}/uploads"
    chmod 777 "${INSTALL_DIR}/uploads"
    echo "Uploads directory set up with proper permissions."
}

# Create wrapper script for managing the service
create_wrapper_script() {
    echo -e "\n\e[34m📝 Creating management script...\e[0m"
    
    local SCRIPT_PATH="/usr/local/bin/plant-tracker"
    local NEEDS_UPDATE=1
    
    if [ -f "$SCRIPT_PATH" ]; then
        echo "Management script exists. Checking if it needs updates..."
        
        # Create a temporary file with our desired content
        local TEMP_SCRIPT=$(mktemp)
        cat > "$TEMP_SCRIPT" << EOL
#!/bin/bash
# Plant Tracker management script

cd "${INSTALL_DIR}"

# Load environment variables
if [ -f "${INSTALL_DIR}/.env" ]; then
  export \$(grep -v '^#' "${INSTALL_DIR}/.env" | xargs)
fi

case "\$1" in
    start)
        systemctl start ${SERVICE_NAME}
        echo "Service started."
        ;;
    stop)
        systemctl stop ${SERVICE_NAME}
        echo "Service stopped."
        ;;
    restart)
        systemctl restart ${SERVICE_NAME}
        echo "Service restarted."
        ;;
    status)
        systemctl status ${SERVICE_NAME}
        ;;
    logs)
        journalctl -u ${SERVICE_NAME} -f
        ;;
    update)
        echo "Updating Plant Tracker..."
        git stash -q
        git fetch
        LOCAL=\$(git rev-parse HEAD)
        REMOTE=\$(git rev-parse @{u})
        
        if [ "\$LOCAL" != "\$REMOTE" ]; then
            systemctl stop ${SERVICE_NAME}
            git pull
            docker compose pull
            systemctl start ${SERVICE_NAME}
            echo "Plant Tracker updated and restarted."
        else
            echo "Already up-to-date."
        fi
        ;;
    config)
        echo "Opening .env configuration file..."
        \${EDITOR:-nano} ${INSTALL_DIR}/.env
        echo "Configuration updated. Restart the service to apply changes:"
        echo "plant-tracker restart"
        ;;
    db)
        ${INSTALL_DIR}/scripts/psql.sh "\$2"
        ;;
    migrate)
        ${INSTALL_DIR}/scripts/migrate.sh "\$2"
        ;;
    backup)
        BACKUP_DIR="\${2:-/home/pi/backups}"
        mkdir -p "\$BACKUP_DIR"
        BACKUP_FILE="\$BACKUP_DIR/plant-tracker-\$(date +%Y%m%d%H%M%S).sql"
        echo "Creating backup at \$BACKUP_FILE..."
        ${INSTALL_DIR}/scripts/psql.sh dump "\$BACKUP_FILE"
        echo "Backup created successfully."
        ;;
    restore)
        if [ -z "\$2" ]; then
            echo "Error: Please provide backup file path"
            echo "Usage: plant-tracker restore /path/to/backup.sql"
            exit 1
        fi
        
        if [ ! -f "\$2" ]; then
            echo "Error: Backup file not found: \$2"
            exit 1
        fi
        
        echo "Restoring from backup \$2..."
        systemctl stop ${SERVICE_NAME}
        ${INSTALL_DIR}/scripts/psql.sh restore "\$2"
        systemctl start ${SERVICE_NAME}
        echo "Restore completed and service restarted."
        ;;
    *)
        echo "Usage: plant-tracker COMMAND"
        echo ""
        echo "Commands:"
        echo "  start       Start the service"
        echo "  stop        Stop the service"
        echo "  restart     Restart the service"
        echo "  status      Check service status"
        echo "  logs        View service logs"
        echo "  update      Update to latest version"
        echo "  config      Edit configuration file"
        echo "  db          Run database commands"
        echo "  migrate     Run database migrations"
        echo "  backup      Create database backup"
        echo "  restore     Restore from backup"
        exit 1
        ;;
esac
EOL

        # Compare existing with new
        if diff -q "$SCRIPT_PATH" "$TEMP_SCRIPT" >/dev/null; then
            echo "Management script is up-to-date."
            NEEDS_UPDATE=0
        else
            echo "Management script needs updating."
        fi
        
        rm "$TEMP_SCRIPT"
    fi
    
    if [ $NEEDS_UPDATE -eq 1 ]; then
        # Create or update the script
        cat > "$SCRIPT_PATH" << EOL
#!/bin/bash
# Plant Tracker management script

cd "${INSTALL_DIR}"

# Load environment variables
if [ -f "${INSTALL_DIR}/.env" ]; then
  export \$(grep -v '^#' "${INSTALL_DIR}/.env" | xargs)
fi

case "\$1" in
    start)
        systemctl start ${SERVICE_NAME}
        echo "Service started."
        ;;
    stop)
        systemctl stop ${SERVICE_NAME}
        echo "Service stopped."
        ;;
    restart)
        systemctl restart ${SERVICE_NAME}
        echo "Service restarted."
        ;;
    status)
        systemctl status ${SERVICE_NAME}
        ;;
    logs)
        journalctl -u ${SERVICE_NAME} -f
        ;;
    update)
        echo "Updating Plant Tracker..."
        git stash -q
        git fetch
        LOCAL=\$(git rev-parse HEAD)
        REMOTE=\$(git rev-parse @{u})
        
        if [ "\$LOCAL" != "\$REMOTE" ]; then
            systemctl stop ${SERVICE_NAME}
            git pull
            docker compose pull
            systemctl start ${SERVICE_NAME}
            echo "Plant Tracker updated and restarted."
        else
            echo "Already up-to-date."
        fi
        ;;
    config)
        echo "Opening .env configuration file..."
        \${EDITOR:-nano} ${INSTALL_DIR}/.env
        echo "Configuration updated. Restart the service to apply changes:"
        echo "plant-tracker restart"
        ;;
    db)
        ${INSTALL_DIR}/scripts/psql.sh "\$2"
        ;;
    migrate)
        ${INSTALL_DIR}/scripts/migrate.sh "\$2"
        ;;
    backup)
        BACKUP_DIR="\${2:-/home/pi/backups}"
        mkdir -p "\$BACKUP_DIR"
        BACKUP_FILE="\$BACKUP_DIR/plant-tracker-\$(date +%Y%m%d%H%M%S).sql"
        echo "Creating backup at \$BACKUP_FILE..."
        ${INSTALL_DIR}/scripts/psql.sh dump "\$BACKUP_FILE"
        echo "Backup created successfully."
        ;;
    restore)
        if [ -z "\$2" ]; then
            echo "Error: Please provide backup file path"
            echo "Usage: plant-tracker restore /path/to/backup.sql"
            exit 1
        fi
        
        if [ ! -f "\$2" ]; then
            echo "Error: Backup file not found: \$2"
            exit 1
        fi
        
        echo "Restoring from backup \$2..."
        systemctl stop ${SERVICE_NAME}
        ${INSTALL_DIR}/scripts/psql.sh restore "\$2"
        systemctl start ${SERVICE_NAME}
        echo "Restore completed and service restarted."
        ;;
    *)
        echo "Usage: plant-tracker COMMAND"
        echo ""
        echo "Commands:"
        echo "  start       Start the service"
        echo "  stop        Stop the service"
        echo "  restart     Restart the service"
        echo "  status      Check service status"
        echo "  logs        View service logs"
        echo "  update      Update to latest version"
        echo "  config      Edit configuration file"
        echo "  db          Run database commands"
        echo "  migrate     Run database migrations"
        echo "  backup      Create database backup"
        echo "  restore     Restore from backup"
        exit 1
        ;;
esac
EOL

        chmod +x "$SCRIPT_PATH"
        echo "Management script created/updated at $SCRIPT_PATH"
    fi
}

# Start or restart service
manage_service() {
    echo -e "\n\e[34m🚀 Managing service...\e[0m"
    
    # Enable service to start on boot if not already enabled
    if ! systemctl is-enabled --quiet "$SERVICE_NAME"; then
        systemctl enable "$SERVICE_NAME"
        echo "Service enabled to start on boot."
    else
        echo "Service already enabled on boot."
    fi
    
    # Start or restart service
    if systemctl is-active --quiet "$SERVICE_NAME"; then
        echo "Service is running. Restarting..."
        systemctl restart "$SERVICE_NAME"
        echo "Service restarted."
    else
        echo "Starting service..."
        systemctl start "$SERVICE_NAME"
        echo "Service started."
    fi
    
    # Show status
    echo "Service status:"
    systemctl status "$SERVICE_NAME" --no-pager
}

# Show completion message with helpful info
show_completion() {
    IP=$(hostname -I | awk '{print $1}')
    
    echo -e "\n\e[32m✅ Plant Tracker setup complete!\e[0m"
    echo -e "\n\e[33mAccess your Plant Tracker at:\e[0m http://${IP}:8000"
    echo -e "\n\e[33mManage your Plant Tracker with these commands:\e[0m"
    echo "  plant-tracker start     - Start the service"
    echo "  plant-tracker stop      - Stop the service"
    echo "  plant-tracker restart   - Restart the service"
    echo "  plant-tracker status    - Check service status"
    echo "  plant-tracker logs      - View service logs"
    echo "  plant-tracker update    - Update to latest version"
    echo "  plant-tracker config    - Edit configuration file"
    echo "  plant-tracker backup    - Create a database backup"
    echo "  plant-tracker restore   - Restore from a backup"
    echo "  plant-tracker db        - Access database functions"
    echo "  plant-tracker migrate   - Run database migrations"
    echo -e "\n\e[33mInstallation directory:\e[0m ${INSTALL_DIR}"
    echo -e "\e[33mConfiguration file:\e[0m ${ENV_FILE}"
    
    # Check if GEMINI_API_KEY is set properly
    source "$ENV_FILE"
    if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_api_key_here" ]; then
        echo -e "\n\e[31mNOTE: Gemini API key is not configured.\e[0m"
        echo -e "OCR functionality will not work until you set your API key:"
        echo -e "  plant-tracker config"
    fi
}

# Run database migrations if needed
run_initial_migrations() {
    echo -e "\n\e[34m🔄 Running initial database setup...\e[0m"
    
    # Wait for database to be ready
    echo "Waiting for database to be ready..."
    sleep 10
    
    # Run migrations
    echo "Running database migrations..."
    "${INSTALL_DIR}/scripts/migrate.sh" upgrade
    
    echo "Database setup complete."
}

# Main installation function
main() {
    show_banner
    check_root
    install_dependencies
    clone_repository
    create_env_file
    setup_uploads_directory
    create_service
    create_wrapper_script
    manage_service
    run_initial_migrations
    show_completion
}

# Run the main function
main