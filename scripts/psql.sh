#!/bin/bash
set -e

# Determine the project root directory (one level up from this script)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Load environment variables from .env file if it exists
if [ -f "$PROJECT_ROOT/.env" ]; then
  export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
fi

# Help message
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
  echo "Usage: scripts/psql.sh [COMMAND]"
  echo ""
  echo "Commands:"
  echo "  shell       Open a psql shell in the database"
  echo "  exec        Execute a SQL command (e.g. 'scripts/psql.sh exec \"SELECT * FROM seeds;\"')"
  echo "  dump        Dump the database to a file (default: backup.sql)"
  echo "  restore     Restore the database from a file (default: backup.sql)"
  echo "  createdb    Create a new database"
  echo "  dropdb      Drop a database"
  echo "  --help      Show this help message"
  exit 0
fi

# Run PostgreSQL commands in the database container
case "$1" in
  shell)
    docker compose exec db psql -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-postgres}
    ;;
  exec)
    shift
    if [ -z "$1" ]; then
      echo "Error: No SQL command provided"
      echo "Usage: scripts/psql.sh exec \"SQL COMMAND\""
      exit 1
    fi
    docker compose exec db psql -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-postgres} -c "$1"
    ;;
  dump)
    OUTPUT_FILE="${2:-backup.sql}"
    echo "Dumping database to $OUTPUT_FILE..."
    docker compose exec db pg_dump -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-postgres} > "$OUTPUT_FILE"
    echo "Database dumped to $OUTPUT_FILE"
    ;;
  restore)
    INPUT_FILE="${2:-backup.sql}"
    if [ ! -f "$INPUT_FILE" ]; then
      echo "Error: File $INPUT_FILE not found"
      exit 1
    fi
    echo "Restoring database from $INPUT_FILE..."
    cat "$INPUT_FILE" | docker compose exec -T db psql -U ${POSTGRES_USER:-postgres} -d ${POSTGRES_DB:-postgres}
    echo "Database restored from $INPUT_FILE"
    ;;
  createdb)
    shift
    DB_NAME="${1:-${POSTGRES_DB:-postgres}}"
    docker compose exec db createdb -U ${POSTGRES_USER:-postgres} "$DB_NAME"
    echo "Database $DB_NAME created"
    ;;
  dropdb)
    shift
    DB_NAME="${1:-${POSTGRES_DB:-postgres}}"
    echo "Are you sure you want to drop database $DB_NAME? [y/N]"
    read -r CONFIRM
    if [[ $CONFIRM =~ ^[Yy]$ ]]; then
      docker compose exec db dropdb -U ${POSTGRES_USER:-postgres} "$DB_NAME"
      echo "Database $DB_NAME dropped"
    else
      echo "Operation cancelled"
    fi
    ;;
  *)
    echo "Unknown command: $1"
    echo "Run 'scripts/psql.sh --help' for usage information."
    exit 1
    ;;
esac
