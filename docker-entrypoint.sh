#!/bin/bash
set -e

# Activate virtual environment
source $VENV_PATH/bin/activate

# Wait for dependencies (if needed)
if [ ! -z "$WAIT_FOR_IT" ]; then
    for host_and_port in $(echo $WAIT_FOR_IT | tr "," "\n"); do
        /usr/local/bin/wait-for-it.sh $host_and_port -t 60
    done
fi

# Run database migrations
if [ "$RUN_MIGRATIONS" = "true" ]; then
    echo "Running database migrations..."
    alembic upgrade head
fi

# Start the application based on the command
case "$1" in
    "api")
        echo "Starting API server..."
        uvicorn src.main:app --host 0.0.0.0 --port 8000
        ;;
    "dashboard")
        echo "Starting dashboard..."
        streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
        ;;
    "worker")
        echo "Starting Celery worker..."
        celery -A src.tasks worker --loglevel=info
        ;;
    "scheduler")
        echo "Starting Celery beat..."
        celery -A src.tasks beat --loglevel=info
        ;;
    *)
        echo "Starting main application..."
        python -m src.main
        ;;
esac 