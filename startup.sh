#!/bin/bash

# Start the FastAPI app using gunicorn with Uvicorn worker
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
