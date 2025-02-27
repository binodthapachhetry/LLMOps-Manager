#!/bin/bash
set -e

# Create necessary directories if they don't exist
mkdir -p test_model_cache test_prompt_cache

# Run unit tests
echo "Running unit tests..."
python -m pytest tests/unit -v

# Run integration tests
echo "Running integration tests..."
python -m pytest tests/integration -v

# Run with coverage
echo "Running tests with coverage..."
python -m pytest --cov=src --cov-report=term --cov-report=html

echo "Tests completed successfully!"
