#!/bin/bash
set -e

echo "Setting up Heart Disease Prediction Project...."

echo "Installing ZenML integrations..."
uv run zenml integration install mlflow sklearn --uv -y

if [! -f .env]; then
    cp .env.example .env
    echo ".env File Created"
else
    echo ".env file exists already"
fi


mkdir -p data/external data/features data/interim data/processed data/raw
mkdir -p logs/api logs/inference logs/training
mkdir -p models/experiment models/production models/staging

echo ""
echo "Environment setup complete!"
echo "Next: run 'uv run scripts/setup_zenml_stack.py'"  