#!/bin/bash

# Build and Run Script for Dockerizing Water Quality AI Predictor
echo "======================================================"
echo "🚀 Building Docker Image: water-quality-ai:latest"
echo "======================================================"

# Build the Docker image
docker build -t water-quality-ai:latest .

echo "======================================================"
echo "🎉 Image built successfully! Running on port 8080"
echo "======================================================"

# Run the Docker container
docker run -d -p 8080:8080 \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  --name water-quality-app \
  water-quality-ai:latest

echo "✅ Container is up and running in background!"
echo "📍 Access the app at http://localhost:8080"
