#!/bin/bash
# Script to download the required data files

echo "Setting up data directory..."

# Create data directories if they don't exist
mkdir -p data/raw
mkdir -p data/cleaned
mkdir -p data/features
mkdir -p data/model_inputs

# Check if data already exists
if [ -d "data/raw/tennis_atp-master" ]; then
    echo "Data directory already exists. Skipping download."
    exit 0
fi

# Download the tennis_atp dataset
echo "Downloading tennis_atp data from GitHub..."
cd data/raw
git clone https://github.com/JeffSackmann/tennis_atp.git tennis_atp-master

if [ $? -eq 0 ]; then
    echo "Data downloaded successfully!"
    echo "You can now run: python src/atp_forecaster/data/clean_data.py"
else
    echo "Download failed. Please download manually from:"
    echo "  https://github.com/JeffSackmann/tennis_atp"
    exit 1
fi

