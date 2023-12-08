#!/bin/bash


# Create a virtual environment
python3 -m venv venv

chmod +x shellfile.sh

chmod +x requirements.txt 

# Activate the virtual environment
source venv/bin/activate

# Install dependencies
# pip install -r results/requirements.txt

pip3 install -r requirements.txt

#to print the model accuracy test results
# python3 Model_Accuracy_Results.py

# Run your Python script
python3 testing.py

# Deactivate the virtual environment
deactivate


# Delete the virtual environment
rm -rf venv