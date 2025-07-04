import os
from pathlib import Path
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Project Name
project_name = 'UAE_CAR_PRICE_PREDICTION'

# List of required files and directories
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/__init__.py",
    f"src/components/__init__.py",
    f"src/pipeline/__init__.py",
    f"src/utils.py",
    f"src/logger.py",
    f"src/exception.py",
    "notebook/data/.gitkeep",
    "notebook/EDA_Price_Prediction.ipynb",
    "requirements.txt",
    "app.py",
    "setup.py",
    "templates/index.html"
]

# Create each file and directory as needed
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass  # create an empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")


""""
Explanation:
•	This script automatically creates the project folder structure and all the necessary files.
•	If the file already exists, it will be skipped.
•	Empty placeholder files like .gitkeep are used to maintain empty directories in Git.

"""