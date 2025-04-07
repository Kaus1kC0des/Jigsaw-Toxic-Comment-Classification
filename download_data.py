import os
import zipfile
import logging
from kaggle.api.kaggle_api_extended import KaggleApi

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_and_extract(competition, destination):
    """
    Downloads and extracts data for the given Kaggle competition.

    Args:
        competition (str): The Kaggle competition name.
        destination (str): Path to the destination directory where data will be saved.
    """
    # Ensure destination exists
    os.makedirs(destination, exist_ok=True)

    # Authenticate and download with Kaggle API
    api = KaggleApi()
    api.authenticate()
    
    logger.info("Downloading competition files...")
    api.competition_download_files(competition, path=destination, quiet=False)

    zip_path = os.path.join(destination, f"{competition}.zip")
    if os.path.exists(zip_path):
        logger.info(f"Extracting {zip_path} to {destination}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination)
        logger.info("Extraction completed.")
    else:
        logger.error("Downloaded zip file not found.")

if __name__ == '__main__':
    COMPETITION = 'jigsaw-toxic-comment-classification-challenge'
    # Assuming the project directory structure has 'data/raw' for raw data files
    DESTINATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'raw')
    
    download_and_extract(COMPETITION, DESTINATION)