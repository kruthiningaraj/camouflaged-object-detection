"""
Script to download CAMO dataset from Kaggle.
REQUIRED: Ensure Kaggle API key is configured locally.
"""
import os

def download_dataset():
    os.system("kaggle datasets download -d ivanomelchenkoim11/camo-dataset -p data/ --unzip")
    print("Dataset downloaded and extracted into 'data/' folder.")

if __name__ == "__main__":
    download_dataset()
