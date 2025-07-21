import subprocess
import sys
import os

def install_requirements():
    """
    Install required packages from requirements.txt
    """
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def download_dataset():
    """
    Download the LongDocURL dataset
    """
    print("Downloading LongDocURL dataset...")
    try:
        from download_dataset import download_longdocurl_dataset
        return download_longdocurl_dataset()
    except ImportError as e:
        print(f"Error importing download script: {e}")
        return False

def main():
    """
    Main function to set up environment and download dataset
    """
    print("Setting up LongDocURL dataset download...")
    
    # Install requirements
    if not install_requirements():
        print("Failed to install requirements. Please check your Python environment.")
        return
    
    # Download dataset
    if not download_dataset():
        print("Failed to download dataset. Please check your Hugging Face login.")
        return
    
    print("Setup completed successfully!")

if __name__ == "__main__":
    main() 