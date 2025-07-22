import os
from datasets import load_dataset

def download_longdocurl_dataset():
    """
    Download the LongDocURL dataset from Hugging Face and save it to the dataset directory.
    """
    print("Starting download of LongDocURL dataset...")
    
    # Create dataset directory if it doesn't exist
    dataset_dir = "LongDocURL"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"Created directory: {dataset_dir}")
    
    try:
        # Load the dataset
        print("Loading dataset from Hugging Face...")
        ds = load_dataset("dengchao/LongDocURL")
        
        # Save the dataset to the local directory
        print("Saving dataset to local directory...")
        ds.save_to_disk(dataset_dir)
        
        print("Dataset downloaded successfully!")
        print(f"Dataset saved to: {dataset_dir}")
        
        # Print dataset information
        print("\nDataset information:")
        print(f"Number of splits: {len(ds)}")
        for split_name, split_data in ds.items():
            print(f"  {split_name}: {len(split_data)} examples")
        
        # Show sample data structure
        if len(ds) > 0:
            first_split = list(ds.keys())[0]
            sample = ds[first_split][0] if len(ds[first_split]) > 0 else None
            if sample:
                print(f"\nSample data structure from {first_split} split:")
                for key, value in sample.items():
                    if isinstance(value, str) and len(value) > 100:
                        print(f"  {key}: {value[:100]}... (truncated)")
                    else:
                        print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Make sure you have logged in to Hugging Face using 'huggingface-cli login'")
        return False
    
    return True

if __name__ == "__main__":
    download_longdocurl_dataset() 