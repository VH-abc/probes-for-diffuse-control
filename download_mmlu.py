"""
Download the MMLU (Massive Multitask Language Understanding) dataset.
"""

from datasets import load_dataset
import os
import json

def download_mmlu():
    """
    Download the MMLU dataset from Hugging Face.
    The dataset will be cached in the default Hugging Face cache directory.
    """
    print("Downloading MMLU dataset...")

    # Download the MMLU dataset
    # The dataset has multiple subsets/tasks
    dataset = load_dataset("cais/mmlu", "all")

    print(f"\nDataset downloaded successfully!")
    print(f"Dataset structure: {dataset}")
    print(f"\nSplits available: {list(dataset.keys())}")

    # Print some information about the dataset
    for split in dataset.keys():
        print(f"\n{split} split: {len(dataset[split])} examples")

    # Print example from the validation set
    if 'validation' in dataset:
        print(f"\nExample from validation set:")
        print(dataset['validation'][0])
    elif 'dev' in dataset:
        print(f"\nExample from dev set:")
        print(dataset['dev'][0])

    # Save the dataset to disk
    data_dir = "mmlu_data"
    os.makedirs(data_dir, exist_ok=True)

    print(f"\nSaving dataset to {data_dir}/...")
    for split in dataset.keys():
        output_path = os.path.join(data_dir, f"{split}.json")
        dataset[split].to_json(output_path)
        print(f"  Saved {split} split to {output_path}")

    return dataset

if __name__ == "__main__":
    dataset = download_mmlu()
    print("\n✓ MMLU dataset is ready to use and saved to disk!")

