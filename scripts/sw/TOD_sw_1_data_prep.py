import os
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/Data/Spokenwoz')
    parser.add_argument('--output_dir', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/sw/data')
    return parser.parse_args()

def remove_words_field(data):
    """
    Recursively removes the 'words' field from a nested dictionary or list.

    Args:
        data (dict or list): The JSON data to process.

    Returns:
        dict or list: The modified JSON data without 'words' field.
    """
    if isinstance(data, dict):
        return {key: remove_words_field(value) for key, value in data.items() if key != "words"}
    elif isinstance(data, list):
        return [remove_words_field(item) for item in data]
    else:
        return data

def process_files(data, file_list, output_path):
    """
    Processes a list of files, removing the 'words' field and saving them in JSONL format.

    Args:
        data (dict): The full dataset to extract files from.
        file_list (list): List of file names to process.
        output_path (str): Path to save the processed JSONL file.
    """
    processed_data = []
    for file in file_list:
        if file in data:
            # Remove "words" field from the file's data
            cleaned_data = remove_words_field(data[file])
            processed_data.append({file: cleaned_data})
    
    # Save the processed data in JSONL format
    with open(output_path, 'w') as f:
        for line in processed_data:
            f.write(json.dumps(line) + '\n')

def main():
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    # Create output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load test data
    with open(f'{data_dir}/data_test.json', 'r') as f:
        data_test = json.load(f)

    # Load train and valid data
    with open(f'{data_dir}/data_train_valid.json', 'r') as f:
        data_train_valid = json.load(f)

    # Load file lists
    with open(f'{data_dir}/testListFile.json', 'r') as f:
        test_files = f.read().splitlines()
    with open(f'{data_dir}/valListFile.json', 'r') as f:
        valid_files = f.read().splitlines()

    # Determine train files (those not in valid or test files)
    all_files = set(data_train_valid.keys())
    valid_and_test_files = set(valid_files + test_files)
    train_files = list(all_files - valid_and_test_files)

    # Process and save test, valid, and train data
    print("Processing test files...")
    process_files(data_test, test_files, os.path.join(output_dir, 'test_data.json'))

    print("Processing valid files...")
    process_files(data_train_valid, valid_files, os.path.join(output_dir, 'valid_data.json'))

    print("Processing train files...")
    process_files(data_train_valid, train_files, os.path.join(output_dir, 'train_data.json'))

if __name__ == '__main__':
    main()
