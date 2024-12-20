import json
import os
import pandas as pd
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Make a table of the TODs')
    parser.add_argument('--input_folder', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/O7BI/Domain/')
    parser.add_argument('--output_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/O7BI/TOD_results_table_extended_dom.csv')
    args = parser.parse_args()
    return args

def compute_accuracy(correct, total):
    """Helper function to compute accuracy safely."""
    return (correct / total * 100) if total != 0 else 0

def main():
    args = arg_parse()
    input_folder = args.input_folder
    output_file = args.output_file

    # Get all the folders in the input folder
    folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

    # Initialize the dataframe
    all_data = []

    # Loop over all the folders
    for folder in folders:
        # Get the TOD name
        print(folder)

        # Get parameters from the folder name
        dialog_side = 'UA' if '_UA' in folder else 'U'
        punct = 'PN' if '_PN' in folder else 'PY'
        tag = 'TN' if '_TN' in folder else 'TY'

        # Get the file inside the folder that ends with results.json
        files = [f for f in os.listdir(os.path.join(input_folder, folder)) if f.endswith('results.json')]
        if not files:
            continue

        # Read the JSON file
        with open(os.path.join(input_folder, folder, files[0]), 'r') as f:
            results = json.load(f)

        # Extract overall stats
        accuracy = results.get('accuracy', 0)
        na_accuracy = results.get('NA_accuracy', 0)
        non_na_accuracy = results.get('Non_na_accuracy', 0)

        # Initialize row dictionary
        row_data = {
            'TOD': folder,
            'parameters': f'{dialog_side}_{punct}_{tag}',
            'overall_accuracy': accuracy,
            'NA_accuracy': na_accuracy,
            'Non_na_accuracy': non_na_accuracy
        }

        # Extract and calculate domain-wise accuracies
        domain_stats = results.get('domain_wise_stats', {})
        for domain, stats in domain_stats.items():
            correct = stats.get('correct', 0)
            total = stats.get('total', 0)
            domain_accuracy = compute_accuracy(correct, total)
            row_data[f'{domain}_accuracy'] = domain_accuracy

        # Append row to the list of all data
        all_data.append(row_data)

    # Convert all data into a DataFrame and write to CSV
    df = pd.DataFrame(all_data)
    df.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()
