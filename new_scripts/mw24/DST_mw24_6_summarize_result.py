import argparse
import os
import csv
import json
from multiwoz_evaluation.evaluate import evaluate

def arg_parser():
    parser = argparse.ArgumentParser(description='Run evaluate.py on multiple JSON files.')
    parser.add_argument('--input_folder', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/results/mw24/baseline', help='Path to the folder containing JSON input files.')
    parser.add_argument('--output_csv', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/results/results_csv.csv', help='Path to the output CSV file.')
    return parser

def main():
    parser = arg_parser()
    args = parser.parse_args()
    input_folder = args.input_folder
    output_csv_path = args.output_csv
    golden_file = "/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/new_scripts/mw24/multiwoz_evaluation/speech_aware_dialogue/test-dstc11.2022-1102.gold.json"
    
    # list the files inside the input folder
    files = os.listdir(input_folder)
    
    with open(output_csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = ['folder_name', 'file_name'] # common headers
        # Extract keys from a sample dst dict to make headers
        sample_file = os.path.join(input_folder, files[0]) # gets the first file to get the keys.
        with open(sample_file, 'r') as f:
            sample_input_data = json.load(f)
        
        sample_results = evaluate(sample_input_data, bleu=False, dst=True, success=False, richness=False, golden=golden_file) # runs evaluate to get keys
        if sample_results and 'dst' in sample_results:
          header.extend(list(sample_results['dst'].keys()))
          
        csv_writer.writerow(header)

        for file in files:
            # full path
            file_path = os.path.join(input_folder, file)
            with open(file_path, 'r') as f:
                input_data = json.load(f)

                results = evaluate(input_data, bleu=False, dst=True, success=False, richness=False, golden=golden_file)   # load and evaluate the input data
                
                row = [os.path.basename(input_folder), file]  # add the file and folder name.

                if results and 'dst' in results:
                   row.extend(list(results['dst'].values())) # only add if dst key exists and there are values in it.
                csv_writer.writerow(row)
                

if __name__ == "__main__":
    main()