# Take the path, list the tsv files inside the folder and combine them into a single tsv file. The output tsv will have the same name as the folder.

import os
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Combine domain results")
    parser.add_argument("--folder_path", type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/experiments/MW24/domain/MW24_MP_ST_PH-nm_SO_Labse_NN-10_UA_SV', help="Path to the folder containing the tsv files")
    parser.add_argument("--output_file", type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/experiments/MW24/domain/MW24_MP_ST_PH-nm_SO_Labse_NN-10_UA_SV.tsv', help="Path to the output file")
    return parser.parse_args()

def combine_tsv_files(folder_path, output_file):
    tsv_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".tsv")])  # Sort files alphabetically
    # print(f"Found {tsv_files} ")
    # breakpoint()
    all_data = []

    for tsv_file in tsv_files:
        file_path = os.path.join(folder_path, tsv_file)
        try:
            df = pd.read_csv(file_path, sep='\t', header=None, encoding='utf-8')  # Handle potential encoding issues
            all_data.append(df)
        except Exception as e:
            print(f"Error reading file {tsv_file}: {e}")
            continue

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.to_csv(output_file, sep='\t', header=False, index=False, encoding='utf-8')
        print(f"Combined data saved to {output_file}")
    else:
        print("No TSV files found or all files failed to load.")


def main():
    args = parse_args()
    folder_path = args.folder_path
    output_file = args.output_file
    combine_tsv_files(folder_path, output_file)


if __name__ == "__main__":
    main()