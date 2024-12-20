import json
import os

# import numpy as np
import pandas as pd
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Make a table of the TODs')
    parser.add_argument('--input_folder', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/O7BI/Slot/')
    parser.add_argument('--output_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/O7BI/TOD_results_table.csv')
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    input_folder = args.input_folder
    output_file = args.output_file

    # Get all the folders in the input folder
    folders = [f for f in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, f))]

    # Initialize the dataframe
    # df = pd.DataFrame(columns=['TOD', 'LLM', 'LLM_1', 'LLM_2', 'LLM_3', 'LLM_4', 'LLM_5', 'LLM_6', 'LLM_7', 'LLM_8', 'LLM_9', 'LLM_10'])

    # Loop over all the folders
    for folder in folders:
        # Get the TOD name
        print(folder)
        # get parameters from the folder name
        if '_UA' in folder:
            dialog_side = 'UA'
        elif '_U_' in folder:
            dialog_side = 'U'
        
        if '_PN' in folder:
            punct = 'PN'
        elif '_PY' in folder:
            punct = 'PY'

        if '_TN' in folder:
            tag = 'TN'
        elif '_TY' in folder:
            tag = 'TY'

        # get the file inside the folder the ends with results.json
        files = [f for f in os.listdir(os.path.join(input_folder, folder)) if f.endswith('results.json')]
        
        # read the json file
        with open(os.path.join(input_folder, folder, files[0]), 'r') as f:
            results = json.load(f)

        # get the dict value 'accuracy' from the json file
        accuracy = results['accuracy']
        na_accuracy = results['NA_accuracy']
        non_na_accuracy = results['Non_na_accuracy']

        # add the accuracy to the csv file with the corresponding parameters
        parameters = dialog_side + '_' + punct + '_' + tag
        df = pd.DataFrame({'TOD': [folder], 'parameters': parameters, parameters: [accuracy], 'NA_accuracy': [na_accuracy], 'Non_na_accuracy': [non_na_accuracy]})
        df.to_csv(output_file, mode='a', header=False, index=False)

        

if __name__ == '__main__':
    main()


        
