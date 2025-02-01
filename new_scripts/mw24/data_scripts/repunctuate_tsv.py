# export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1); export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_EVALUATE_OFFLINE=1; /homes/kazi/hegde/miniconda3/envs/dialog/bin/python /mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/sandbox/repunctuate_tsv.py

import os # For file path manipulation
import argparse # For parsing command-line arguments
import re # For regular expressions (used for replacing <unk> tokens)
from typing import List # For type hinting
from punctuators.models import PunctCapSegModelONNX # For using punctuation model
from tqdm import tqdm

def parse_args(): # Sets up command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MW24/placeholder_vary_v1/MW24_MP_ST_PH-empty_SO_Labse_NN-10_U_SV/MW24_test_U_MP.tsv', type=str, help='Path to the input TSV file')
    parser.add_argument("--output_dir", default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MW24', type=str, help='Path to the output directory')
    return parser.parse_args()    

def replace_unk_to_dash(text):      # replace <unk> to '-'
    return re.sub(r'<[^>]*>', '-', text)

def remove_punc(s):  # Removes punctuation from a given string                                                                                                                              
    puncs = '!"#$%&()*+,./;<=>?@[\\]^_`{|}~' 
    return s.translate(str.maketrans('', '', puncs))


def punctuate_and_segment_texts(input_texts: List[str]) -> List[List[str]]:
    m: PunctCapSegModelONNX = PunctCapSegModelONNX.from_pretrained(
        "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
    )

    results: List[List[str]] = m.infer(texts=input_texts, apply_sbd=True)
    return results

def repunctuate_tsv(input_file): # Main function to process the TSV file

    with open(input_file, 'r', encoding='utf-8') as infile: # Open input file
        lines = [] # Initialize an empty list to store lines
        for line in infile: # Read input file line by line
            lines.append(line.strip().split('\t')) # Split line by tabs and append it to the lines list

    processed_lines = [] # Initialize an empty list to store processed lines
    for i in tqdm(range(0, len(lines), 1000)): # Process lines in batches of 20
        batch = lines[i:i+1000]

        texts_to_punctuate = [] # Initialize an empty list to store text to be punctuated
        for parts in batch: # Iterate through each line
            text_col = parts[1] # Get the second column which contains text
            split_texts = text_col.split("---") # Split text by "---"
            texts_to_punctuate.extend(split_texts) # Add the split texts to the list

        # remove punctuation
        texts_to_punctuate = [remove_punc(replace_unk_to_dash(text)) for text in texts_to_punctuate] # Remove punctuations from text
        punctuated_results = punctuate_and_segment_texts(texts_to_punctuate) # Punctuate the texts

        punctuated_texts = [] # Initialize an empty list to store punctuated lines
        index = 0 # Initialize index to keep track of results
        for parts in batch: # Iterate through each line again
            text_col = parts[1] # Get the second column text
            split_texts = text_col.split("---") # Split text with "---"
            replaced_text = "---".join([" ".join(punctuated_results[i]) for i in range(index, index+len(split_texts))]) # Reconstruct text with punctuated result
            parts[1] = replaced_text # Replace the original text with punctuated text
            punctuated_texts.append(parts) # Append the processed parts to the list
            index += len(split_texts) # Update the index

        processed_lines.extend(punctuated_texts) # Add the processed batch to the processed lines

    with open(input_file, 'w', encoding='utf-8') as outfile: # Open the output file to write back the processed lines
        for parts in processed_lines: # Iterate through each processed line
            outfile.write('\t'.join(parts) + '\n') # Join the parts with tab and write to the file


if __name__ == '__main__':
    args = parse_args() # Parse command-line arguments
    input_file = args.input_file # Get input file path
    output_dir = args.output_dir # Get output directory path
    input_file = '/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MW24/MW24_test_U_OP.tsv'
    repunctuate_tsv(input_file)