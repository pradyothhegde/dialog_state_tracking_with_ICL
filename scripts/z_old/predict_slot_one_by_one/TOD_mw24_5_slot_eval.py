import argparse

# Parse the arguments
parser = argparse.ArgumentParser(description='Evaluate the slot filling accuracy from the log file')
parser.add_argument('--eval_folder', default='/mnt/matylda4/hegde/int_ent/LLM_dialog_state/debug_model_out/TOD1_mw24_O7BI_U_P_10_DS_1_0/log_files/', type=str, help='Path to the log file')
# Path to your file
# eval_file = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/debug_model_out/TOD1_mw24_O7BI_U_P_10_DS_1_0/log_files_1/TOD1_mw24_O7BI_U_P_10_DS_1_0_7000_log.json'

args = parser.parse_args()
eval_file = args.eval_folder

# condsider all the json files in the folder
import os
import glob
eval_files = glob.glob(eval_file+'/*.json')

global_total_hits = 0
global_total_misses = 0
global_hits_count = 0
global_misses_count = 0
# Run the evaluation for each file
for eval_file in eval_files:
    print("Evaluating:", os.path.basename(eval_file))
    total_hits = 0
    total_misses = 0
    hits_count = 0
    misses_count = 0

    with open(eval_file, 'r') as f:
        for line in f:
            # Search for "hits" and add the values
            if '"hits":' in line:
                try:
                    # Extract the number after "hits": and add it to total_hits
                    hits_value = int(line.split('"hits":')[1].split(',')[0].strip())
                    total_hits += hits_value
                    hits_count += 1
                except ValueError:
                    print("Warning: Could not parse 'hits' value in line:", line)
            
            # Search for "misses" and add the values
            if '"misses":' in line:
                try:
                    # Extract the number after "misses": and add it to total_misses
                    misses_value = int(line.split('"misses":')[1].split(',')[0].strip())
                    total_misses += misses_value
                    misses_count += 1
                except ValueError:
                    print("Warning: Could not parse 'misses' value in line:", line)
    # print everything in one line
    print("Total Hits:", total_hits, "Total Misses:", total_misses, "Hits Count:", hits_count, "Misses Count:", misses_count, "Accuracy:", total_hits/(total_hits+total_misses))
    # print("Hits:", total_hits)
    # print("Misses:", total_misses)
    # print("Hits Count:", hits_count)
    # print("Misses Count:", misses_count)
    # print(f"Accuracy: {total_hits/(total_hits+total_misses)}")

    global_total_hits += total_hits
    global_total_misses += total_misses
    global_hits_count += hits_count
    global_misses_count += misses_count

print("Global Total Hits:", global_total_hits, "Global Total Misses:", global_total_misses, "Global Hits Count:", global_hits_count, "Global Misses Count:", global_misses_count, "Global Accuracy:", global_total_hits/(global_total_hits+global_total_misses))
