# Domain evaluation script 

import ast
import string

# Path to the evaluation file
eval_file_path = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/model_results/TOD1_mw24Domain_baseline/TOD1_mw24Domain_baseline_baseline.tsv'

# Load the evaluation file
with open(eval_file_path, 'r') as eval_file:
    eval_data = eval_file.readlines()

# Define translation table for removing punctuation
punctuation_table = str.maketrans('', '', string.punctuation)

empty_domains = 0
global_matching_domains = 0
global_total_domains = 0
empty_slots = 0

domain_prediction_count = 0 

for line in eval_data:
    extracted_text = ''
    # Extract prediction and ground truth as strings
    prediction_str = line.split('\t')[2]
    ground_truth_str = line.split('\t')[1]

    ground_truth = ast.literal_eval(ground_truth_str)

    end_idx = prediction_str.find(']')
    if end_idx != -1:
        extracted_text = prediction_str[:end_idx].strip()
    
    extracted_text = extracted_text.lower()
    extracted_text = ''.join(c for c in extracted_text if c.isalnum() or c == ',')
    prediction_list = extracted_text.split(',')

    # print(f'Ground truth: {ground_truth}')
    # print(f"Prediction: {prediction_list}")
    domain_prediction_count = 0

    total_domain_count = len(ground_truth)
    for predicted_domain in prediction_list:
        # print(f'predicted domain: {predicted_domain}')
        if predicted_domain == '':
            empty_domains += 1
        # print(f'ground truth: {ground_truth}')
        if predicted_domain in ground_truth:
            domain_prediction_count += 1
            # print("incremented")
        pass

    global_total_domains += total_domain_count
    global_matching_domains += domain_prediction_count


    # breakpoint()

print(f"Total Number of domains: {global_total_domains}")
print(f"Total correctly predicted domains: {global_matching_domains}")
print(f"Domain accuracy: {(global_matching_domains/global_total_domains)*100}")
print(f"Empty domains: {empty_domains}")
    
