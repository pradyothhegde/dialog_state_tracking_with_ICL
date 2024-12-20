# sample run: 
# export CUDA_VISIBLE_DEVICES=$(free-gpus.sh 1); export HF_HUB_OFFLINE=1; export HF_DATASETS_OFFLINE=1; export TRANSFORMERS_OFFLINE=1; export HF_EVALUATE_OFFLINE=1; python TOD_open_ended.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
import numpy as np
import argparse
import ast
import json
import os
import platform

from debug_TODx import finding_test_labels
a = torch.rand(1, 1).cuda()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='allenai/OLMo-7B-Instruct', help='model name')
    parser.add_argument('--input_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/sandbox/LLM_input_sentences_zero_shot.txt')
    parser.add_argument('--data_path', type=str, default='/mnt/matylda4/hegde/int_ent/LLM_dialog_state/Data/MULTIWOZ2.4/',  help='Path to the data directory')
    parser.add_argument('--output_folder_path', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/zero_shot/')
    parser.add_argument('--output_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/sandbox/LLM_output_sentences_zero_shot.json')
    parser.add_argument('--offset_start', type=int, default=0, help='Start offset for the test file')
    parser.add_argument('--offset_add', type=int, default=8000, help='Add offset for the test file')
    return parser.parse_args()

dom2slots = {
    'taxi': ['leaveAt', 'destination', 'departure', 'arriveBy'], 
    'restaurant': ['people', 'day', 'time', 'food', 'pricerange', 'name', 'area'], 
    'attraction': ['type', 'name', 'area'], 
    'train': ['people', 'leaveAt', 'destination', 'day', 'arriveBy', 'departure'], 
    'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type']
}

def get_instruction(process_line):
    # find domain
    # print("Process line: ", process_line)
    process_line = ast.literal_eval(process_line)[0]
    # Domain: ["taxi"] Slots: {
    temp_text = process_line.split('\n')
    # get the last element in temp_text
    temp_text = temp_text[-1]
    templ_domain = temp_text.split('Domain: ')
    # get things within [ ]
    domain = templ_domain[1].split('[')[1].split(']')[0]
    # print("Domain: ", domain)
    # print("Last line: ", temp_text)
    domain_process_string = "[" + domain + "]"
    domains = ast.literal_eval(domain_process_string)
    # print(domains) # domains is a list

    # build instruction
    # Instruction: Identify slot names and values from the dialogue. The domains are taxi, restaurant. The slots for taxi are leaveAt, destination, departure, arriveBy. The slots for restaurant are people, day, time, food, pricerange, name, area.
    # if there is only one domain, then the instruction will the "The domain is taxi":
    if len(domains) == 1:
        instruction = "Identify slot names and values from the dialogue. The domain is "
    else:
        instruction = "Identify slot names and values from the dialogue. The domains are "
    for i in range(len(domains)):
        instruction += domains[i]
        if i != len(domains) - 1:
            instruction += ", "
        else:
            instruction += "."
    
    for i in range(len(domains)):
        instruction += " The slots for "
        instruction += domains[i] + " are "
        for j in range(len(dom2slots[domains[i]])):
            instruction += dom2slots[domains[i]][j]
            if j != len(dom2slots[domains[i]]) - 1:
                instruction += ", "
            else:
                instruction += "."
    # print(instruction)

    # breakpoint()
    # pass
    instruction = instruction + "\n" 
    return instruction

class NewlineStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, newline_token_id):
        self.tokenizer = tokenizer
        self.newline_token_id = newline_token_id

    def __call__(self, input_ids, scores, **kwargs):
        # Check if the most recent token is a newline
        return input_ids[0, -1] == self.newline_token_id

def main():
    args = parse_args()

    # Model and tokenizer
    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto")
    model = model.cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Get the token ID for newline
    newline_token_id = tokenizer.encode("\n", add_special_tokens=False)[0]

    output_folder_path = args.output_folder_path

    # Create log file
    log_folder = os.path.join(output_folder_path, 'log_files')
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    logg_file = os.path.join(log_folder, str(args.offset_start)+'_log.json')

    # Load input file
    input_file = args.input_file
    with open(input_file, 'r') as f:
        in_file = f.readlines()
    
    # Load test file
    test_tsv_file_path = os.path.join(output_folder_path, 'mw24_DST_test_U_punct_turns.tsv')
    with open(test_tsv_file_path, 'r') as tst_tsv_file:
        test_data = tst_tsv_file.readlines()

    # Create config file
    config_file = os.path.join(output_folder_path, str(args.offset_start)+'_config.json')
    with open(config_file, 'w') as config_file:
        config_file.write(json.dumps(vars(args), indent=4))
        # Add details of the run
        config_file.write(platform.node())
        config_file.write('\n')
        config_file.write(platform.platform())
        config_file.write('\n')


    counter = 0
    for i, (line, gold_line) in enumerate(zip(in_file, test_data)):
        if i < args.offset_start:
            counter += 1
            continue
        if i >= args.offset_start + args.offset_add:
            break
        counter += 1

        # get the gold line
        gold_fname, gold_sentence, gold_domain, gold_slots = finding_test_labels(gold_line)

        context = ast.literal_eval(line)[0]

        instruction = get_instruction(line)

        total_input_to_LLM = instruction + context

        # print(total_input_to_LLM)
        # print("-------------------")

        input_ids = tokenizer.encode(total_input_to_LLM, return_tensors='pt')
        input_ids = input_ids.cuda()
        
        # Define stopping criteria
        stopping_criteria = StoppingCriteriaList([NewlineStoppingCriteria(tokenizer, newline_token_id)])

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=200,
                return_dict_in_generate=True,
                output_scores=False,
                stopping_criteria=stopping_criteria
            )

        decoded_response = tokenizer.decode(output.sequences[0][input_ids.shape[-1]:], skip_special_tokens=True)
        # print(decoded_response)
        # write to file
        # print("========")

        # breakpoint()
        log_dictionary = {
            "line_number": counter,
            "gold_fname": gold_fname,
            "gold_domain": gold_domain,
            "gold_slots": gold_slots,
            "decoded_response": decoded_response, 
            # "processed_output": final_slot_dict,
            # "hits":  hits,
            # "misses": misses,
            # "total hits and misses = total number of slots" : hits+misses,
            # "number of slots present": slot_count,
            "total_input_to_LLM": total_input_to_LLM
            }

        with open(logg_file, 'a', encoding='utf-8') as log_file:
            log_file.write(json.dumps(log_dictionary))
            # log_file.write(json.dumps(log_dictionary, indent=4))
            log_file.write('\n')

if __name__ == '__main__':
    main()



# "User: Please help me with a taxi I'm going to pizza hut fen ditton. I am leaving from the beautiful saint catharine's college. Oh anytime after 11:45 a.m would be fine Domain: ["taxi"] Slots: {"taxi": {"leaveAt": "11:45", "destination": "pizza hut fenditton", "departure": "saint catharines college"}}\nUser: I need to get a taxi from emmanuel college I'm heading to finches bed and breakfast. Domain: ["taxi"] Slots: {"taxi": {"destination": "finches bed and breakfast", "departure": "emmanuel college"}}\nUser: Can you help me get a taxi to pizza express Fen Ditton? Domain: ["taxi"] Slots: {"taxi": {"destination": "pizza hut fenditton"}}\nUser: Can you help me get a taxi to pizza express Fen Ditton? I want to depart from sidney sussex college. I can't leave until after 11:45 please. Domain: ["taxi"] Slots: {"taxi": {"leaveAt": "after 11:45", "destination": "pizza hut fenditton", "departure": "sidney sussex college"}}\nUser: I need to book a taxi from bridge guest house to sidney sussex college. Domain: ["taxi"] Slots: {"taxi": {"destination": "sidney sussex college", "departure": "bridge guest house"}}\nUser: I need to book a taxi from Kings college to Pizza express Fen Ditton sometime after 17:15. I need the car type and contact number please. Domain: ["taxi"] Slots: {"taxi": {"leaveAt": "17:15", "destination": "pizza hut fenditton", "departure": "kings college"}}\nUser: I need to get a taxi out of holy trinity church. I'm going to saint johns chop house. Domain: ["taxi"] Slots: {"taxi": {"destination": "saint johns chop house", "departure": "holy trinity church"}}\nUser: I need to taxi from Ian hong house. I'd like to go to saint johns chop house. Domain: ["taxi"] Slots: {"taxi": {"destination": "saint johns chop house", "departure": "lan hong house"}}\nUser: Please help me with a taxi I'm going to pizza hut fen ditton. I am leaving from the beautiful saint catharine's college. Domain: ["taxi"] Slots: {"taxi": {"destination": "pizza hut fenditton", "departure": "saint catharines college"}}\nUser: Can you help me get a taxi to pizza express Fen Ditton? I want to depart from sidney sussex college. Domain: ["taxi"] Slots: {"taxi": {"destination": "pizza hut fenditton", "departure": "sidney sussex college"}}\nUser: I would like a taxi from Saint John's college to Pizza Hut Fen Ditton. Domain: ["taxi"] Slots: {"








# {"taxi": {"leaveAt": "N.A.", "destination": "cow pizza kitchen and bar", "departure": "el shaddia guesthouse", "arriveBy": "14:45"}, "restaurant": {"people": "6", "day": "tuesday", "time": "14:45", "food": "gastropub", "pricerange": "moderate", "name": "cow pizza kitchen and bar", "area": "centre"}, "hotel": {"people": "N.A.", "day": "N.A.", "stay": "N.A.", "name": "el shaddia guesthouse", "area": "N.A.", "parking": "yes", "pricerange": "N.A.", "stars": "0", "internet": "yes", "type": "N.A."}}