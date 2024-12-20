import ast
import argparse
import json
import os
import json_repair

def arg_parse():
    parser = argparse.ArgumentParser(description='Evaluate open ended slot prediction.')
    parser.add_argument('--input_file_path', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/NA_removed/log_files/', help='input file')
    # parser.add_argument('--output_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/sandbox/LLM_input_sentences_zero_shot.txt', help='output file')
    args = parser.parse_args()
    return args

def main():
    args = arg_parse()
    eval_file = args.input_file_path
    # output_file = args.output_file
    # for all .json files in the input directory


    dom2slots = {
    'taxi': ['leaveAt', 'destination', 'departure', 'arriveBy'], 
    'restaurant': ['people', 'day', 'time', 'food', 'pricerange', 'name', 'area'], 
    'attraction': ['type', 'name', 'area'], 
    'train': ['people', 'leaveAt', 'destination', 'day', 'arriveBy', 'departure'], 
    'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type']
}
    

    for file in os.listdir(eval_file):
        if file.endswith('.json'):
            print(file)
            log_file = os.path.join(eval_file, file)
            with open(log_file, 'r') as f:
                eval_data = f.readlines()
            for line in eval_data:
                line_dict = json.loads(line.strip())

                # line_dict = {"line_number": 1, "gold_fname": "SNG0073.json", "gold_domain": "['taxi']", "gold_slots": "[{'leaveAt': 'N.A.', 'destination': 'pizza hut fenditton', 'departure': 'saint johns college', 'arriveBy': 'N.A.'}]", "decoded_response": " \"taxi\": { \"destination\": \"Pizza Hut Fen Ditton\", \"departure\": \"Saint John's college\" } }", "total_input_to_LLM": "Identify slot names and values from the dialogue. The domain is taxi. The slots for taxi are leaveAt, destination, departure, arriveBy.\nUser: Please help me with a taxi I'm going to pizza hut fen ditton. I am leaving from the beautiful saint catharine's college. Oh anytime after 11:45 a.m would be fine Domain: [\"taxi\"] Slots: {\"taxi\": {\"leaveAt\": \"11:45\", \"destination\": \"pizza hut fenditton\", \"departure\": \"saint catharines college\"}}\nUser: I need to get a taxi from emmanuel college I'm heading to finches bed and breakfast. Domain: [\"taxi\"] Slots: {\"taxi\": {\"destination\": \"finches bed and breakfast\", \"departure\": \"emmanuel college\"}}\nUser: Can you help me get a taxi to pizza express Fen Ditton? Domain: [\"taxi\"] Slots: {\"taxi\": {\"destination\": \"pizza hut fenditton\"}}\nUser: Can you help me get a taxi to pizza express Fen Ditton? I want to depart from sidney sussex college. I can't leave until after 11:45 please. Domain: [\"taxi\"] Slots: {\"taxi\": {\"leaveAt\": \"after 11:45\", \"destination\": \"pizza hut fenditton\", \"departure\": \"sidney sussex college\"}}\nUser: I need to book a taxi from bridge guest house to sidney sussex college. Domain: [\"taxi\"] Slots: {\"taxi\": {\"destination\": \"sidney sussex college\", \"departure\": \"bridge guest house\"}}\nUser: I need to book a taxi from Kings college to Pizza express Fen Ditton sometime after 17:15. I need the car type and contact number please. Domain: [\"taxi\"] Slots: {\"taxi\": {\"leaveAt\": \"17:15\", \"destination\": \"pizza hut fenditton\", \"departure\": \"kings college\"}}\nUser: I need to get a taxi out of holy trinity church. I'm going to saint johns chop house. Domain: [\"taxi\"] Slots: {\"taxi\": {\"destination\": \"saint johns chop house\", \"departure\": \"holy trinity church\"}}\nUser: I need to taxi from Ian hong house. I'd like to go to saint johns chop house. Domain: [\"taxi\"] Slots: {\"taxi\": {\"destination\": \"saint johns chop house\", \"departure\": \"lan hong house\"}}\nUser: Please help me with a taxi I'm going to pizza hut fen ditton. I am leaving from the beautiful saint catharine's college. Domain: [\"taxi\"] Slots: {\"taxi\": {\"destination\": \"pizza hut fenditton\", \"departure\": \"saint catharines college\"}}\nUser: Can you help me get a taxi to pizza express Fen Ditton? I want to depart from sidney sussex college. Domain: [\"taxi\"] Slots: {\"taxi\": {\"destination\": \"pizza hut fenditton\", \"departure\": \"sidney sussex college\"}}\nUser: I would like a taxi from Saint John's college to Pizza Hut Fen Ditton. Domain: [\"taxi\"] Slots: {"}

                # Retrieve gold slots
                gold_domain = ast.literal_eval(line_dict['gold_domain'])
                gold_slots = ast.literal_eval(line_dict['gold_slots'])


                # Retrieve predicted slots
                decoded_response ='{' + line_dict['decoded_response']
                print(gold_domain)
                print(gold_slots)

                print(json_repair.loads(decoded_response))
                print(decoded_response)

                # for each gold domain, retrieve the slots.
                # if the 
                breakpoint()
    pass


if __name__ == '__main__':
    main()