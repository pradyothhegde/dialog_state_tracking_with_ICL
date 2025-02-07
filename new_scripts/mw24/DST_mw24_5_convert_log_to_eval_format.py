import argparse
import os
import json
import json
from json_repair import json_repair
from etc.time_string_normalizer import standardize_time
import re

def arg_parser():
    parser = argparse.ArgumentParser(description='Convert log to eval format')
    parser.add_argument('--log_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/experiments/mw24/mistrals/M7BI03_MW24_MP_ST_PH-nm_SO_Labse_NN-10_UA_SV_emb-U', help='Log file path')
    parser.add_argument('--output_folder', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/results/mw24/mistrals', help='Output file path')
    # Below options need to be debugged; Dont use them.
    parser.add_argument('--folder_operation', type=bool, default=False, help='If True, then iterate over all files in the folder')
    parser.add_argument('--main_folder', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/experiments/mw24/PLH', help='Main folder path')
    return parser

time_related_slots = ['leaveAt', 'arriveBy', 'time']

dom2slots = {
    'taxi': ['leaveAt', 'destination', 'departure', 'arriveBy'], 
    'restaurant': ['people', 'day', 'time', 'food', 'pricerange', 'name', 'area'], 
    'attraction': ['type', 'name', 'area'], 
    'train': ['people', 'leaveAt', 'destination', 'day', 'arriveBy', 'departure'], 
    'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type']
}

empty_placeholders = ["not mentioned", "N.A.", "none", ""]

def convert_to_string(value):
  ''' This function converts the given value to a string '''
  if isinstance(value, (dict, list)):
    return json.dumps(value)
  return str(value)

# function to remove punctuation from the string, except ":"
def punct_rem(text):
    return re.sub(r"[^\w\s:]", "", text)


def main():
    parser = arg_parser()
    args = parser.parse_args()

    output_data = {}

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    log_directory = os.path.join(args.log_file, 'log_files')
    log_files = os.listdir(log_directory)

    for file in log_files:
        file_path = os.path.join(log_directory, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line_string in lines:
                line_json = json.loads(line_string)
                processed_output = json_repair.loads(line_json['processed_output'])

                current_state = {}
                active_domains = []
                filename = line_json['gold_fname']
                filename = filename.split('.')[0].lower()
                
                if filename not in output_data:
                    output_data[filename] = []
                
                if output_data[filename]:
                    cumulative_state = output_data[filename][-1].get('state',{})
                else:
                    cumulative_state = {}

                current_state_slots = {}
                # Update cumulative state and create current_state
                for domain, slot_values in processed_output.items():
                    if domain not in cumulative_state:
                         cumulative_state[domain] = {}
                    
                    if domain not in current_state_slots:
                        current_state_slots[domain] = {}
                        
                    for slot, value in slot_values.items():
                        if value in empty_placeholders:
                            if slot.lower() in cumulative_state[domain]:
                                del cumulative_state[domain][slot.lower()]
                        elif slot.lower() in dom2slots.get(domain,[]): # only add valid slots
                            cumulative_state[domain][slot.lower()] = convert_to_string(value)
                            current_state_slots[domain][slot.lower()] = convert_to_string(value)

                    if current_state_slots[domain] == {}:
                        del current_state_slots[domain]

                # Filter current state
                for domain, slots in current_state_slots.items():
                    for slot in slots:
                        slots[slot] = punct_rem(slots[slot]).strip()        # removing punctuation from the slot values if any , except ":".
                        if slot in time_related_slots:
                            slots[slot] = standardize_time(slots[slot])
                            try: 
                                if slots[slot][0] == '0':
                                    slots[slot] = slots[slot][1:]
                            except:
                                pass

                    current_state[domain] = slots

                # Remove null values from current_state
                for domain in list(current_state.keys()): # Use list() to avoid dict size change during iteration
                    for slot, value in list(current_state[domain].items()): # Use list() to allow deletion during iteration
                       if value is None or value == "null":
                           del current_state[domain][slot]
                    if not current_state[domain]:
                        del current_state[domain] # Remove the domain if it's empty
                
                active_domains = list(current_state.keys())
                
                output_data[filename].append({
                    "response": "",
                    "state": current_state,
                    "active_domains": active_domains
                })

    output_file_name = os.path.basename(args.log_file)
    output_file_name = os.path.join(args.output_folder, f"{output_file_name}.json")
    print(f"Output file name: {output_file_name}")
    
    # Write to output JSON file
    with open(output_file_name, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"Output written to: {output_file_name}")

if __name__ == "__main__":
    parser = arg_parser()
    args = parser.parse_args()
    if args.folder_operation:
        main_folder = args.main_folder
        folders = os.listdir(main_folder)
        for folder in folders:
            args.log_file = os.path.join(main_folder, folder)
            print(f"Processing folder: {args.log_file}")
            main()
    else:
        main()