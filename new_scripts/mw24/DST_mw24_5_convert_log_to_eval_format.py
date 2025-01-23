import argparse
import os
import json

def arg_parser():
    parser = argparse.ArgumentParser(description='Convert log to eval format')
    parser.add_argument('--log_file', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/experiments/mw24/MW24_OP_NST_PH-nm_SU_Labse_NN-10_UA_SKV', help='Log file path')
    parser.add_argument('--output_folder', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/results/mw24/baseline', help='Output file path')
    return parser

dom2slots = {
    'taxi': ['leaveat', 'destination', 'departure', 'arriveby'],
    'restaurant': ['people', 'day', 'time', 'food', 'pricerange', 'name', 'area'],
    'attraction': ['type', 'name', 'area'],
    'train': ['people', 'leaveat', 'destination', 'day', 'arriveby', 'departure'],
    'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type']
}

empty_placeholders = ["not mentioned", "N.A.", "none", "", "none"]

def convert_to_string(value):
  ''' This function converts the given value to a string '''
  if isinstance(value, (dict, list)):
    return json.dumps(value)
  return str(value)


def main():
    parser = arg_parser()
    args = parser.parse_args()

    output_data = {}

    log_directory = os.path.join(args.log_file, 'log_files')
    log_files = os.listdir(log_directory)

    for file in log_files:
        file_path = os.path.join(log_directory, file)
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line_string in lines:
                line_json = json.loads(line_string)
                processed_output = json.loads(line_json['processed_output'])

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
                        if value in empty_placeholders:                 # remove empty placeholders
                            if slot.lower() in cumulative_state[domain]:
                                del cumulative_state[domain][slot.lower()]
                                
                        elif slot.lower() in dom2slots.get(domain,[]): # only add valid slots
                            cumulative_state[domain][slot.lower()] = convert_to_string(value)
                            current_state_slots[domain][slot.lower()] = convert_to_string(value)

                    if current_state_slots[domain] == {}:
                        del current_state_slots[domain]

                # Filter current state
                for domain, slots in current_state_slots.items():
                   current_state[domain] = slots

                active_domains = list(current_state.keys())

                
                output_data[filename].append({
                    "response": "",
                    "state": current_state,
                    "active_domains": active_domains
                })

    output_file_name = os.path.basename(args.log_file)
    output_file_name = os.path.join(args.output_folder, f"{output_file_name}.json")
    print(f"Output file name: {output_file_name}")
    # breakpoint()
    # Write to output JSON file
    with open(output_file_name, 'w') as outfile:
        json.dump(output_data, outfile, indent=4)

    print(f"Output written to: {output_file_name}")

if __name__ == "__main__":
    main()