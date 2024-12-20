import ast
import os
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import re

def remove_punctuations(sentences):
    # print('Removing punctuations from the sentences...')
    table = str.maketrans('', '', string.punctuation)
    return [s.translate(table).lower() for s in sentences]


# Function to process SpokenWOZ data and write to TSV
def json_to_tsv_format(input_data_path, tsv_file_path, dialog_side):
    def has_valid_slots(domain_data):
        """Check if a domain has any non-empty slots."""
        for slot_type in ["book", "semi"]:
            for slot_value in domain_data.get(slot_type, {}).values():
                if slot_value and slot_value != "not mentioned":
                    return True
        return False

    with open(input_data_path, 'r') as input_file:
        for line in input_file:
            # Load JSON line data
            line_data = json.loads(line.strip())

            for fname in line_data:
                filename = fname
                log = line_data[fname]['log']
                conv_history = ''  # Conversation history
                writing_string = ''
                
                for i, log_entry in enumerate(log):
                    text = log_entry['text'].strip()
                    # print("---",text.strip(), "---")
                    # breakpoint()
                    if log_entry['tag'] == 'user':
                        # User turn
                        user_utterance = text
                        conv_history += user_utterance.strip()  # Add to conversation history
                    elif log_entry['tag'] == 'system':
                        # Agent turn
                        agent_utterance = text
                        if conv_history.strip():
                            # Write conversation data
                            domains, slots = '', ''
                            if log_entry.get('metadata'):
                                domains = []
                                slots_list = []
                                for domain, domain_data in log_entry['metadata'].items():
                                    if domain == "police":
                                        continue  # Skip "police" domain
                                    if not has_valid_slots(domain_data):
                                        continue  # Skip domain if all slots are empty

                                    domains.append(domain)
                                    domain_slots = {}
                                    for slot_type in ["book", "semi"]:
                                        for slot, slot_value in domain_data.get(slot_type, {}).items():
                                            if slot == "booked":
                                                continue
                                            if not slot_value or slot_value == "not mentioned":
                                                slot_value = "N.A."
                                            domain_slots[slot] = slot_value
                                    slots_list.append(domain_slots)
                                slots = str(slots_list)
                                domains = str(domains)
                            
                            writing_string = (
                                f"{filename}\t{conv_history.strip()}\t{domains}\t{slots}\n"
                            )

                            with open(tsv_file_path, 'a') as tsv_file:
                                tsv_file.write(writing_string)

                        # Update conversation history
                        if dialog_side == 'UA':
                            conv_history += '---' + agent_utterance.strip()  
                        conv_history += '---'  # Indicate turn change
    print('TSV file created successfully!')




if __name__ =="__main__":
    pass
    # input_data_path = '/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/sw/data/test_data.json'
    # tsv_file_path = '/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/sw/data/test_data.tsv'
    # dialog_side = 'U'
    # punct = 'Y'
    # json_to_tsv_format(input_data_path, tsv_file_path, dialog_side)