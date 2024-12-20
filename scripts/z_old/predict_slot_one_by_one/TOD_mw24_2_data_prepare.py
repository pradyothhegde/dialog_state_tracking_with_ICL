import json
from tqdm import tqdm

# Paths to input files
input_data_path = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/Data/MULTIWOZ2.4/data_refined.json'
testlistfile = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/Data/MULTIWOZ2.4/testListFile.json'
validlistfile = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/Data/MULTIWOZ2.4/valListFile.json'

# Paths to output JSON files for test, valid, and training sets
test_write_path = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/Data/MULTIWOZ2.4/mw24_DST_test.json'
valid_write_path = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/Data/MULTIWOZ2.4/mw24_DST_valid.json'
train_write_path = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/Data/MULTIWOZ2.4/mw24_DST_train.json'

count_test_files = 0
count_valid_files = 0
count_train_files = 0

# Helper function to check if a dictionary has any non-empty values
def has_data(d):
    for v in d.values():
        # Check if the value is non-empty and not "not mentioned"
        if isinstance(v, dict):
            # If the value is a nested dictionary, apply the check recursively
            if has_data(v):
                return True
        elif isinstance(v, list):
            # Skip empty lists
            if len(v) > 0:
                return True
        elif v and v != "not mentioned" and v != "":
            return True
    return False

# Load input data (full conversations)
with open(input_data_path) as in_file:
    data = json.load(in_file)

# Load test and valid list files
with open(testlistfile) as test_list:
    test_data = test_list.readlines()

with open(validlistfile) as valid_list:
    valid_data = valid_list.readlines()

# Some strings have \n in the middle. getting rid of that.
test_data = {x.strip() for x in test_data}
valid_data = {x.strip() for x in valid_data}

# Process each conversation
for key, value in tqdm(data.items()):
    # print(key)
    log = data.get(key).get("log")
    fin_text = ""
    conversation_domain = ""
    conversation_json_list = []
    conversation_json_list.append(key)

    # Iterate through logs
    for i, log_entry in enumerate(log):

        text1 = ""
        text2 = ""
        conversation_domains = []
        conversation_slots = []

        text = log_entry.get("text")
        if i % 2 == 0:
            # User's text
            text1 = text.replace('\n', ' ').replace('\t', ' ').strip()
            user_utterance = {"User": text1}
            # append user part here.
            # print(user_utterance)
            conversation_json_list.append(user_utterance)
        elif i % 2 == 1:
            # Agent's text
            text2 = text.replace('\n', ' ').replace('\t', ' ').strip()
            agent_utterance = {"Agent": text2}

            metadata = log_entry.get("metadata")
            for domain, domain_data in metadata.items():
                if has_data(domain_data):
                    conversation_domains.append(domain)
                    # print(domain)                                   # domain name
                    # initialize dictionary for slots. 
                    domain_slots = {}
                    for meta_slot_keys, slot_dict in domain_data.items():
                        for slot, slot_value in slot_dict.items():
                            if slot in {"booked", "ticket"}:
                                # + ticket
                                # print("ignoring booking")
                                continue
                            else:
                                if slot_value == "not mentioned" or slot_value == "":
                                    slot_value = "N.A."
                                # print(slot)                         # slot name
                                # print(slot_value)                   # slot value
                            domain_slots[slot] = slot_value

                    conversation_slots.append(domain_slots)
                    
            # print(conversation_domains)
            # print(conversation_slots)
            # print(agent_utterance)
            # print("-------------------")
            conversation_json_list.append(conversation_domains)
            conversation_json_list.append(conversation_slots)
            conversation_json_list.append(agent_utterance)
            # append domain, slots, and agent part here. 
    
    if key in test_data:
        count_test_files += 1
        with open(test_write_path, 'a') as test_write_file:
            test_write_file.write(str(conversation_json_list))
            test_write_file.write('\n')
    elif key in valid_data:
        count_valid_files += 1
        with open(valid_write_path, 'a') as valid_write_file:
            valid_write_file.write(str(conversation_json_list))
            valid_write_file.write('\n')
    else:        
        count_train_files += 1
        with open(train_write_path, 'a') as train_write_file:
            train_write_file.write(str(conversation_json_list))
            train_write_file.write('\n')

print(f'Total number of test set files: {count_test_files}')
print(f'Total number of valid set files: {count_valid_files}')
print(f'Total number of training set files: {count_train_files}')