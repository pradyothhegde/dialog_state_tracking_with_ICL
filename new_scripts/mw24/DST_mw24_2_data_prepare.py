import json
from tqdm import tqdm
import argparse
import os

# Parse command line arguments
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/dialog_state_tracking/data/MULTIWOZ2.4')
    
    parser.add_argument('--data_file', type=str, default='data_refined.json')
    parser.add_argument('--testlistfile', type=str, default='testListFile.json')
    parser.add_argument('--validlistfile', type=str, default='valListFile.json')

    parser.add_argument('--test_write', type=str, default='mw_DST_test.json')
    parser.add_argument('--valid_write', type=str, default='mw_DST_valid.json')
    parser.add_argument('--train_write', type=str, default='mw_DST_train.json')
    args = parser.parse_args()
    return args


def main():

    count_test_files = 0
    count_valid_files = 0
    count_train_files = 0

    args = arg_parser()
    input_data_path = args.data_path
    data_file = os.path.join(input_data_path, args.data_file)
    testlistfile = os.path.join(input_data_path, args.testlistfile)
    validlistfile = os.path.join(input_data_path, args.validlistfile)
    test_write_path = os.path.join(input_data_path, args.test_write)
    valid_write_path = os.path.join(input_data_path, args.valid_write)
    train_write_path = os.path.join(input_data_path, args.train_write)

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
    with open(data_file) as in_file:
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
                                        slot_value = "not mentioned"        # making all the empty slots as "not mentioned". 
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


if __name__ == '__main__':
    main()
