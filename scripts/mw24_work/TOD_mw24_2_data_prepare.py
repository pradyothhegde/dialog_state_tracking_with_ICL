# Pradyoth Hegde (pradyothhegde@gmail.com), Santosh Kesiraju (kesiraju@fit.vutbr.cz)

import os
import sys
import argparse
import json
from tqdm import tqdm


def refine(input_data_path, output_data_path):

    valid_domain_list = [
        "taxi",
        "police",
        "hospital",
        "hotel",
        "attraction",
        "train",
        "restaurant",
        "bus",
    ]  # also there is bus!!but not in conversation!
    valid_fname_domain_pair_dict = {}
    with open(input_data_path, "r", encoding="utf-8") as in_file:
        data = json.load(in_file)

    for key, value in tqdm(data.items()):

        goal = data.get(key).get("goal")
        log = data.get(key).get("log")
        # print(key)
        # print(value)

        ### goal operation
        valid_fname_domain_pair_dict.update(fname=key)
        temp_domain_list = []
        for goal_domain_and_others in goal.keys():
            if goal_domain_and_others in valid_domain_list:
                # print(goal_domain_and_others)
                if goal.get(goal_domain_and_others):
                    temp_domain_list.append(goal_domain_and_others)
                    valid_fname_domain_pair_dict.update(domains=temp_domain_list)

        ### log operation
        for idx, turn in enumerate(log):
            # We only look at even turns (metadata present)
            if idx % 2 != 0:
                metadata = turn.get("metadata", {})

                # Check each domain in metadata
                for domain in valid_domain_list:
                    if domain not in valid_fname_domain_pair_dict["domains"]:
                        if domain in metadata:
                            metadata[domain] = {"book": {"booked": []}, "semi": {}}

    with open(output_data_path, "w", encoding="utf-8") as out_file:
        json.dump(data, out_file, indent=4, ensure_ascii=False)


# Helper function to check if a dictionary has any non-empty values
def has_data(d):
    """Helper function to check if a dictionary has any non-empty values"""

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


def check_file_exists(fname, ovr: bool):
    """check if file exists. overwrite if ovr flag is True"""

    if os.path.exists(fname):
        print(f"FILE ALREADY EXISTS: {fname}")
        if ovr:
            print("Overwriting..")
            with open(fname, "w", encoding="utf-8") as fpw:
                fpw.write("")
        else:
            print("Pass --ovr if you would like to overwrIte.")
            sys.exit()


def main():
    """main method"""

    args = parse_arguments()

    os.makedirs(args.out_dir, exist_ok=True)

    input_data_path = os.path.join(args.in_mwoz24_dir, "data.json")
    refine_data_path = os.path.join(args.out_dir, "data_refined.json")

    check_file_exists(refine_data_path, args.ovr)
    # Step 1: Refine
    refine(input_data_path, refine_data_path)

    # Paths to input files
    testlistfile = os.path.join(args.in_mwoz24_dir, "testListFile.json")
    validlistfile = os.path.join(args.in_mwoz24_dir, "valListFile.json")

    # Paths to output JSON files for test, valid, and training sets
    test_write_path = os.path.join(args.out_dir, "mw24_DST_test.json")
    valid_write_path = os.path.join(args.out_dir, "mw24_DST_valid.json")
    train_write_path = os.path.join(args.out_dir, "mw24_DST_train.json")

    check_file_exists(train_write_path, args.ovr)
    check_file_exists(valid_write_path, args.ovr)
    check_file_exists(test_write_path, args.ovr)

    count_test_files = 0
    count_valid_files = 0
    count_train_files = 0

    # Load input data (full conversations)
    with open(refine_data_path, "r", encoding="utf-8") as in_file:
        data = json.load(in_file)

    # Load test and valid list files
    with open(testlistfile, "r", encoding="utf-8") as test_list:
        test_data = test_list.readlines()

    with open(validlistfile, "r", encoding="utf-8") as valid_list:
        valid_data = valid_list.readlines()

    # Some strings have \n in the middle. getting rid of that.
    test_data = {x.strip() for x in test_data}
    valid_data = {x.strip() for x in valid_data}

    # Process each conversation
    for key, value in tqdm(data.items()):
        # print(key)
        log = data.get(key).get("log")
        # fin_text = ""
        # conversation_domain = ""
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
                text1 = text.replace("\n", " ").replace("\t", " ").strip()
                user_utterance = {"User": text1}
                # append user part here.
                # print(user_utterance)
                conversation_json_list.append(user_utterance)
            elif i % 2 == 1:
                # Agent's text
                text2 = text.replace("\n", " ").replace("\t", " ").strip()
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
                                    if slot_value in ("not mentioned", ""):
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
            with open(test_write_path, "a", encoding="utf-8") as test_write_file:
                test_write_file.write(str(conversation_json_list))
                test_write_file.write("\n")
        elif key in valid_data:
            count_valid_files += 1
            with open(valid_write_path, "a", encoding="utf-8") as valid_write_file:
                valid_write_file.write(str(conversation_json_list))
                valid_write_file.write("\n")
        else:
            count_train_files += 1
            with open(train_write_path, "a", encoding="utf-8") as train_write_file:
                train_write_file.write(str(conversation_json_list))
                train_write_file.write("\n")

    print(f"Total number of test set dialogues: {count_test_files}")
    print(f"Total number of valid set dialogues: {count_valid_files}")
    print(f"Total number of training set dialogues: {count_train_files}")


def parse_arguments():
    """parse command line args"""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_mwoz24_dir", help="path to original MULTIWOZ2.4 dir containing data.json"
    )
    parser.add_argument("out_dir", help="path to outdir to save processed data")
    parser.add_argument(
        "--ovr", action="store_true", help="overwrite if out files exist."
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
