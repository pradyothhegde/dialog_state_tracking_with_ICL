import os
import json
import argparse
from tqdm import tqdm
import json_repair

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/sw/data')
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = args.data_dir
    file = os.path.join(data_dir, 'train_data.json')
    with open(file, 'r') as f:
        data = f.readlines()

    # Dictionary to store unique slots for each domain
    unique_slots = {}
    hospital_department_values = []  # List to store 'department' values for 'hospital' domain

    for conversation in tqdm(data):
        convers = json.loads(conversation.strip())
        for fname in convers.keys():
            domains = convers[fname].get('goal', {}).keys()
            for domain in domains:
                if domain not in unique_slots:
                    unique_slots[domain] = {'book': set(), 'info': set(), 'all': set()}

                domain_data = convers[fname].get('goal').get(domain)
                if domain_data:
                    # Check 'book' slots
                    if 'book' in domain_data:
                        for slot, value in domain_data.get('book').items():
                            if value:  # Add only if the value is non-empty
                                unique_slots[domain]['book'].add(slot)
                                unique_slots[domain]['all'].add(slot)

                    # Check 'info' slots
                    if 'info' in domain_data:
                        for slot, value in domain_data.get('info').items():
                            if value:  # Add only if the value is non-empty
                                unique_slots[domain]['info'].add(slot)
                                unique_slots[domain]['all'].add(slot)

                    # # Check for 'hospital' domain and 'department' slot
                    # if domain == 'hospital' and 'info' in domain_data:
                    #     department_value = domain_data['info'].get('department')
                    #     if department_value is not None:
                    #         hospital_department_values.append(department_value)

    # Print all unique slot values for each domain
    print("\nUnique slots for each domain:\n")
    for domain, slots in unique_slots.items():
        print(f"Domain: {domain}")
        print(f"  Book Slots: {sorted(slots['book'])}")
        print(f"  Info Slots: {sorted(slots['info'])}")
        print(f"  All Slots: {sorted(slots['all'])}")
        print()

    # # Print values of 'department' slot in 'hospital' domain
    # print("\nValues for 'department' slot in 'hospital' domain:\n")
    # if hospital_department_values:
    #     for value in hospital_department_values:
    #         print(f"  Department Value: {value}")
    # else:
    #     print("  No values found for 'department' slot in 'hospital' domain.")

if __name__ == '__main__':
    main()



# Unique slots for each domain:
# Domain: attraction
#   All Slots: ['area', 'name', 'type']
# Domain: hospital
#   All Slots: ['department']
# Domain: hotel
#   All Slots: ['area', 'day', 'internet', 'name', 'parking', 'people', 'pricerange', 'stars', 'stay', 'type']
# Domain: police
#   All Slots: []
# Domain: profile
#   All Slots: ['email', 'idnumber', 'name', 'phonenumber', 'platenumber']
# Domain: restaurant
#   All Slots: ['area', 'day', 'food', 'name', 'people', 'pricerange', 'time']
# Domain: taxi
#   All Slots: ['arriveBy', 'departure', 'destination', 'leaveAt']
# Domain: train
#   All Slots: ['arriveBy', 'day', 'departure', 'destination', 'leaveAt', 'people']


# dom2slots = {
#     'attraction': ['area', 'name', 'type'],
#     'hospital': ['department'],
#     'hotel': ['area', 'day', 'internet', 'name', 'parking', 'people', 'pricerange', 'stars', 'stay', 'type'],
#     'profile': ['email', 'idnumber', 'name', 'phonenumber', 'platenumber'],
#     'restaurant': ['area', 'day', 'food', 'name', 'people', 'pricerange', 'time'],
#     'taxi': ['arriveBy', 'departure', 'destination', 'leaveAt'],
#     'train': ['arriveBy', 'day', 'departure', 'destination', 'leaveAt', 'people']
# }