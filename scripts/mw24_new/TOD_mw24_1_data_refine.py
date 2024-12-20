# Pradyoth Hegde (pradyothhegde@gmail.com), Santosh Kesiraju (kesiraju@fit.vutbr.cz)
# Reconciling the multiwoz2.4 data.json with the goal and log domains.

import json
from tqdm import tqdm

input_data_path = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/Data/MULTIWOZ2.4/data.json'
output_data_path = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/Data/MULTIWOZ2.4/data_refined.json'

valid_domain_list = ["taxi", "police", "hospital", "hotel", "attraction", "train", "restaurant", "bus"] # also there is bus!!but not in conversation!
valid_fname_domain_pair_dict = {}
with open(input_data_path) as in_file:
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
            if (goal.get(goal_domain_and_others)):
                temp_domain_list.append(goal_domain_and_others)
                valid_fname_domain_pair_dict.update(domains=temp_domain_list)

    ### log operation
    for idx, turn in enumerate(log):
        # We only look at even turns (metadata present)
        if idx % 2 != 0:
            metadata = turn.get("metadata", {})
            
            # Check each domain in metadata
            for domain in valid_domain_list:
                if domain not in valid_fname_domain_pair_dict['domains']:
                    if domain in metadata:
                        metadata[domain] = {
                            "book": {"booked": []}, 
                            "semi": {}
                        }

with open(output_data_path, 'w') as out_file:
    json.dump(data, out_file, indent=4)
