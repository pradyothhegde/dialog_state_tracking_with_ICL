import numpy as np

dom2slots = {
    'taxi': ['leaveAt', 'destination', 'departure', 'arriveBy'], 
    'restaurant': ['people', 'day', 'time', 'food', 'pricerange', 'name', 'area'], 
    'attraction': ['type', 'name', 'area'], 
    'train': ['people', 'leaveAt', 'destination', 'day', 'arriveBy', 'departure'], 
    'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type']
}

def get_sorted_slots(domain, input_file_name):
    # Finish the function
    if '_SO_' in input_file_name:
        sorted_slots = sorted(dom2slots[domain])
    elif '_SU_' in input_file_name:
        sorted_slots = dom2slots[domain]
    # if 'SU1' or 'SU2' in input_file_name, shuffle with the seed number. The number is present after 'SU' till next '_'
    elif '_SU' in input_file_name:
        seed_number = int(input_file_name.split('SU')[1].split('_')[0])
        np.random.seed(seed_number)
        sorted_slots = np.random.permutation(dom2slots[domain])
    return sorted_slots

# Test
if __name__ == '__main__':
    domain = 'restaurant'
    input_file_name = 'MW24_OP_NST_PH-nm_SU2_Labse_NN-10_UA_SKV'
    print(get_sorted_slots(domain, input_file_name))