
def get_instruction_for_domain(domain):
    dom2slots = {
    'taxi': ['leaveAt', 'destination', 'departure', 'arriveBy'], 
    'restaurant': ['people', 'day', 'time', 'food', 'pricerange', 'name', 'area'], 
    'attraction': ['type', 'name', 'area'], 
    'train': ['people', 'leaveAt', 'destination', 'day', 'arriveBy', 'departure'], 
    'hotel': ['stay', 'day', 'people', 'name', 'area', 'parking', 'pricerange', 'stars', 'internet', 'type']
    }
    domain_processing_string = "[" + domain + "]"
    instruction = "Identify slot names and values from the dialogue. The domain is " + domain + "." + " The slots for " + domain + " are "
    for j in range(len(dom2slots[domain])):
        instruction += dom2slots[domain][j]
        if j != len(dom2slots[domain]) - 1:
            instruction += ", "
        else:
            instruction += "."
    return instruction










# def get_instruction(process_line):
#     # find domain
#     # print("Process line: ", process_line)
#     process_line = ast.literal_eval(process_line)[0]
#     # Domain: ["taxi"] Slots: {
#     temp_text = process_line.split('\n')
#     # get the last element in temp_text
#     temp_text = temp_text[-1]
#     templ_domain = temp_text.split('Domain: ')
#     # get things within [ ]
#     domain = templ_domain[1].split('[')[1].split(']')[0]
#     # print("Domain: ", domain)
#     # print("Last line: ", temp_text)
#     domain_process_string = "[" + domain + "]"
#     domains = ast.literal_eval(domain_process_string)
#     # print(domains) # domains is a list

#     # build instruction
#     # Instruction: Identify slot names and values from the dialogue. The domains are taxi, restaurant. The slots for taxi are leaveAt, destination, departure, arriveBy. The slots for restaurant are people, day, time, food, pricerange, name, area.
#     # if there is only one domain, then the instruction will the "The domain is taxi":
#     if len(domains) == 1:
#         instruction = "Identify slot names and values from the dialogue. The domain is "
#     else:
#         instruction = "Identify slot names and values from the dialogue. The domains are "
#     for i in range(len(domains)):
#         instruction += domains[i]
#         if i != len(domains) - 1:
#             instruction += ", "
#         else:
#             instruction += "."
    
#     for i in range(len(domains)):
#         instruction += " The slots for "
#         instruction += domains[i] + " are "
#         for j in range(len(dom2slots[domains[i]])):
#             instruction += dom2slots[domains[i]][j]
#             if j != len(dom2slots[domains[i]]) - 1:
#                 instruction += ", "
#             else:
#                 instruction += "."
#     # print(instruction)

#     # breakpoint()
#     # pass
#     instruction = instruction + "\n" 
#     return instruction