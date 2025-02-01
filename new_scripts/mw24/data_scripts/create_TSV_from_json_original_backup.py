import ast
import tqdm
from . import new_punctuate
import re

def replace_unk_to_dash(text):      # replace <unk> to '-'
    return re.sub(r'<[^>]*>', '-', text)

# for punctuation, first remove the punctuation from the text, then add the punctuation, and clean the text. 

def remove_punc(s):                                                                                                                               
    puncs = '!"#$%&()*+,./;<=>?@[\\]^_`{|}~' 
    return s.translate(str.maketrans('', '', puncs))

# Put Domain, Slots of train and test data in a TSV file. 
def create_TSV_from_json(json_file_path, tsv_file_path, args):
    print('Creating TSV files...')
    # punct_processor = punctuate.TextProcessor('pcs_en')

    dialog_side = args.dialog_history
    punctuation = args.punct
    
    with open(json_file_path, 'r') as train_file:
        for i, line in tqdm.tqdm(enumerate(train_file)) :
            # if i < 620:                                           # for testing, remove later   
            #     continue
            line_data = ast.literal_eval(line.strip())
            filename = line_data[0]
            dialogue = line_data[1:]
            # print(gold_filename)
            # print(gold_dialogue)
            writing_string = ''

            conv_history = ''                     # Modify whatever is required
            domain_write = ''
            slots_write = ''
            for identifier, turn in enumerate(dialogue):
                if identifier % 4 == 0:
                    # user turn
                    # If the option in O, keep the original text as it is.
                    # Getting original text as it is from the input file. 
                    user_utterance = turn.get('User')
                    user_utterance = user_utterance.replace('  ', ' ') # remove double spaces, if any.
                    user_utterance = user_utterance.replace('  ', ' ')
                    # print(user_utterance)
                    # writing_string = writing_string + user_utterance + '\t'     # user utterance add

                    # punctuation to user utterance operation
                    if punctuation == 'N':      # No punctuation
                        user_utterance = remove_punc(user_utterance)
                        user_utterance = user_utterance.strip()
                        # pass
                    elif punctuation == 'M':    # Model punctuation
                        user_utterance = user_utterance
                        # print(f"Original: {user_utterance}")
                        user_utterance = remove_punc(user_utterance)
                        # print(f"Removed Punc: {user_utterance}")
                        # print("------------------------------------")
                        user_utterance = new_punctuate.punctuate_and_segment_texts([user_utterance])
                        # the user_utterance will be a list of list.
                        # print("------------------------------------")
                        user_utterance = ' '.join(user_utterance[0])
                        user_utterance = replace_unk_to_dash(user_utterance)    # dash handle
                        # print(f"Added Punc: {user_utterance}")
                        # print("====================================")
                        # breakpoint()
                        # user_utterance = punct_processor.add_punctuation(user_utterance)
                        # user_utterance = punct_processor.clean_text(user_utterance)
                        # user_utterance = user_utterance.strip()
                        # pass

                    conv_history = conv_history + user_utterance.strip()         # --- is to indicate turn change
                    # breakpoint()
                    pass 
                elif identifier % 4 == 1:
                    # domain
                    domain_write = str(turn) + '\t'          # domain add
                    # for domain in turn:
                    #     print(domain)
                    # breakpoint()
                    pass
                elif identifier % 4 == 2:
                    # slots
                    slots_write = str(turn) + '\t'          # slots add
                    # for slot in turn:
                    #     print(str(slot))
                        # go further into the slot
                        # for slot_key, slot_value in slot.items():
                    # breakpoint()
                    pass
                elif identifier % 4 == 3:  
                    # agent turn
                    agent_utterance = turn.get('Agent')
                    agent_utterance = agent_utterance.replace('  ', ' ')    # remove double spaces, if any.
                    agent_utterance = agent_utterance.replace('  ', ' ')

                    # punctuation to agent utterance operation
                    if punctuation == 'N':    # No punctuation
                        # agent_utterance = punct_processor.remove_punctuation(agent_utterance)
                        agent_utterance = remove_punc(agent_utterance)
                        agent_utterance = agent_utterance.strip()
                        # pass
                    elif punctuation == 'M':    # Model punctuation
                        # agent_utterance = punct_processor.remove_punctuation(agent_utterance)
                        # agent_utterance = punct_processor.add_punctuation(agent_utterance)
                        # agent_utterance = punct_processor.clean_text(agent_utterance)
                        # print(f"Original: {agent_utterance}")
                        agent_utterance = remove_punc(agent_utterance)
                        # print("------------------------------------")
                        # print(f"Removed Punc: {agent_utterance}")
                        agent_utterance = new_punctuate.punctuate_and_segment_texts([agent_utterance])
                        # the agent_utterance will be a list of list.       
                        agent_utterance = ' '.join(agent_utterance[0])
                        agent_utterance = replace_unk_to_dash(agent_utterance)  # dash unk handle.
                        # print("------------------------------------")
                        # print(agent_utterance)
                        # print("====================================")
                        # breakpoint()
                        # pass
                    # print(agent_utterance)
                    writing_string = filename + '\t' + conv_history.strip() + '\t' + domain_write + slots_write + '\n'
                
                    # print(writing_string)
                    with open(tsv_file_path, 'a') as tsv_file:
                        tsv_file.write(writing_string)
                    # breakpoint()
                    writing_string = ''         

                    # Add agent utterance to conversation history if dialog_side is UA
                    if dialog_side == 'UA': 
                        conv_history = conv_history + '---' + agent_utterance.strip() 
                    
                    conv_history = conv_history + '---'          # --- is to indicate turn change
                    conv_history.replace('  ', ' ')     # remove double spaces, if any.

    print('TSV files created successfully!')