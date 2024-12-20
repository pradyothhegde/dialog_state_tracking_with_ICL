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
    # Remove punctuations from the sentence and convert to lowercase
# test case for remove_punctuations()
# print(remove_punctuations(['Hello, how are you?', 'I am fine.', 'What about you?']))


def prepare_data_for_sentence_embedding(tsv_file_path, option):
    # Load the [train / test] tsv file to extract [user / user-agent] sentences.
    # open test tsv file.
    all_sentences = []
    # print('Preparing data for sentence embedding...')

    if option == 'U':
    # Considering only user sentences
        with open(tsv_file_path, 'r') as tsv_file:
            for i, line in enumerate(tsv_file):
                line_data = line.strip().split('\t')
                gold_filename = line_data[0]

                user_utterance = line_data[1].strip()
                user_utterance = user_utterance.replace('---', ' ')     # remove --- from the user utterance
                # domain = line_data[2]
                # slots = line_data[3]
                # agent_utterance = line_data[4]
                # selecting user only sentences and appending them to a list
                # conv_utterance = conv_utterance + ' ' + user_utterance
                all_sentences.append(user_utterance)
                # print(conv_utterance)
                # print(len(all_sentences))
                # if i == 1000:                                           # for testing, remove later   
                #     break
        return all_sentences
    
    if option == 'UA':
    # Considering user-agent sentences
        with open(tsv_file_path, 'r') as tsv_file:
            for line in tsv_file:
                line_data = line.strip().split('\t')
                gold_filename = line_data[0]

                user_agent_utterance = line_data[1].strip()
                user_agent_utterance = user_agent_utterance.replace('---', ' ')     # remove --- from the user-agent utterance
                # domain = line_data[2]
                # slots = line_data[3]
                # agent_utterance = line_data[4].strip()
                all_sentences.append(user_agent_utterance)
        return all_sentences

def calculate_save_sentence_embedding(tsv_file_path, dialog_side, punct):    # file, dialog_side, punct
    # Calculate sentence embedding and save in npy file
    # train_tsv_file = os.path.join(data_path, 'mw24_DST_train_turns.tsv')
    # test_tsv_file = os.path.join(data_path, 'mw24_DST_test_turns.tsv')
    # print('Calculating sentence embeddings...')


    all_sentences = prepare_data_for_sentence_embedding(tsv_file_path, dialog_side)    # pick option from args
    # test_sentences = prepare_data_for_sentence_embedding(test_tsv_file, sentence_embedding_option)     # pick option from args

    # numpy file will be saved in the same directory, name as the tsv file
    numpy_file_path = tsv_file_path.replace('.tsv', '.npy')
    
    # Removing punctuations, if required
    if punct == 'no_punct':
        print("No punct numpy doing")
        all_sentences = remove_punctuations(all_sentences)
    
    import torch
    from sentence_transformers import SentenceTransformer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sentence_model = SentenceTransformer('sentence-transformers/LaBSE')
    # sentence_model = SentenceTransformer('sergioburdisso/dialog2flow-single-bert-base')
    sentence_model = sentence_model.to(device)  # Move model to GPU if available

    # Calculate sentence embedding for train set
    embeddings = sentence_model.encode(all_sentences, convert_to_tensor=True, device=device)
    embeddings = embeddings.cpu().detach().numpy()
    np.save(numpy_file_path, embeddings)
    del embeddings
    del sentence_model
    torch.cuda.empty_cache()

    print('Sentence embeddings saved successfully!')
    return numpy_file_path


# Put Domain, Slots of train and test data in a TSV file. 
def json_to_tsv_format(json_file_path, tsv_file_path, dialog_side):
    with open(json_file_path, 'r') as train_file:
        for i, line in enumerate(train_file):
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
                # print(turn)
                if identifier % 4 == 0:
                    # user turn
                    user_utterance = turn.get('User')
                    # print(user_utterance)
                    # writing_string = writing_string + user_utterance + '\t'     # user utterance add
                    conv_history = conv_history + user_utterance         # --- is to indicate turn change
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
                    # agenr turn
                    agent_utterance = turn.get('Agent')
                    # print(agent_utterance)
                    writing_string = filename + '\t' + conv_history.strip() + '\t' + domain_write + slots_write + '\n'
                
                    # print(writing_string)
                    with open(tsv_file_path, 'a') as tsv_file:
                        tsv_file.write(writing_string)
                    # breakpoint()
                    writing_string = ''         

                    # Add agent utterance to conversation history if dialog_side is UA
                    if dialog_side == 'UA': 
                        conv_history = conv_history + ' ' + '---' + agent_utterance 
                    
                    conv_history = conv_history + '---'          # --- is to indicate turn change
                    conv_history.replace('  ', ' ')     # remove double spaces, if any.
    print('TSV files created successfully!')



def finding_test_labels(test_line):
    test_line_data = test_line.strip().split('\t')
    gold_filename = test_line_data[0]
    gold_sentence = test_line_data[1]
    gold_sentence = gold_sentence.replace('---', ' ')
    gold_domain = test_line_data[2]
    gold_slots = test_line_data[3]
    return gold_filename, gold_sentence, gold_domain, gold_slots


def get_instruction_string(instruction_template, predict):
    if instruction_template == 0:
        return ''
    elif instruction_template == 1 and predict == 'domain':
        instruction_string = 'Instruction: Identify the domain(s).\n'
        return instruction_string
    elif instruction_template == 1 and predict == 'domain_slots':
        instruction_string = 'Instruction: Identify the slots.\n'
        return instruction_string
    elif instruction_template == 2 and predict == 'domain':
        instruction_string = 'Instruction: Your job is to detect domain(s) in a multi-turn conversation. There are similar examples given in particular format, please identify the domain(s) for the given dialog and close the format.\n'
        return instruction_string
    elif instruction_template == 2 and predict == 'domain_slots':
        instruction_string = 'Instruction: Your job is to detect domain(s) and slot(s) in multi-turn dialog. There are similar examples given in particular format, please identify the domain(s) and slot(s) for the last dialog and close the format.\n'
        return instruction_string
    
def get_in_simple_user_agent_format(sentence, dialog_side, punct):
    fin_conv = ''
    
    if dialog_side == 'U':
        if punct == 'no_punct':
            sentence = remove_punctuations([sentence])[0]
            fin_conv = 'User: ' + sentence.strip().replace('---', ' ')
        elif punct == 'punct':
            fin_conv = 'User: ' + sentence.strip().replace('---', ' ') 
        return fin_conv
    
    elif dialog_side == 'UA':
        utterances = sentence.split('---')
        for i, utterance in enumerate(utterances):
            if i % 2 == 0:
                if punct == 'no_punct':
                    utterance = remove_punctuations([utterance])[0]
                fin_conv = 'User: ' + utterance
            else:
                if punct == 'no_punct':
                    utterance = remove_punctuations([utterance])[0]
                fin_conv = fin_conv + ' Agent: ' + utterance
        return fin_conv

def naming_function(model_name, punct, predict):
    return_model_name = ''
    return_punct = ''
    return_predict = ''

    if model_name == 'allenai/OLMo-7B-Instruct':
        return_model_name = 'O7BI'
    
    if punct == 'punct':
        return_punct = 'P'
    elif punct == 'no_punct':
        return_punct = 'NP'

    if predict == 'domain':
        return_predict = 'D'
    elif predict == 'domain_slots':
        return_predict = 'DS'
    
    return return_model_name, return_punct, return_predict


def extract_string_after_punct(nested_list):
    result = ''
    def traverse_list(lst):
        nonlocal result
        for item in lst:
            if isinstance(item, list):
                traverse_list(item)
            elif isinstance(item, str):
                result += item
    traverse_list(nested_list)
    return result

def clean_text_after_punct(text: str) -> str:
        cleaned_text = re.sub(r'<[^>]*>', '', text)
        cleaned_text = re.sub(r'\[.*?\]', '', cleaned_text)
        
        # cleaned_text = cleaned_text.replace(".Speaker:", ". Speaker")
        # cleaned_text = cleaned_text.replace("?Speaker:", "? Speaker")

        # Replace multiple spaces and incorrect punctuations
        cleaned_text = cleaned_text.replace("  ", " ")
        cleaned_text = cleaned_text.replace("?.", "?")
        cleaned_text = cleaned_text.replace(",.", ".")
        cleaned_text = cleaned_text.replace(",?", "?")
        cleaned_text = cleaned_text.replace(". .", ".")
        cleaned_text = cleaned_text.replace("? ?", "?")
        cleaned_text = cleaned_text.replace(". ,", ".")

        # Correct punctuation and contractions
        cleaned_text = re.sub(r'\b(?:im|i m)\b', "I'm", cleaned_text, flags=re.IGNORECASE)
        cleaned_text = re.sub(r'\b(?:P\.?\s?M\.?|P\s?M|P\.?\s?m|p\.?\s?m)\b', 'pm', cleaned_text, flags=re.IGNORECASE)
        cleaned_text = cleaned_text.replace("Youre", "You're")
        cleaned_text = cleaned_text.replace("youre", "you're")
        cleaned_text = cleaned_text.replace("You. re", "You're")
        cleaned_text = cleaned_text.replace("you. re", "you're")
        cleaned_text = re.sub(r'([.?,])\1+', r'\1', cleaned_text)

        cleaned_text = cleaned_text.replace(" ?", "?")
        cleaned_text = cleaned_text.replace(" .", ".")
        cleaned_text = cleaned_text.replace(" ,", ",")

        cleaned_text = cleaned_text.replace("?", "? ")
        cleaned_text = cleaned_text.replace(".", ". ")
        cleaned_text = cleaned_text.replace(",", ", ")
        cleaned_text = cleaned_text.replace("!", "! ")

        cleaned_text = cleaned_text.replace("\t", " ")
        cleaned_text = cleaned_text.replace("  ", " ")
        cleaned_text = cleaned_text.replace("  ", " ")

        # Handle punctuation sequences
        cleaned_text = re.sub(r'\.\s*,\s*,', '. ', cleaned_text)  # Replace ". , ," with ". "
        cleaned_text = re.sub(r'\.\s*,', '. ', cleaned_text)      # Replace ". ," with ". "
        cleaned_text = re.sub(r',\s*\.', '. ', cleaned_text)      # Replace ", ." with ". "
        cleaned_text = re.sub(r'\.\s*\?', '? ', cleaned_text)      # Replace ". ?" with "? "
        cleaned_text = re.sub(r',\s*\?', '? ', cleaned_text)      # Replace ", ?" with "? "

        # Replace multiple consecutive occurrences of the same punctuation mark with a single occurrence
        cleaned_text = re.sub(r'([.?,])\1+', r'\1', cleaned_text)
        cleaned_text = re.sub(r'([.?,])\s*([.?,])', r'\1 \2', cleaned_text)  # Ensures proper spacing
        cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)  # Replace multiple spaces with a single space

        return cleaned_text.strip()

def json_to_tsv_format_asrTrans_mix(json_file_path, tsv_file_path, dialog_side, human_trans_sorted_file, punct):
    if punct == 'Y':
        from typing import List
        from punctuators.models import PunctCapSegModelONNX
        m = PunctCapSegModelONNX.from_pretrained("pcs_en")
    human_index = 0
    # with open(human_trans_sorted_file, 'r') as human:
    #     human.readlines()
    # human = open(human_trans_sorted_file, 'r')
    # read human transcription file
    with open(human_trans_sorted_file, 'r') as f:
        human = f.readlines()


    with open(json_file_path, 'r') as train_file:
        for i, line in enumerate(train_file):
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
                # print(turn)
                if identifier % 4 == 0:
                    # user turn
                    # user_utterance = turn.get('User')
                    # take human transcription of human_index
                    # try expect if human_index exceeds the length of the human transcription file
                    try:
                        user_utterance = human[human_index].strip()
                    except:
                        print('Human index exceeds the length of the human transcription file')
                        break
                    # user_utterance = human[human_index].strip()
                    user_utterance = ' '.join(user_utterance.split(' ')[1:]).strip()
                    user_utterance = user_utterance.strip()
                    user_utterance = user_utterance.replace('  ', ' ')   # remove double spaces, if any.
                    user_utterance = user_utterance.replace('  ', ' ')
                    human_index += 1
                    # print(user_utterance)
                    # print('-------------------')
                    # breakpoint()
                    if punct == 'Y':
                        user_utterance = user_utterance.strip() # strip
                        input_text = [user_utterance] 
                        out_sentence = m.infer(input_text)
                        # print(out_sentence)
                        user_utterance = extract_string_after_punct(out_sentence)
                        user_utterance = clean_text_after_punct(user_utterance)

                    # print(user_utterance)
                    # print('====================')
                    # breakpoint()
                    # print(user_utterance)
                    # writing_string = writing_string + user_utterance + '\t'     # user utterance add
                    conv_history = conv_history + user_utterance         # --- is to indicate turn change
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
                    # agenr turn
                    agent_utterance = turn.get('Agent')
                    agent_utterance = agent_utterance.replace('  ', ' ') 
                    agent_utterance = agent_utterance.replace('  ', ' ') 
                    # print(agent_utterance)
                    writing_string = filename + '\t' + conv_history.strip() + '\t' + domain_write + slots_write + '\n'
                    writing_string.replace('  ', ' ')     # remove double spaces, if any.

                
                    # print(writing_string)
                    with open(tsv_file_path, 'a') as tsv_file:
                        tsv_file.write(writing_string)
                    # breakpoint()
                    writing_string = ''         

                    # Add agent utterance to conversation history if dialog_side is UA
                    if dialog_side == 'UA': 
                        if punct == 'Y':
                            agent_utterance = agent_utterance.strip()
                            input_text = [agent_utterance]
                            out_sentence = m.infer(input_text)
                            # print(out_sentence)
                            agent_utterance = extract_string_after_punct(out_sentence)
                            agent_utterance = clean_text_after_punct(agent_utterance)
                        conv_history = conv_history + '---' + agent_utterance 
                    
                    conv_history = conv_history + '---'          # --- is to indicate turn change
                    conv_history.replace('  ', ' ')     # remove double spaces, if any.
                    conv_history.replace('  ', ' ')     # remove double spaces, if any.
    print('TSV files created successfully!')






if __name__ == '__main__':
    pass
#     # sample
    # file = '/mnt/matylda4/hegde/int_ent/TOD_llm/Data/Multiwoz2.1/text_half_human_sorted'
    # # json_file_path, tsv_file_path, dialog_side, human_trans_sorted_file, punct
    # json_file_path = '/mnt/matylda4/hegde/int_ent/TOD_llm/Data/Multiwoz2.1/mw21_DST_test.json'
    # tsv_file_path = '/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/mw21/NecessaryFiles_text/mw21_DST_test_U_punct_turns_zzzzzz.tsv'
    # dialog_side = 'U'
    # human_trans_sorted_file = '/mnt/matylda4/hegde/int_ent/TOD_llm/Data/Multiwoz2.1/text_half_human_sorted'
    # punct = 'N'
    # json_to_tsv_format_asrTrans_mix(json_file_path, tsv_file_path, dialog_side, human_trans_sorted_file, punct)
    # print('Done')
