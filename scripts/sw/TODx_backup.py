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

                            # write with utf-8 encoding
                            with open(tsv_file_path, 'a', encoding='utf-8') as tsv_file:
                                tsv_file.write(writing_string)
                        # Update conversation history
                        if dialog_side == 'UA':
                            conv_history += '---' + agent_utterance.strip()  
                        conv_history += '---'  # Indicate turn change
    print('TSV file created successfully!')







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
                # print(user_utterance)
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
    # print(type(all_sentences))
    # print("Total sentences: ", len(all_sentences))

    # numpy file will be saved in the same directory, name as the tsv file
    numpy_file_path = tsv_file_path.replace('.tsv', '.npy')
    
    # Removing punctuations, if required
    if punct == 'N':
        print("No punct numpy doing")
        all_sentences = remove_punctuations(all_sentences)
    elif punct == 'R':
        print("Repunct numpy doing")
        all_sentences = remove_punctuations(all_sentences)
        new_sentences = []
        import TODx_add_punctuation_clean
        for sentence in all_sentences:
            sentence = TODx_add_punctuation_clean.process_line(sentence)
            new_sentences.append(sentence)
        all_sentences = new_sentences
            
        # punctuation model.


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

def finding_test_labels(test_line):
    test_line_data = test_line.strip().split('\t')
    gold_filename = test_line_data[0]
    gold_sentence = test_line_data[1]
    gold_sentence = gold_sentence.replace('---', ' ')
    gold_domain = test_line_data[2]
    gold_slots = test_line_data[3]
    return gold_filename, gold_sentence, gold_domain, gold_slots



if __name__ =="__main__":
    pass
    # input_data_path = '/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/sw/data/test_data.json'
    # tsv_file_path = '/mnt/matylda4/hegde/int_ent/TOD_llm/experiments/sw/data/test_data.tsv'
    # dialog_side = 'U'
    # punct = 'Y'
    # json_to_tsv_format(input_data_path, tsv_file_path, dialog_side, punct)