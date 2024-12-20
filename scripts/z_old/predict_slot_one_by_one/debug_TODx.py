import ast
import os
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json


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
            # if i == 500:                                           # for testing, remove later   
            #     break
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

def find_x_similar_sentences(train_embeddings, test_embeddings, number_of_similar_sentences, train_tsv_file_path):
    # print('inside find_x_similar_sentences')
    # breakpoint()
    picked_train_file_line = []

    # Load the training sentences
    with open(train_tsv_file_path, 'r', encoding='utf-8') as f:
        train_sentences = [line.strip() for line in f]

        # train_sentences = [line.strip() for line in f]

    # Ensure the test_embeddings are of shape (num_test_samples, embedding_size)
    if len(test_embeddings.shape) == 1:
        test_embeddings = test_embeddings.reshape(1, -1)
    
    # Calculate cosine similarity between this test sentence and all training sentences
    cosine_sim = cosine_similarity(test_embeddings, train_embeddings)

    top_20_sim_indices = np.argsort(cosine_sim[0])[::-1][:number_of_similar_sentences]
    for idx in top_20_sim_indices:
        picked_train_file_line.append(train_sentences[idx])
    
    return picked_train_file_line



# Template 1, return in simple text format
def get_text_dom_in_template1(similar_sentences_lines, dialog_side, punct):
    formatted_context = ''
    if dialog_side == 'U':
        for line in reversed(similar_sentences_lines):  # reversed to get the most similar sentence in the last. / near to the testing sentence
            line_data = line.strip().split('\t')
            user_utterance = line_data[1].replace("---", " ")
            if punct == 'no_punct':
                user_utterance = remove_punctuations([user_utterance])[0]
            formatted_context = formatted_context + "User: " + user_utterance + ' Domain: '+ line_data[2] + '\n'

        return formatted_context
    elif dialog_side == 'UA':
        for line in reversed(similar_sentences_lines): # reversed to get the most similar sentence in the last. / near to the testing sentence
            line_data = line.strip().split('\t')
            utterances = line_data[1].strip()
            if '---' in utterances:
                utterances = utterances.split('---')
            else:
                utterances = [utterances]
            for i, utterance in enumerate(utterances):
                if i % 2 == 0:
                    user_utterance = utterance
                    if punct == 'no_punct':
                        user_utterance = remove_punctuations([user_utterance])[0]
                    formatted_context = formatted_context + "User: " + user_utterance.strip() 
                else:
                    agent_utterance = utterance
                    if punct == 'no_punct':
                        agent_utterance = remove_punctuations([agent_utterance])[0]
                    formatted_context = formatted_context.strip() + " Agent: " + agent_utterance.strip() + ' '
            formatted_context = formatted_context.strip() + ' Domain: '+ line_data[2] + '\n'
        return formatted_context

def get_text_dom_slots_in_template1(similar_sentences_lines, dialog_side, punct):
    formatted_context = ''
    if dialog_side == 'U':
        for line in reversed(similar_sentences_lines): # reversed to get the most similar sentence in the last. / near to the testing sentence
            line_data = line.strip().split('\t')
            user_utterance = line_data[1].replace("---", " ")   # 
            if punct == 'no_punct':
                user_utterance = remove_punctuations([user_utterance])[0]
            formatted_context = formatted_context + "User: " + user_utterance + ' Domain: '+ line_data[2] + ' Slots: ' + line_data[3] + '\n'
        return formatted_context
    elif dialog_side == 'UA':
        for line in reversed(similar_sentences_lines): # reversed to get the most similar sentence in the last. / near to the testing sentence
            line_data = line.strip().split('\t')
            utterances = line_data[1].strip()
            if '---' in utterances:
                utterances = utterances.split('---')
            else:
                utterances = [utterances]
            for i, utterance in enumerate(utterances):
                if i % 2 == 0:
                    user_utterance = utterance
                    if punct == 'no_punct':
                        user_utterance = remove_punctuations([user_utterance])[0]
                    formatted_context = formatted_context + "User: " + user_utterance.strip() 
                else:
                    agent_utterance = utterance
                    if punct == 'no_punct':
                        agent_utterance = remove_punctuations([agent_utterance])[0]
                    formatted_context = formatted_context.strip() + " Agent: " + agent_utterance.strip() + ' '
            formatted_context = formatted_context.strip() + ' Domain: '+ line_data[2] + ' Slots: ' + line_data[3] + '\n'
        return formatted_context





    # for line in reversed(similar_sentences_lines):  # reversed to get the most similar sentence in the last. / near to the testing sentence
        
        # line_data = line.strip().split('\t')
        # user_utterance = line_data[1].replace("---", " ")
        # formatted_context = formatted_context + "User: " + user_utterance + ' Domain: '+ line_data[2] + ' Slots: ' + line_data[3] + '\n'
        pass
    return formatted_context

# template 2, return in JSON format

def get_text_dom_in_template2(similar_sentences_lines, dialog_side, punct):
    # print("i'm in template 2")
    list_of_conversations = []
    
    if dialog_side == 'U':
        for line in reversed(similar_sentences_lines):
            one_conv_in_dicts_list = []
            line_data = line.strip().split('\t')
            user_utterance = line_data[1].replace("---", " ")
            if punct == 'no_punct':
                user_utterance = remove_punctuations([user_utterance])[0]
            one_conv_in_dicts_list.append({'User': user_utterance})
            one_conv_in_dicts_list.append({'Domain': ast.literal_eval(line_data[2])})
            list_of_conversations.append(one_conv_in_dicts_list)
        return list_of_conversations
    elif dialog_side == 'UA':
        for line in reversed(similar_sentences_lines):
            one_conv_in_dicts_list = []
            line_data = line.strip().split('\t')
            utterances = line_data[1].strip().split('---')
            for i, utterance in enumerate(utterances):
                if i % 2 == 0:
                    user_utterance = {'User': utterance}
                    one_conv_in_dicts_list.append(user_utterance)
                else:
                    agent_utterance = {'Agent': utterance}
                    one_conv_in_dicts_list.append(agent_utterance)

            one_conv_in_dicts_list.append({'Domain': ast.literal_eval(line_data[2])})
            list_of_conversations.append(one_conv_in_dicts_list)
        return list_of_conversations

    # for line in reversed(similar_sentences_lines):  # reversed to get the most similar sentence in the last. / near to the testing sentence
    #     one_conv_in_dicts_list = []
    #     line_data = line.strip().split('\t')
    #     utterances = line_data[1].strip().split('---')
    #     # breakpoint()
    #     for i, utterance in enumerate(utterances):
    #         if i % 2 == 0:
    #             user_utterance = {'User': utterance}
    #             one_conv_in_dicts_list.append(user_utterance)
    #         else:
    #             agent_utterance = {'Agent': utterance}
    #             one_conv_in_dicts_list.append(agent_utterance)

    #     one_conv_in_dicts_list.append({'Domain': line_data[2]})
    #     list_of_conversations.append(one_conv_in_dicts_list)
    #     # print(one_conv_in_dicts_list)
    return list_of_conversations

def get_text_dom_slots_in_template2(similar_sentences_lines, dialog_side, punct):
    # print("i'm in template 2")
    list_of_conversations = []

    if dialog_side == 'U':
        for line in reversed(similar_sentences_lines):
            one_conv_in_dicts_list = []
            line_data = line.strip().split('\t')
            user_utterance = line_data[1].replace("---", " ")
            if punct == 'no_punct':
                user_utterance = remove_punctuations([user_utterance])[0]
            one_conv_in_dicts_list.append({'User': user_utterance})
            one_conv_in_dicts_list.append({'Domain': ast.literal_eval(line_data[2])})
            one_conv_in_dicts_list.append({'Slots': ast.literal_eval(line_data[3])})
            list_of_conversations.append(one_conv_in_dicts_list)
        return list_of_conversations
    elif dialog_side == 'UA':
        for line in reversed(similar_sentences_lines):
            one_conv_in_dicts_list = []
            line_data = line.strip().split('\t')
            utterances = line_data[1].strip().split('---')
            for i, utterance in enumerate(utterances):
                if i % 2 == 0:
                    user_utterance = {'User': utterance}
                    one_conv_in_dicts_list.append(user_utterance)
                else:
                    agent_utterance = {'Agent': utterance}
                    one_conv_in_dicts_list.append(agent_utterance)

            one_conv_in_dicts_list.append({'Domain': ast.literal_eval(line_data[2])})
            one_conv_in_dicts_list.append({'Slots': ast.literal_eval(line_data[3])})
            list_of_conversations.append(one_conv_in_dicts_list)
        return list_of_conversations

    
    # for line in reversed(similar_sentences_lines):  # reversed to get the most similar sentence in the last. / near to the testing sentence
    #     one_conv_in_dicts_list = []
    #     line_data = line.strip().split('\t')
    #     utterances = line_data[1].strip().split('---')
    #     # breakpoint()
    #     for i, utterance in enumerate(utterances):
    #         if i % 2 == 0:
    #             user_utterance = {'User': utterance}
    #             one_conv_in_dicts_list.append(user_utterance)
    #         else:
    #             agent_utterance = {'Agent': utterance}
    #             one_conv_in_dicts_list.append(agent_utterance)

    #     one_conv_in_dicts_list.append({'Domain': line_data[2]})
    #     one_conv_in_dicts_list.append({'Slots': line_data[3]})
    #     list_of_conversations.append(one_conv_in_dicts_list)
        # print(one_conv_in_dicts_list)
    return list_of_conversations

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



# def context_len_check(similar_sentences_lines, context_count):
#     # load the tokenizer
    
        
def get_text_dom_slots_in_template1(similar_sentences_lines, dialog_side, punct):
    formatted_context = ''
    if dialog_side == 'U':
        for line in reversed(similar_sentences_lines): # reversed to get the most similar sentence in the last. / near to the testing sentence
            line_data = line.strip().split('\t')
            user_utterance = line_data[1].replace("---", " ")   # 
            if punct == 'no_punct':
                user_utterance = remove_punctuations([user_utterance])[0]
            formatted_context = formatted_context + "User: " + user_utterance + ' Domain: '+ line_data[2] + ' Slots: ' + line_data[3] + '\n'
        return formatted_context
    elif dialog_side == 'UA':
        for line in reversed(similar_sentences_lines): # reversed to get the most similar sentence in the last. / near to the testing sentence
            line_data = line.strip().split('\t')
            utterances = line_data[1].strip()
            if '---' in utterances:
                utterances = utterances.split('---')
            else:
                utterances = [utterances]
            for i, utterance in enumerate(utterances):
                if i % 2 == 0:
                    user_utterance = utterance
                    if punct == 'no_punct':
                        user_utterance = remove_punctuations([user_utterance])[0]
                    formatted_context = formatted_context + "User: " + user_utterance.strip() 
                else:
                    agent_utterance = utterance
                    if punct == 'no_punct':
                        agent_utterance = remove_punctuations([agent_utterance])[0]
                    formatted_context = formatted_context.strip() + " Agent: " + agent_utterance.strip() + ' '
            formatted_context = formatted_context.strip() + ' Domain: '+ line_data[2] + ' Slots: ' + line_data[3] + '\n'
        return formatted_context


def get_text_dom_slots_in_template3(similar_sentences_lines, dialog_side, punct):
    conversation = ''
    list_of_conversations = []
    # breakpoint()

    if dialog_side == 'U':
        for line in similar_sentences_lines:
            final_format = ''
            conversation = ''
            line_data = line.strip().split('\t')
            user_utterances = line_data[1].strip().replace('---', ' ')
            for user_utterance in user_utterances:
                if punct == 'no_punct':
                    user_utterance = remove_punctuations([user_utterance])[0]
                conversation = conversation + user_utterance
            if punct == 'no_punct':
                user_utterance = remove_punctuations([user_utterance])[0]
            conversation = 'User: ' + conversation.strip()

            # get the domain
            domains = ast.literal_eval(line_data[2])
            # print(domains)
            # print(type(domains))

            # get the slots
            slots = ast.literal_eval(line_data[3])
            # print(slots)
            # print(type(slots))
            # The format of the slot need to be changed - {domain: {slot_key: slot_value}, domain: {slot_key: slot_value}}
            new_dict = {}
            for domain, slot in zip(domains, slots):
                # print(domain)   # domain is string
                # print(slot)     # slot is dictionary
                # print(type(slot))
                # print(type(domain))
                # new_dict = {domain: slot}
                new_dict[domain] = slot
                # print(new_dict)
            
            final_format = {"Dialog": conversation, "Domain": domains, "Slots": new_dict}
            list_of_conversations.append(final_format)
                # else:
                    # break
            # return list_of_conversations
        return list_of_conversations

    # elif dialog_side == 'UA':       # fix this. 
    #     for line in similar_sentences_lines:
    #         final_format = ''
    #         conversation = ''
    #         line_data = line.strip().split('\t')
    #         utterances = line_data[1].strip().split('---')
    #         for i, utterance in enumerate(utterances):
    #             if i % 2 == 0:
    #                 user_utterance = utterance
    #                 if punct == 'no_punct':
    #                     user_utterance = remove_punctuations([user_utterance])[0]
    #                 conversation = conversation + ' User: ' + user_utterance
    #             else:
    #                 agent_utterance = utterance
    #                 if punct == 'no_punct':
    #                     agent_utterance = remove_punctuations([agent_utterance])[0]
    #                 conversation = conversation + ' Agent: ' + agent_utterance
    #         conversation = conversation.strip()
    #         tok_line = tokenizer(conversation, return_tensors='pt')
    #         tok_len = tok_line['input_ids'].shape[1]
    #         total_tok_len = total_tok_len + tok_len
    #         if total_tok_len <= max_tokens:
    #             list_of_conversations.append(conversation)
    #         else:
    #             break
    #     # return list_of_conversations
    #     # reverse the list_of_conversations
    #     sorted_list_of_conversations = list_of_conversations[::-1]
    

    #     # get the domain
    #     domains = ast.literal_eval(line_data[2])
    #     # print(domains)
    #     # print(type(domains))

    #     # get the slots
    #     slots = ast.literal_eval(line_data[3])
    #     # print(slots)
    #     # print(type(slots))
    #     # The format of the slot need to be changed - {domain: {slot_key: slot_value}, domain: {slot_key: slot_value}}
    #     new_dict = {}
    #     for domain, slot in zip(domains, slots):
    #         # print(domain)   # domain is string
    #         # print(slot)     # slot is dictionary
    #         # print(type(slot))
    #         # print(type(domain))
    #         # new_dict = {domain: slot}
    #         new_dict[domain] = slot
    #         # print(new_dict)
        
    #     final_format = {"Dialog": conversation, "Domain": domains, "Slots": new_dict}

    #     # print(str(final_format))

    #     # tokenise, check if the tokens are less than max_tokens, if yes, append to the list_of_conversations. also have a counter for how many contexts are appended.

    #     tok_line = tokenizer(str(final_format), return_tensors='pt')
    #     tok_len = tok_line['input_ids'].shape[1]
    #     # print(tok_len)
    #     total_tok_len = total_tok_len + tok_len
    #     # print(total_tok_len)
    #     if total_tok_len <= max_tokens:
    #         list_of_conversations.append(final_format)
    #         # print("appended")
    #     else:
    #         break

    # return list_of_conversations
    #     # breakpoint()

if __name__ == '__main__':
    pass
#     # sample
#     file = '/mnt/matylda4/hegde/int_ent/LLM_dialog_state/similar_sentences1.tsv'
#     with open(file, 'r') as f:
#         similar_sentences_lines = [line.strip() for line in f]
#     list_of_cconversation = get_text_dom_slots_in_template3(similar_sentences_lines, 'UA', 'punct')
#     print("=================================")
#     # from list, make list_of_conversations a dictionary.
#     conversation_dictionary = list_of_cconversation[0]

#     testing_sentence = '{Dialog: ' + str(conversation_dictionary['Dialog']) + ', Domain: [' 
#     testing_sentence = '{Dialog: ' + str(conversation_dictionary['Dialog']) + ', Domain: ' + str(conversation_dictionary['Domain']) + ', Slots: ['  
#     print(testing_sentence)