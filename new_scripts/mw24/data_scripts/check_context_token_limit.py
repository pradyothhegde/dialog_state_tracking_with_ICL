import ast
import torch
from transformers import AutoTokenizer
import string


def remove_punctuations(sentences):
    # print('Removing punctuations from the sentences...')
    table = str.maketrans('', '', string.punctuation)
    return [s.translate(table).lower() for s in sentences]

def finding_test_labels(test_line):
    test_line_data = test_line.strip().split('\t')
    gold_filename = test_line_data[0]
    gold_sentence = test_line_data[1]
    gold_sentence = gold_sentence.replace('---', ' ')
    gold_domain = test_line_data[2]
    gold_slots = test_line_data[3]
    return gold_filename, gold_sentence, gold_domain, gold_slots

def check_context_token_limit(top_sentences, test_sentence, args):
    dialog_side = args.dialog_history

    model_name = args.LLM_model_tokenizer
    max_token_limit = args.LLM_model_tokenizer_limit
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    limited_top_sentences = []
    total_tok_len = 0
    tmp_tst_fname, tmp_tst_sentence, tmp_tst_domain, tmp_tst_slots = finding_test_labels(test_sentence)
    tmp_tst_conv = 'User: ' + tmp_tst_sentence
    tmp_tsv_domain = ast.literal_eval(tmp_tst_domain)
    tmp_tsv_final_slots = {}
    for tmp_dom, tmp_slt in zip(ast.literal_eval(tmp_tst_domain), ast.literal_eval(tmp_tst_slots)):
        tmp_tsv_final_slots[tmp_dom] = tmp_slt
    tst_token_checker = tmp_tst_conv + ' Domain: ["' + '", "'.join(tmp_tsv_domain) + '"] Slots: ' + str(tmp_tsv_final_slots)
    test_tokens = tokenizer(tst_token_checker, return_tensors='pt')
    test_tok_len = test_tokens['input_ids'].shape[1]
    max_tokens = max_token_limit - 64 - test_tok_len    # 64 tokens less to be safe. 

    for line in top_sentences:
        conversation = ''
        _, _, temp_domain, temp_slots = finding_test_labels(line)
        if dialog_side == 'U':
            # change the token_checker to include the domain and slots
            temp_fname, temp_sentence, temp_domain, temp_slots = finding_test_labels(line)
            # temp_domain = ast.literal_eval(line.split('\t')[2])
            # temp_slots = ast.literal_eval(line.split('\t')[3])
            if args.punct == 'N':
                temp_sentence = remove_punctuations([temp_sentence])[0]
            conversation = 'User: ' + temp_sentence  # Sentence for token count
            # print(f"temp conversation: {conversation}")
            # print(f"temp_domain: {temp_domain}")
            # print(f"temp_slots: {temp_slots}")
        elif dialog_side == 'UA':
            line_data = line.split('\t')
            utterances = line_data[1].strip()
            utterances = utterances.split('---')
            # print(f"utterances: {utterances}")
            # print(f"type of utterances: {type(utterances)}")
            for i, utterance in enumerate(utterances):
                if args.punct == 'N':
                    utterance = remove_punctuations([utterance])[0]
                if i % 2 == 0:
                    user_utterance = utterance
                    conversation = conversation + "User: " + user_utterance.strip() 
                else:
                    agent_utterance = utterance
                    conversation = conversation.strip() + " Agent: " + agent_utterance.strip() + ' '
            # print(f"conversation: {conversation}")
        # breakpoint()
        # continue
        
        slot_dict = {}
        for tmp_dom, tmp_slt in zip(ast.literal_eval(temp_domain), ast.literal_eval(temp_slots)):
            # print(f"domain: {tmp_dom}, slot: {tmp_slt}")
            slot_dict[tmp_dom] = tmp_slt
        token_checker = conversation + ' Domain: ["' + '", "'.join(ast.literal_eval(temp_domain)) + '"] Slots: ' + str(slot_dict)
        
        # print(f"token_checker: {token_checker}")            
        # breakpoint()
        tok_line = tokenizer(token_checker, return_tensors='pt')
        tok_len = tok_line['input_ids'].shape[1]
        total_tok_len += tok_len

        if total_tok_len <= max_tokens:
            limited_top_sentences.append(line)
        else:
            break

    return limited_top_sentences, len(limited_top_sentences)