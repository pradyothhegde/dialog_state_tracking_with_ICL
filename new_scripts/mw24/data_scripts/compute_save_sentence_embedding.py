import torch
from sentence_transformers import SentenceTransformer
import numpy as np

def compute_save_sentence_embedding(tsv_file_path):    # file, dialog_side, punct
    # Calculate sentence embedding and save in npy file
    # train_tsv_file = os.path.join(data_path, 'mw24_DST_train_turns.tsv')
    # test_tsv_file = os.path.join(data_path, 'mw24_DST_test_turns.tsv')
    # print('Calculating sentence embeddings...')

    all_sentences = prepare_data_for_sentence_embedding(tsv_file_path)    # pick option from args
    # test_sentences = prepare_data_for_sentence_embedding(test_tsv_file, sentence_embedding_option)     # pick option from args

    # numpy file will be saved in the same directory, name as the tsv file
    numpy_file_path = tsv_file_path.replace('.tsv', '.npy')
    
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


def prepare_data_for_sentence_embedding(tsv_file_path):
    # Load the [train / test] tsv file to extract [user / user-agent] sentences.
    # open test tsv file.
    all_sentences = []
    # print('Preparing data for sentence embedding...')
    with open(tsv_file_path, 'r') as tsv_file:
        for line in tsv_file:
            line_data = line.strip().split('\t')
            gold_filename = line_data[0]

            sentences = line_data[1].strip()
            sentences = sentences.replace('---', ' ')     # remove --- from the user-agent utterance
            # domain = line_data[2]
            # slots = line_data[3]
            # agent_utterance = line_data[4].strip()
            all_sentences.append(sentences)
    return all_sentences

