import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_x_similar_sentences(train_embeddings, test_embedding, number_of_similar_sentences, train_sentences):
    # Ensure the test_embedding is of shape (1, embedding_size)
    if len(test_embedding.shape) == 1:
        test_embedding = test_embedding.reshape(1, -1)
    
    # Calculate cosine similarity between this test sentence and all training sentences
    cosine_sim = cosine_similarity(test_embedding, train_embeddings)

    top_sim_indices = np.argsort(cosine_sim[0])[::-1][:number_of_similar_sentences]
    picked_train_file_line = [train_sentences[idx] for idx in top_sim_indices]
    
    return picked_train_file_line