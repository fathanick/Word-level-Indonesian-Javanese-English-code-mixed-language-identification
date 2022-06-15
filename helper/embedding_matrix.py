import numpy as np

def build_embd_matrix(filename, w_idx):
    embedding_idx = {}
    f = open(filename, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        coeff = np.asarray(values[1:], dtype="float32")
        embedding_idx[word] = coeff
    f.close()

    embedding_dim = 50
    embedding_matrix = np.zeros((len(w_idx), embedding_dim))

    for word, i in w_idx.items():
        embedding_vector = embedding_idx.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix