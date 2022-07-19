from helper.dataset_reader import read_tsv
from helper.data_transformer import *
from gensim.models import Word2Vec

def w2v_training(method):

    # read raw data
    data = read_tsv("../../dataset/09-06-22/comlid-data-140422-v1.tsv")
    all_data, all_words, all_tags = data

    # create list of words and list of tags
    word_list, tag_list = get_list_words_tags(all_data)

    # build word2vec
    if method == 1: # Skip-gram
        name = 'skip_gram'
        val = 1
    else: # CBOW
        name = 'cbow'
        val = 0

    w2v_model = Word2Vec(min_count=5,
                         window=2,
                         vector_size=50,
                         sample=6e-5,
                         alpha=0.03,
                         min_alpha=0.0007,
                         negative=20,
                         sg=val)

    w2v_model.build_vocab(word_list)
    w2v_model.train(word_list, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    w2v_model.save(f"w2v_{name}.model")
    w2v_model.wv.save_word2vec_format(f"w2v_{name}.txt")

if __name__ == "__main__":
    w2v_training(1) # Skip-gram
    w2v_training(0) # CBOW