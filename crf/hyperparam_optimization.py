from langid_crf import *
from helper.dataset_reader import read_tsv

if __name__ == "__main__":
    data = read_tsv('../dataset/ijelid-batch-1.tsv')
    langid = LanguageIdentifier()
    langid.hyperparameter_optimization(data)