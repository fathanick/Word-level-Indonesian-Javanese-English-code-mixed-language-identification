from helper.dataset_reader import read_tsv
from helper.metrics import code_mix_index
import numpy as np


def calculate_cmi(filename, labels):
    data = read_tsv(filename)

    cmi_all = []

    for words, tags in data[0]:
        cmi = code_mix_index(tags, labels)
        cmi_all.append(cmi)

    cmi_all = np.array(cmi_all)

    # Compute CMI at the corpus level by averaging the values for all sentences.
    # CMI all, include 0 score in the data
    # CMI mixed: only consider data with mix language, exclude tweets with 0 score
    cmi = np.average(cmi_all) * 100
    cmi_mixed = np.average(cmi_all[cmi_all > 0]) * 100

    print('CMI: ', round(cmi, 2))
    print('CMI Mixed: ', round(cmi_mixed, 2))
