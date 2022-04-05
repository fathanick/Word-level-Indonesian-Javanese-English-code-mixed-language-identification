import dataset_reader
import numpy as np
from metrics import code_mix_index

data = dataset_reader.read_tsv('../dataset/new-tagged-500.tsv')

cmi_all = []

for words, tags in data[0]:
    cmi = code_mix_index(tags, ['ID', 'JV', 'EN','MIX-JV-EN', 'MIX-ID-JV', 'MIX-ID-EN'])
    cmi_all.append(cmi)

cmi_all = np.array(cmi_all)

print('CMI: ', np.average(cmi_all) * 100)
print('CMI Mixed: ', np.average(cmi_all[cmi_all > 0]) * 100)