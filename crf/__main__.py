from lang_id_crf import LanguageIdentifier
from helper.dataset_reader import read_tsv

import sys
sys.path.insert(0, '../helper')

langid = LanguageIdentifier()

data = read_tsv('../raw dataset/all-tagged-280322.tsv')
#data = read_tsv('../raw dataset/all-tagged-280322-v2.tsv')
# Scenario 1: 60:40
#print('Scenario 1')
#langid.pipeline(data, test_size=0.4, model_name='mod_6_4_dtv2.pkl')

# Scenario 2: 70:30
#print('\n\n Scenario 2')
#langid.pipeline(data, test_size=0.3, model_name='mod_7_3_dtv2.pkl')

# Scenario 3: 80:20
print('\n\n Scenario 3')
langid.pipeline(data, test_size=0.2, model_name='mod_8_2.pkl')

