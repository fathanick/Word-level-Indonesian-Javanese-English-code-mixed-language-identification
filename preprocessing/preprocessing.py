import sys
sys.path.insert(0, '../')
import re
from helper.dataset_reader import read_tsv


def replace_retweet_mention(text):
    text = re.sub('RT@[\w_]+: ', '_RTUSERNAME_', text)
    return text

def replace_username(text):
    text = re.sub('@[^\s]+','_USERNAME_', text)
    return text

def replace_hashtags(text):
    text = re.sub('#(\w+)', '_HASHTAG_', text)
    return  text

def replace_url(text):
    text = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+'
                  r'[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))'
                  r'+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '_URL_', text)
    return text

def preprocess(data):
    # input: list of tokens
    # output: preprocessed list of tokens
    new_list = []
    for token in data:
        token = replace_url(token)
        token = replace_username(token)
        token = replace_hashtags(token)
        token = replace_url(token)

        new_list.append(token)

    return new_list

if __name__ == '__main__':
    lst = ['@budi', 'saya', 'http://t.co/1W7NBYC', '#tagMikirSek', 'sudah', '200rb']
    res = preprocess(lst)
    print(res)

    #token_list = data[1]
    #print(token_list[:5])

    #preprocess(data[])
    '''
        Sunday, July 3, 2022
        - Try to execute all comlid data.
        - Pair the preprocessed result with the corresponding tags from data[2]
        - Write new file
    '''






