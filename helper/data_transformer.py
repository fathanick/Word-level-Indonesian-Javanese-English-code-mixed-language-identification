import pandas as pd


def to_token_tag_list(data):

    # input: [[[list of tokens],[list of tags]], [[list of tokens],[list of tags]]]
    # transform data to token and tag pairs: [[(tk1, tg1),(tk2, tg2)], [(tk1, tg1), (tk2, tg2)]]

    # 1. convert to dataframe
    df = pd.DataFrame(data[0], columns=['Tweets', 'Tags'])

    # 2. create token and tag pairs
    token_tag_pair = []
    for index, row in df.iterrows():
        pair = list(zip(row['Tweets'], row['Tags']))
        token_tag_pair.append(pair)

    return token_tag_pair

def token_tag_list(data):

    # input: [[[list of tokens],[list of tags]], [[list of tokens],[list of tags]]]
    # transform data to token and tag pairs: [[[tk1, tg1],[tk2, tg2]], [[tk1, tg1], [tk2, tg2]]]

    # 1. convert to dataframe
    df = pd.DataFrame(data[0], columns=['Tweets', 'Tags'])

    # 2. create token and tag pairs
    token_tag_pair = []
    for index, row in df.iterrows():
        pair = list(zip(row['Tweets'], row['Tags']))
        new_pair = list(map(list, pair))
        token_tag_pair.append(new_pair)

    return token_tag_pair

def list_to_dataframe(data):
    all_data, words, tags = data

    word_tag = list(zip(words, tags))
    # print(word_tag)
    # convert list to dataframe
    df_wordtag = pd.DataFrame(word_tag, columns=['Token', 'Label'])

    return df_wordtag

def get_list_words_tags(all_data):
    # input: all_data format [[[w1, w2, ...],[t1, t2, ...]], ...]
    word_per_sent_list = []
    tag_per_sent_list = []

    for item in all_data:
        word_per_sent_list.append(item[0])
        tag_per_sent_list.append(item[1])

    return word_per_sent_list, tag_per_sent_list

def to_df_tweet_tags(data):
    # input: all_data format [[[w1, w2, ...],[t1, t2, ...]], ...]
    # output: dataframe containing 2 columns
    # Tweets column: list of words
    # Tags column: list of tags
    df = pd.DataFrame(data, columns=['Tweets','Tags'])

    return df

def list_to_tsv(filename, X, y):
    # input X = list of words [w1, w2, ...]
    # input y = list of tags [t1, t2, ...]
    # output => write a tsv file containing word and token pairs

    with open(filename, 'a', encoding='utf-8') as f:
        word_tag = [[w, t] for w, t in zip(X, y)]
        for word, tag in word_tag:
            ls = list(zip(word, tag))
            for l in ls:
                f.write('\t'.join([str(i) for i in l]))
                f.write('\n')
            f.write('\n')

def create_sentence_list(data):
    # input: all_data format [[[w1, w2, ...],[t1, t2, ...]], ...
    # output: list of sentences
    word_lists, tag_lists = get_list_words_tags(data[0])
    sent_list = []
    for word in word_lists:
        sent = ' '.join(word)
        sent_list.append(sent)

    return sent_list