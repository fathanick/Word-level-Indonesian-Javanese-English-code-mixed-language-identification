import re

symbols = ['@', '#', '&', '$']
window = 5

def feature_extraction(tokens):
    features = []

    for index, token in enumerate(tokens):
        feature = {
            'n_gram_0': token,
            'token_BOS': index == 0,
            'token_EOS': index == len(tokens) - 1,
            'token.prefix_2': token[:2],
            'token.prefix_3': token[:3],
            'token.suffix_2': token[-2:],
            'token.suffix_3': token[-3:],
            'token.length': len(token),
            'token.is_alpha': token.isalpha(),
            'token.is_numeric': token.isnumeric(),
            'token.is_capital': token.isupper(),
            'token.is_title': token.istitle(),
            'token.startswith_symbols': (any(token.startswith(x) for x in symbols)),
            'token.contains_numeric': bool(re.search('[0-9]', token)),
            'token.contains_capital': (any(letter.isupper() for letter in token)),
            'token.contains_quotes': ('\"' in token) or ('\'' in token),
            'token.contains_hyphen': '-' in token,
        }

        features.append(feature)

    return features


def token2features(sentence, index):
    # sentence: [w1, w2, ...], index: the index of the word
    # apply features for each token

    token = sentence[index][0]

    features = {
        'token': token,
        'n_gram_0': token,
        'token_BOS': index == 0,
        'token_EOS': index == len(sentence) - 1,
        'prev_tag': '' if index == 0 else sentence[index - 1][1],
        'next_tag': '' if index == len(sentence) - 1 else sentence[index + 1][1],
        'prev_2tag': '' if index == 0 or index == 1 else sentence[index - 2][1],
        'next_2tag': '' if index == len(sentence) - 1 or index == len(sentence) - 2 else sentence[index + 2][1],
        'token.prefix_2': token[:2],
        'token.prefix_3': token[:3],
        'token.suffix_2': token[-2:],
        'token.suffix_3': token[-3:],
        'token.length': len(token),
        'token.is_alpha': token.isalpha(),
        'token.is_numeric': token.isnumeric(),
        'token.is_capital': token.isupper(),
        'token.is_title': token.istitle(),
        'token.startswith_symbols': (any(token.startswith(x) for x in symbols)),
        'token.contains_numeric': bool(re.search('[0-9]', token)),
        'token.contains_capital': (any(letter.isupper() for letter in token)),
        'token.contains_quotes': ('\"' in token) or ('\'' in token),
        'token.contains_hyphen': '-' in token,
    }

    if len(token) > 5 and not (any(token.startswith(x) for x in symbols)):
        for i in range(0, len(token) - window):
            features[f'n_gram_{i}'] = token[i:(i + window)]

    return features