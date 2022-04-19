import re


def token2features(sent, i):
    window = 3
    token = sent[i][0]
    tag = sent[i][1]

    symbols = ['@', '#', '&', '$']

    features = {
        'n_gram_0': token,
        'token.lower': token.lower(),
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

    if i > 0:
        token1 = sent[i - 1][0]
        tag1 = sent[i - 1][1]
        features.update({
            '-1:token.lower': token1.lower(),
            '-1:token.is_alpha': token1.isalpha(),
            '-1:token.is_numeric': token1.isnumeric(),
            '-1:token.is_capital': token1.isupper(),
            '-1:token.is_title': token1.istitle(),
            '-1:token.startswith_symbols': (any(token1.startswith(x) for x in symbols)),
            '-1:token.contains_numeric': bool(re.search('[0-9]', token1)),
            '-1:token.contains_capital': (any(letter.isupper() for letter in token1)),
            '-1:token.contains_quotes': ('\"' in token1) or ('\'' in token1),
            '-1:token.contains_hyphen': '-' in token1,
            '-1:tag': tag1,
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        token1 = sent[i + 1][0]
        tag1 = sent[i + 1][1]
        features.update({
            '+1:token.lower': token1.lower(),
            '+1:token.is_alpha': token1.isalpha(),
            '+1:token.is_numeric': token1.isnumeric(),
            '+1:token.is_capital': token1.isupper(),
            '+1:token.is_title': token1.istitle(),
            '+1:token.startswith_symbols': (any(token1.startswith(x) for x in symbols)),
            '+1:token.contains_numeric': bool(re.search('[0-9]', token1)),
            '+1:token.contains_capital': (any(letter.isupper() for letter in token1)),
            '+1:token.contains_quotes': ('\"' in token1) or ('\'' in token1),
            '+1:token.contains_hyphen': '-' in token1,
            '+1:tag': tag1,
        })
    else:
        features['EOS'] = True

    if len(token) > 2 and not(any(token.startswith(x) for x in symbols)):
        for i in range(0, len(token) - window):
            features[f'n_gram_{i}'] = token[i:(i + window)]

    return features


'''
def token2features(sent, i):
    token = sent[i][0]
    tag = sent[i][1]
    window = 5

    features = {
        'bias': 1.0,
        'n_gram_0': token,
        'token.lower()': token.lower(),
        'token[-3:]': token[-3:],
        'token[-2:]': token[-2:],
        'token.isupper()': token.isupper(),
        'token.istitle()': token.istitle(),
        'token.isdigit()': token.isdigit(),
        'tag': tag,
    }
    if len(token) > 5:
        for i in range(1, len(token) - window):
            features[f'n_gram_{i}'] = token[i:(i + window)]

    if i > 0:
        token1 = sent[i - 1][0]
        tag1 = sent[i - 1][1]
        features.update({
            '-1:token.lower()': token1.lower(),
            '-1:token.istitle()': token1.istitle(),
            '-1:token.isupper()': token1.isupper(),
            '-1:tag': tag1,
            '-1:tag[:2]': tag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent) - 1:
        token1 = sent[i + 1][0]
        tag1 = sent[i + 1][1]
        features.update({
            '+1:token.lower()': token1.lower(),
            '+1:token.istitle()': token1.istitle(),
            '+1:token.isupper()': token1.isupper(),
            '+1:tag': tag1,
            '+1:tag[:2]': tag1[:2],
        })
    else:
        features['EOS'] = True

    return features
'''

def feature_extraction(tokens, window):
    features = []

    for idx, token in enumerate(tokens):
        feature = {
            'n_gram_0': token,
            'is_alpha': token.isalpha(),
            'is_numeric': token.isnumeric(),
            'is_capital': token.isupper(),
            'contains_alpha': bool(re.search('[a-zA-Z]', token)),
            'contains_numeric': bool(re.search('[0-9]', token)),
            'contains_aphostrope': '\'' in token,
        }

        if len(token) > 5:
            for i in range(1, len(token) - window):
                feature[f'n_gram_{i}'] = token[i:(i + window)]

        features.append(feature)

    return features
