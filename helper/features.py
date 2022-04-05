import re, string

def feature_extraction_basic(tokens, window):
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

def feature_extraction_added(tokens, window):
    features = []

    for idx, token in enumerate(tokens):
        feature = {
            'n_gram_0': token,
            'is_alpha': token.isalpha(),
            'is_numeric': token.isnumeric(),
            'is_capital': token.isupper(),
            'is_punctuation': (token in string.punctuation),
            'contains_alpha': bool(re.search('[a-zA-Z]', token)),
            'contains_numeric': bool(re.search('[0-9]', token)),
            'contains_aphostrope': '\'' in token,
        }

        if len(token) > 5:
            for i in range(1, len(token) - window):
                feature[f'n_gram_{i}'] = token[i:(i+window)]

        features.append(feature)
    
    return features