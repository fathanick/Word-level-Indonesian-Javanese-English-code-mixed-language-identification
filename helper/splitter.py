from nltk.tokenize import wordpunct_tokenize, word_tokenize, TweetTokenizer
import re
import emot
from helper.emoticon import STANDARD_EMOTICON_LISTS, NON_STANDARD_EMOTICON_LISTS

tknzr = TweetTokenizer()


def isFloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

    # A function to check emoji from string


def text_has_emoji(text):
    # input: a sentence
    # output: True or False

    emot_obj = emot.core.emot()
    result = emot_obj.emoji(text)

    return result['flag']


def sentence_splitter(sentence):
    token_list = []
    # symbols = ['_','-','~','.',',','&','+','!']
    tokens = sentence.split()
    for t in tokens:
        if t in STANDARD_EMOTICON_LISTS.values() or t in NON_STANDARD_EMOTICON_LISTS.values() or isFloat(t):
            token_list.append(t)
        elif re.match(r'[\w+]+[-]+[\w+]+[.,!?]', t):
            tkns = word_tokenize(t)
            for tkn in tkns:
                token_list.append(tkn)
        elif re.match(r'\b(\w[^0-9]+[-]\w[^0-9]+)', t) or re.match(
                r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+', t):
            token_list.append(t)
        elif re.match(r'[0-9]+[%]|[#]+[\w]', t) or re.match(r'[0-9]+[\-]+[0-9]+[\-]+[0-9]', t):
            token_list.append(t)
        elif re.match(r'[a-zA-Z*]+[0-9*]+[\.\,*]+[0-9*]+[\.\,*]+[0-9*]', t):
            token_list.append(t)
        elif re.match(r'[\w+]+[\*\“\"\”\`\']+[\w+]', t):
            token_list.append(t)
        elif text_has_emoji(t) or re.match(r'[a-zA-Z]+[…]|[\w+]+[:]', t):
            # if a word contains emoji
            tkns = tknzr.tokenize(t)
            for tk in tkns:
                token_list.append(tk)
        elif re.match(r'[\(\[\{]+[\w+]+[\)\]\}]|[\(\[\{]+[\w+]|[\w+]+[\)\]\}]', t) or \
                re.match(r'[0-9]+[\,\-]+[0-9]', t):
            tkns = word_tokenize(t)
            for tkn in tkns:
                token_list.append(tkn)
        elif (re.match(r'\b(\w[^0-9]+[.]\w[^0-9]+)', t) or re.match(r'[a-zA-Z]+[/][a-zA-Z]', t)) and not (
        re.match(r'https?:\/\/.*[\r\n]*', t)) and not (
        re.match(r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+', t)):
            tkns = wordpunct_tokenize(t)
            for tkn in tkns:
                token_list.append(tkn)
        elif re.match(r'[~-]|[\+]+[a-zA-Z0-9]|[a-zA-Z0-9]+[~-]|[\+]', t) or re.match(
                r'[\*\“\"\”\`\']+[\w]|[\w]+[\*\“\"\”\`\']', t):
            tkns = wordpunct_tokenize(t)
            for tkn in tkns:
                token_list.append(tkn)
        elif re.match(r'[\.\,\?\!\&]+[\w+]|[\w+]+[\.\,\?\!\&]', t) or re.match(r'[\w+]+[\+]+[\w+]',t):
            tkns = wordpunct_tokenize(t)
            for tkn in tkns:
                token_list.append(tkn)
        else:
            token_list.append(t)

    return token_list
