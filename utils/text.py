from nltk import tokenize

def to_sentences(text):
    return tokenize.sent_tokenize(text)