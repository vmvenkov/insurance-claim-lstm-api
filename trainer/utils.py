import re


MAX_LEN = 30


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^а-яa-z0-9\s-]", "", text)
    return text.split()


def build_vocab(texts):
    vocab = {"<PAD>": 0, "<UNK>": 1}

    for text in texts:
        for token in tokenize(text):
            if token not in vocab:
                vocab[token] = len(vocab)

    return vocab


def encode_text(text, vocab, max_len=MAX_LEN):
    tokens = tokenize(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]

    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    return ids
