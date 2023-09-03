import numpy as np
from ngrams import ngrams

def score_ngram(doc, model, tokenizer, n=3, strip_first=True):
    scores = []
    if strip_first:
        doc = " ".join(doc.split()[:1000])

    # Add this padding so that we capture the first word
    for i in ngrams((n - 1) * [50256] + tokenizer(doc), n, with_counts=False):
        scores.append(model(tuple(i)))

    return np.array(scores)

