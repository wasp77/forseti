import numpy as np
from ngrams import ngrams

# Scalar functions
def fmax(p): return np.max(p), "fmax"
def fmin(p): return np.min(p), "fmin"
def favg(p): return np.mean(p), "favg"
def flen(p): return len(p), "flen"
def fL2(p): return np.linalg.norm(p), "fL2"
def fvar(p): return np.var(p), "fvar"
def favg_top_25(p): return np.mean(sorted(p)[-int(0.25*len(p)):]), "favg_top_25"

# Binary functions
def fadd(p1, p2): return p1 + p2, "fadd"
def fsub(p1, p2): return p1 - p2, "fsub"
def fmul(p1, p2): return p1 * p2, "fmul"
def fdiv(p1, p2): return p1 / p2, "fdiv"
def f_greater(p1, p2): return (p1 > p2).astype(int), "f_greater"
def f_less(p1, p2): return (p1 < p2).astype(int), "f_less"

FUNCTIONS_DICT = {
    "fmax": fmax,
    "fmin": fmin,
    "favg": favg,
    "flen": flen,
    "fL2": fL2,
    "fvar": fvar,
    "favg_top_25": favg_top_25,
    "fadd": fadd,
    "fsub": fsub,
    "fmul": fmul,
    "fdiv": fdiv,
    "f_greater": f_greater,
    "f_less": f_less
}



def score_ngram(doc, model, tokenizer, n=3, strip_first=True):
    scores = []
    if strip_first:
        doc = " ".join(doc.split()[:1000])

    # Add this padding so that we capture the first word
    for i in ngrams((n - 1) * [50256] + tokenizer(doc), n, with_counts=False):
        scores.append(model(tuple(i)))

    return np.array(scores)

