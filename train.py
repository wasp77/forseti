import tiktoken
import tqdm
from nltk.corpus import brown
from ngrams import Pdist, ngrams

def train_ngrams(verbose=True, return_tokenizer=True):
    """
    Trains and returns a trigram model on the brown corpus
    """

    enc = tiktoken.encoding_for_model("davinci")
    tokenizer = enc.encode
    vocab_size = enc.n_vocab

    sentences = brown.sents()

    if verbose:
        print("Tokenizing corpus...")
    tokenized_corpus = []
    for sentence in tqdm.tqdm(sentences):
        tokens = tokenizer(' '.join(sentence))
        tokenized_corpus += tokens

    if verbose:
        print("\nTraining n-gram model...")

    trigrams, counts = ngrams(tokenized_corpus, 3)
    trigram_model = Pdist(data=counts)

    unigrams, counts = ngrams(tokenized_corpus, 1)
    unigram_model = Pdist(data=counts)
    if return_tokenizer:
        return trigram_model, unigram_model, tokenizer
    return trigram_model, unigram_model
