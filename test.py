import tqdm
from train import train_ngrams
from utils import score_ngram

trigram, unigram, tokenizer = train_ngrams()

file_names = ['./data/human/0.txt', './data/gpt/0.txt']
trigram_logprobs, unigram_logprobs = {}, {}

for file in tqdm.tqdm(file_names):
    with open(file, "r") as f:
        doc = f.read()

    trigram_logprobs[file] = score_ngram(doc, trigram, tokenizer, n=3)
    unigram_logprobs[file] = score_ngram(doc, unigram, tokenizer, n=1)

for file in file_names:
    print(f"file: {file}")
    print(f"trigram:")
    print(trigram_logprobs[file][:5])
    print(f"unigram:")
    print(unigram_logprobs[file][:5])
