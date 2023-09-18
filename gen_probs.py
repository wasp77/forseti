from os import listdir
from os.path import isfile, join
import tqdm
from joblib import dump
from utils import score_ngram
from train import train_ngrams
import numpy as np

human_path = './data/human'
gpt_path = './data/gpt'

human_files = [join(human_path, f) for f in listdir(human_path) if isfile(join(human_path, f))]
gpt_files = [join(gpt_path, f) for f in listdir(gpt_path) if isfile(join(gpt_path, f))]

trigram_logprobs, unigram_logprobs = {}, {}

trigram, unigram, tokenizer = train_ngrams()

def gen_name(path, suffix = ''):
    parts = path.split('/')
    parts_len = len(parts)
    return f"{suffix}{parts[parts_len - 1]}"

for file in tqdm.tqdm(human_files):
    with open(file, "r") as f:
        doc = f.read()

    trigram_logprobs[gen_name(file, 'human_')] = np.array(score_ngram(doc, trigram, tokenizer, n=3))
    unigram_logprobs[gen_name(file, 'human_')] = np.array(score_ngram(doc, unigram, tokenizer, n=1))

for file in tqdm.tqdm(gpt_files):
    with open(file, "r") as f:
        doc = f.read()

    trigram_logprobs[gen_name(file, 'gpt_')] = np.array(score_ngram(doc, trigram, tokenizer, n=3))
    unigram_logprobs[gen_name(file, 'gpt_')] = np.array(score_ngram(doc, unigram, tokenizer, n=1))

probs = {'trigram': trigram_logprobs, 'unigram': unigram_logprobs}
dump(probs, './probs.joblib')

