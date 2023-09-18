import numpy as np
import tqdm
from train import train_ngrams
from utils import score_ngram, FUNCTIONS_DICT

trigram, unigram, tokenizer = train_ngrams()

file_names = ['./data/human/0.txt', './data/gpt/0.txt']
trigram_logprobs, unigram_logprobs = {}, {}

for file in tqdm.tqdm(file_names):
    with open(file, "r") as f:
        doc = f.read()

    trigram_logprobs[file] = score_ngram(doc, trigram, tokenizer, n=3)
    unigram_logprobs[file] = score_ngram(doc, unigram, tokenizer, n=1)

# Given data
trigram_probs = np.array(trigram_logprobs['./data/gpt/0.txt'])
unigram_probs = np.array(unigram_logprobs['./data/gpt/0.txt'])

scalar_functions = [FUNCTIONS_DICT['fmax'], FUNCTIONS_DICT['fmin'], FUNCTIONS_DICT['favg'], FUNCTIONS_DICT['flen'], FUNCTIONS_DICT['fL2'], FUNCTIONS_DICT['fvar'], FUNCTIONS_DICT['favg_top_25']]
binary_functions = [FUNCTIONS_DICT['fadd'], FUNCTIONS_DICT['fsub'], FUNCTIONS_DICT['fmul'], FUNCTIONS_DICT['fdiv'], FUNCTIONS_DICT['f_greater'], FUNCTIONS_DICT['f_less']]
vector_data = {"unigram": unigram_probs, "trigram": trigram_probs}
max_depth = 2  # Or any other limit

# Recursive function
def find_all_feats(p, d=1, call_chain="", model_name=""):
    if d > max_depth:
        return []

    features = []

    # Apply scalar functions
    for func in scalar_functions:
        result, func_name = func(p)
        desc = f"{model_name}->{call_chain}->{func_name}".strip('->')  # Construct descriptor including model_name
        features.append((result, desc))

    # Apply binary functions with vector combinations
    for vec_name, vec in vector_data.items():
        for func in binary_functions:
            result, func_name = func(p, vec)
            desc = f"{model_name}->{call_chain}->{func_name}({vec_name})".strip('->')  # Include both models in descriptor
            features.extend(find_all_feats(result, d + 1, desc, vec_name))

    return features

# Main execution
features = find_all_feats(unigram_probs, model_name="unigram")  # Starting with unigram
features.extend(find_all_feats(trigram_probs, model_name="trigram"))  # Then add trigram

# Save results
with open('features.txt', 'w') as file:
    for feature, desc in features:
        file.write(f"{desc}\n")

