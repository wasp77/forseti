from utils import FUNCTIONS_DICT
import numpy as np
from joblib import load, dump


probs = load('./probs.joblib')
unigram_probs = probs['unigram']
trigram_probs = probs['trigram']

def process_chain(parts, model_probs, current_index=0, current_prob=None):
    if current_index == len(parts):
        return current_prob

    part = parts[current_index]

    if not part:
        return process_chain(parts, model_probs, current_index+1, current_prob)

    if part in model_probs:
        current_prob = model_probs[part]
        return process_chain(parts, model_probs, current_index+1, current_prob)

    # binary function
    if 'f' in part and '(' in part:
        function_name = part.split('(')[0]
        next_model = part.split('(')[1][:-1]

        current_prob, func_name = FUNCTIONS_DICT[function_name](current_prob, model_probs[next_model])
        return process_chain(parts, model_probs, current_index+1, current_prob)

    # scalar function
    if 'f' in part:
        current_prob, func_name = FUNCTIONS_DICT[part](current_prob)
        return process_chain(parts, model_probs, current_index+1, current_prob)

sample_size = len(unigram_probs.items())

dataset = {'label': np.array([])}
features = []
with open('./features.txt', "r") as f:
    lines = f.readlines()
    for line in lines:
        feature = line.strip()
        features.append(feature)
        dataset[feature] = np.array([])

for (unigram_file,unigram_prob), (trigram_file,trigram_prob) in zip(unigram_probs.items(), trigram_probs.items()):
    model_probs = {'unigram': unigram_prob, 'trigram': trigram_prob}
    for feature in features:
        parts = feature.split("->")
        np.append(dataset[feature], process_chain(parts, model_probs))
    if 'gpt' in unigram_file:
        np.append(dataset['label'], 1)
    else:
        np.append(dataset['label'], 0)

dump(dataset, './dataset.joblib')
