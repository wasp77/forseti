class Pdist(dict):
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key): 
        if key in self: return self[key]/self.N  
        else: return self.missingfn(key, self.N)

def ngrams(seq, n, with_counts=True):
    grams = [tuple(seq[i:i+n]) for i in range(1+len(seq)-n)]
    if with_counts:
        counts = {}
        for gram in grams:
            counts[gram] = counts.get(gram, 0) + 1
        return grams, list(counts.items())
    return grams

text_data = ['I', 'am', 'a', 'student', '.', 'I', 'like', 'to', 'study', '.']

unigrams, counts = ngrams(text_data, 1)
unigram_pdist = Pdist(data=counts)
unigram_prob = unigram_pdist(('I'))
print("The probability of the unigram 'I' is:", unigram_prob)


bigrams, counts = ngrams(text_data, 2)
bigram_pdist = Pdist(data=counts)
bigram_prob = bigram_pdist(('I', 'am'))
print("The probability of the bigram ('I', 'am') is:", bigram_prob)


trigrams, counts = ngrams(text_data, 3)
trigram_pdist = Pdist(data=counts)
trigram_prob = trigram_pdist(('I', 'am', 'a'))
print("The probability of the trigram ('I', 'am', 'a') is:", trigram_prob)
