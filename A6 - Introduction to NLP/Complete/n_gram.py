import torch
import numpy as np
from collections import defaultdict

from utils import *
from plot import *

class N_Gram(object):

    def __init__(self, vocab, n):
        self.vocab = vocab
        self.n = n
        self.context_length = n-1       # here context_length is not a tuple and is just a single number
                                        # since n-grams assume we only take context from before the target word
        

        self.n_gram_counts = defaultdict(lambda: 0)
        self.context_counts = defaultdict(lambda: 1)

    def train(self, train_loader):
        for batch in train_loader:
            contexts, targets = batch
            for context, target in zip(contexts, targets):
                # converting data from torch objects back to python objects
                context = [int(word) for word in context]
                target = int(target)
                
                # getting counts to calculate empirical distribution at test time
                n_gram = list(context) + [target]
                self.n_gram_counts[tuple(n_gram)] += 1
                self.context_counts[tuple(context)] += 1
    
    # we do the log to improve precision becuase data is sparse
    # since we initialize all context pairs with a count of 1, it will always be defined
    def log_empirical_distribution(self, context, word):
        p_sentence = self.n_gram_counts[ tuple(list(context) + [word]) ]
        p_context = self.context_counts[ tuple(context) ]
        return -np.inf if p_sentence == 0 else np.log( p_sentence / p_context )

    def get_row_of_conditional_probability_table(self, context):
        return {embedded_word : self.log_empirical_distribution(context, embedded_word) for embedded_word in range(len(self.vocab))}

    def predict(self, sentence, embed=True):
        if len(sentence) < self.context_length:
            raise ValueError("Not enough words in the sentence for the model to make a prediction. Needs at least", self.context_length)
        
        # markov assumption of our language model
        context = sentence[-self.context_length:]

        # some error checking
        if embed:
            for word in context:
                if not word in self.vocab:
                    raise ValueError(f"'{word}' is not in model's vocabulary")
            context = [word_to_embedding(vocab, word) for word in context]
        else:
            for embedded_word in context:
                if embedded_word >= len(self.vocab):
                    raise ValueError(f"embedding '{embedded_word}' is out of range of the model's vocabulary. Maximum valued embedding is {len(self.vocab)-1}")
        
        # getting explicit representation of our conditional probability table using the empirical distibution
        distribution = self.get_row_of_conditional_probability_table(context)

        # get the MLE estimator for the next word
        _, max_prob = max(distribution.items(), key=lambda t: t[1])
        
        # breaking ties randomly
        max_prob_words = {word: prob for word, prob in distribution.items() if prob == max_prob}
        embedded_word = np.random.choice(list(max_prob_words.keys()))
        
        # return un-embedded word if the input was not embedded, else return the embedded word
        return embedding_to_word(self.vocab, embedded_word) if embed else embedded_word
             

    def validate(self, loader):
        corr = 0
        total_data = 0
        for batch in train_loader:
            contexts, targets = batch
            for context, target in zip(contexts, targets):
                # converting data from torch objects back to python objects
                context = [int(word) for word in context]
                target = int(target)

                # getting prediction and computing accuracy
                prediction = self.predict(context, embed=False)
                if prediction == target:
                    corr += 1
                total_data += 1
        
        return float(corr) / total_data


if __name__ == "__main__":
    
    # getting args
    opts = AttrDict()
    args_dict = {
        "seed": None,
        "batch_size": None,
        "context_length": (3, 0),       # n-gram --> context_length = (n-1, 0)
        "validate": False
    }
    opts.update(args_dict)

    if not opts.seed is None:
        torch.manual_seed(opts.seed)

    vocab, train_loader, valid_loader = load_data(  batch_size=opts.batch_size, seed=opts.seed,
                                                    preprocess=True, context_length=opts.context_length)

    n = opts.context_length[0] + 1
    model = N_Gram(vocab, n)


    print("Training...")
    model.train(train_loader)
    print("Done Training")

    # This takes forever to run, so I just wrote down the answers in the comments so you don't have to run it
    if opts.validate:
        print("Validating...")
        train_acc = model.validate(train_loader)        # = 54.04%
        valid_acc = model.validate(valid_loader)        # = 54.02%
        print("Done Validating")

        print()
        display_statistics(train_acc=[train_acc], valid_acc=[valid_acc], plot=False)
        print()

    while True:
        sentence = input("Sentence: ")
        sentence = sentence.lower().split(' ')
        
        try:
            next_word = model.predict(sentence)
            print(next_word)
        except ValueError as e:
            print(e)
        except KeyboardInterrupt:
            break