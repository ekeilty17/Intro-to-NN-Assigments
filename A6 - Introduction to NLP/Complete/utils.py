import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_data(batch_size=None, preprocess=True, context_length=None, seed=None):
    
    if preprocess:
        
        if context_length is None:
            raise ValueError("Argument 'context_length' is None")

        with open("../data/raw_sentences.txt") as f:
            sentences = [line[:-1].split(' ') for line in f.readlines()]
            sentences = [[word.lower() for word in words if word != ''] for words in sentences]
        
        # getting vocab list, which will help us embeddings later
        # for reproducibility, we sort the word alphabetically
        vocab = list(sorted(set([word for words in sentences for word in words])))

        # getting skip-grams
        #   if context_length = (n-1, 0), then they are called n-grams
        #   n denotes the length of the entire phrase, i.e. total context length + target length
        #   and the target length is almost never not 1
        n = context_length[0] + 1 + context_length[1]
        contexts = []
        targets = []
        for words in sentences:
            # skipping sentences that are too short
            if len(words) < n:
                continue
            
            for i in range(len(words)-(n-1)):
                before = words[i:i+context_length[0]]
                target = words[i+context_length[0]]
                after = words[i+context_length[0]+1:i+n]
                
                contexts.append( before + after )
                targets.append( target )

        
        # embedding inputs and targets
        contexts = list(map(lambda context: [word_to_embedding(vocab, word) for word in context], contexts))
        targets = [word_to_embedding(vocab, word) for word in targets]

        # train/test split
        splits = train_test_split(contexts, targets, test_size=0.2, random_state=seed)
        train_contexts, valid_contexts, train_targets, valid_targets = splits
        
        # creating and saving data to pickle file
        data_obj = {
            "vocab": vocab,
            "context_length": context_length,
            "train_contexts": train_contexts, "train_targets": train_targets,
            "valid_contexts": valid_contexts, "valid_targets": valid_targets
        }
        pickle.dump(data_obj, open("../data/data.pk", 'wb'))

    # loading data object
    data_obj = pickle.load(open("../data/data.pk", 'rb'))
    vocab = data_obj['vocab']
    train_contexts, train_targets = data_obj['train_contexts'], data_obj['train_targets']
    valid_contexts, valid_targets = data_obj['valid_contexts'], data_obj['valid_targets']

    # error checking
    if (not context_length is None) and data_obj['context_length'] != context_length:
        raise ValueError(f"Recieved context_length = {data_obj['context_length']} from file, but expected {context_length}")

    # creating data loaders
    train_dataset = TensorDataset(torch.tensor(train_contexts), torch.tensor(train_targets))
    batch_size = len(train_dataset) if batch_size is None else batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    valid_dataset = TensorDataset(torch.tensor(valid_contexts), torch.tensor(valid_targets))
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_contexts))

    return vocab, train_loader, valid_loader


def get_n_samples(loader, n=1):
    for data, targets in loader:
        return data[0:n], targets[0:n]


def word_to_embedding(vocab, word):
    return vocab.index(word)

def embedding_to_word(vocab, embedding):
    return vocab[embedding]

def embed_data(vocab, context, target):
    embedded_context = [word_to_embedding(vocab, word) for word in context]
    embedded_target = word_to_embedding(vocab, target)
    return embedded_context, embedded_target

def reconstruct_sentence(vocab, context_length, embedded_context, embedded_target):
    sentence = [embedding_to_word(vocab, word) for word in embedded_context]
    sentence.insert(context_length[0], embedding_to_word(vocab, embedded_target))
    return ' '.join(sentence)


"""
context_length = (2, 2)
vocab, train_loader, valid_loader = load_data(batch_size=100, preprocess=False, context_length=context_length, seed=None)
contexts, targets = get_n_samples(train_loader, n=5)

print(vocab[0:10])

print(contexts)
print(targets)

for i in range(5):
    print(reconstruct_sentence(vocab, context_length, contexts[i], targets[i]))
"""
