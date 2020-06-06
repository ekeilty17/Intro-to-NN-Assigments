import torch
import torch.nn as nn

class CBOW(nn.Module):

    name = "CBOW"

    def __init__(self, context_length, vocab_size, embedding_size=None):
        super(CBOW, self).__init__()
        self.context_length = context_length
        self.embedding_size = vocab_size if embedding_size is None else embedding_size

        self.encode = lambda batch: nn.functional.one_hot(batch, vocab_size)

        raise NotImplementedError

        self.embedding = ## TODO
        
        self.fc = ## TODO
    
    def forward(self, contexts):
        batch_size = contexts.size(0)

        onehot = self.encode(contexts)
        
        # each word shares the same embedding weights, 
        # since we don't want to learn different embeddings for words in different positions
        embed = torch.empty(batch_size, sum(self.context_length), self.embedding_size)
        for i in range(sum(self.context_length)):
            embed[:, i, :] = self.embedding(onehot[:, i, :].float())

        raise NotImplementedError

        # get word embeddings, which in our case is just the flattened onehot encodings
        embed_flat = ## TODO

        return self.fc(embed_flat)
