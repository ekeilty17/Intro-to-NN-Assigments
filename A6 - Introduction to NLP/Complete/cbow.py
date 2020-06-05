import torch
import torch.nn as nn

class CBOW(nn.Module):

    name = "CBOW"

    def __init__(self, context_length, vocab_size):
        super(CBOW, self).__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size

        self.encode = lambda batch: nn.functional.one_hot(batch, vocab_size)

        self.embedding = nn.Linear(vocab_size, vocab_size)
        
        self.fc = nn.Linear(sum(context_length) * vocab_size, vocab_size)
    
    def forward(self, contexts):
        batch_size = contexts.size(0)

        onehot = self.encode(contexts)
        
        # each word shares the same embedding weights, 
        # since we don't want to learn different embeddings for words in different positions
        embed = torch.empty(batch_size, sum(self.context_length), self.vocab_size)
        for i in range(sum(self.context_length)):
            embed[:, i, :] = self.embedding(onehot[:, i, :].float())

        # get word embeddings, which in our case is just the flattened onehot encodings
        embed_concat = onehot.reshape(batch_size, -1).float()

        return self.fc(embed_concat)