import torch
import torch.nn as nn

class Bengio(nn.Module):

    name = "Bengio"

    def __init__(self, context_length, vocab_size, embedding_size, hidden_size=1000):
        super(Bengio, self).__init__()
        self.context_length = context_length
        self.embedding_size = embedding_size

        self.encode = lambda batch: nn.functional.one_hot(batch, vocab_size)

        self.embedding = nn.Linear(vocab_size, embedding_size)
        
        self.fc = nn.Sequential(
            nn.Linear(sum(context_length) * embedding_size, hidden_size),
            nn.Tanh()
        )

        self.last = nn.Sequential(
            nn.Linear(hidden_size + sum(context_length) * embedding_size, vocab_size)
        )
    
    def forward(self, contexts):
        batch_size = contexts.size(0)
        
        onehot = self.encode(contexts)

        # each word shares the same embedding, since we don't want to learn different embeddings
        # for words in different positions
        embed = torch.empty(batch_size, sum(self.context_length), self.embedding_size)
        for i in range(sum(self.context_length)):
            embed[:, i, :] = self.embedding(onehot[:, i, :].float())
        
        embed_concat = embed.reshape(batch_size, -1).float()
        x = self.fc(embed_concat)
        
        # adding a skip connection
        x = torch.cat((x, embed_concat), dim=1)
        return self.last(x)