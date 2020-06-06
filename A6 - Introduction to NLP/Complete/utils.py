# NLP Libraries
import spacy
# pip install spacy
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_md
import nltk

# data processing
import pickle
import numpy as np
import pandas as pd
import torch
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# gives the pretty progress bar
from tqdm import tqdm


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


# I don't need this as a separate function, but it makes the code a bit more readable
def preprocess_data(context_length, seed, lemmatize, stem, remove_stopwords, library):
    """
        When skimming through this code, pay particularly close attention to when I have
            if library == "spacy":
                # stuff
            else:   # library == "nltk"
                # stuff
        becuase this shows you the syntax for using these libraries to do NLP preprocessing for you.
        Their documentation is not the best, so finding the sytax took a bit of digging
    """

    # we need to narrow the number of tokens used down the vocabulary because right now it can be over 1000
    # and for demo purposes, that's way too many to visualize, so we will find the top 250 more frequent tokens 
    # and only take sentences from the dataset that contains those tokens
    MAX_VOCAB_LENGHT = 250

    # load full dataset
    print("Loading data...")
    df_friends = pd.read_csv("../data/friends_dataset.csv", index_col=0)
    
    # dropping NaN rows
    df_friends = df_friends.dropna()
    
    # extracting text from full corpus
    corpus = list(df_friends["Text"].values)            # if we use all of the data the tokenizer takes way too long

    # loading tokenizer
    print("Tokenizing...")
    
    tokenizer = None
    if library == "spacy":
        #tokenizer = spacy.load("en_core_web_sm")       # small-sized model   
        tokenizer = spacy.load("en_core_web_md")        # medium-sized model
    else:    # library == "nltk"
        nltk.download('punkt')
        tokenizer = nltk.tokenize.word_tokenize
    

    # tokenizing sentences
    #   the spacy tokenizer returns a spacy object, which does stuff that we don't really need
    #   the nltk tokenizer just returns a list of strings
    corpus = [ tokenizer(sentence.lower()) for sentence in tqdm(corpus) ]
    
    # Here we implement the option to lemmatize, stem, removing stopwords or none
    if lemmatize:
        print("Lemmatizing...")
        
        if library == "spacy":
            # lemma's of each token are part of the token object
            # you can also call
            #   token.pos_ to get the "Part-Of-Speech" = {NOUN, VERB, etc ...}
            #   token.dep_ to get its "sytactic dependency" = {punc, nsubj, neg, dobj, etc ...}
            corpus = [ [token.lemma_ for token in sentence] for sentence in corpus ]
        else:    # library == "nltk"
            nltk.download('wordnet')
            lemma = nltk.wordnet.WordNetLemmatizer()
            corpus = [ [lemma.lemmatize(token) for token in sentence] for sentence in corpus ]
    
    elif stem:
        print("Stemming...")
        
        # spacy does not have a stemmer,
        # but nltk has a variety of different English stemming algorithms
        # different stemming algorithms simply use different heuristics to determine when to chop off the end of a word and when not to
        stemmer = nltk.stem.porter.PorterStemmer()                  # most well known
        #stemmer = nltk.stem.snowball.SnowballStemmer("english")    # this stemmer works for a number of languages
        #stemmer = nltk.stem.lancaster.LancasterStemmer()           
        
        if library == "spacy":
            corpus = [ [stemmer.stem(token.text) for token in sentence] for sentence in corpus ]
        else:    # library == "nltk"
            corpus = [ [stemmer.stem(token) for token in sentence] for sentence in corpus ]

    else:
        if library == "spacy":
            #                .text just gets the plain python string from the spacy object
            corpus = [ [token.text for token in sentence] for sentence in corpus ]
        else:    # library == "nltk"
            # nltk just returns a list of strings so we don't need to do anything
            pass
    
    # filtering out sentences containing stopwords
    if remove_stopwords:
        print("Removing stop words...")
        stopwords = [
                '*', '..', '...', '....', '......', '[', ']', '--', '{', '}', '…',
                '1,000', '1,500', '10', '100', '18th', '20', '200', '2300', '250', '29', '300', '8', '930', '2', '27', '400',
                'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'sixty', 'eighth',
                'ow', 'ah', 'ahh', 'ahhh', 'aahhuuhhh', 'ahahaaa', 'b', 'dirk', 'em', 'gay', 'ha', 'i.', 'mmm', 'moo', 'whoo', 'whooo'
                'nah', 'oy', 'tov', 'what-', "'m", 'y', "y'know", 'ya', 'yo-', 'wewee', 'wha', 'what-',
                'emily', 'ben', 'nina', 'espn', 'mazel', 'richard', 'jess', 'janice', 'laura', 'mary', 'dan', 'merrill', 'mindy',
                'uh', 'um', 'umm', 'wo', 'woo', 'woohoo', 'oww', 'oh', 'ohh', 'huh', 'ho', 'hm', 'hmm', 'en', 'je', "c'mon",
                "m'appelle", 'yes…', 'you-you', '—nothing', '…betrothed', 'betrothed…', 'well…', 'oh…', 'ooh', 'oooh', 'oooooh',
                'ni-hi-ice', 'everybody…', 'hey-hey', 'hey…', 'ha-ha-ha', 'hah', 'cheerie-o', 'eh', 'ehh'
        ]
        if library == "spacy":
            stopwords += spacy.lang.en.stop_words.STOP_WORDS
        else:    # library == "nltk"
            nltk.download('stopwords')
            stopwords += nltk.corpus.stopwords.words('english')

        corpus = list(filter(lambda sentence: not any(token in sentence for token in stopwords), corpus))


    # count the top MAX_VOCAB_LENGTH most used words in the corpus and and use those as our model's vocabulary
    token_counts = Counter([ token for sentence in corpus for token in sentence  ])
    vocab = [ token for token, _ in token_counts.most_common(MAX_VOCAB_LENGHT) ]
    vocab = list(sorted(vocab))

    # filter out any sentencese that do not contain the top MAX_VOCAB_LENGTH most used words
    corpus = list(filter(lambda sentence: all(token in vocab for token in sentence), corpus))

    # getting data
    #   if context_length = (n-1, 0), then they are called n-grams
    #   n denotes the length of the entire phrase, i.e. total context length + target length
    #   and the target length is almost always 1
    n = context_length[0] + 1 + context_length[1]
    contexts = []
    targets = []
    for sentence in corpus:
        
        # skipping sentences that are too short
        if len(sentence) < n:
            continue
        
        for i in range(len(sentence)-(n-1)):
            before = sentence[i:i+context_length[0]]
            target = sentence[i+context_length[0]]
            after = sentence[i+context_length[0]+1:i+n]
            
            contexts.append( before + after )
            targets.append( target )
    
    # integer encoding inputs and targets so torch.tensors can store their values
    contexts = list(map(lambda context: [word_to_embedding(vocab, word) for word in context], contexts))
    targets = [word_to_embedding(vocab, word) for word in targets]

    # train/test split
    splits = train_test_split(contexts, targets, test_size=0.1, random_state=seed)
    train_contexts, valid_contexts, train_targets, valid_targets = splits
    
    # creating and saving data to pickle file
    data_obj = {
        "vocab": vocab,
        "context_length": context_length,
        "train_contexts": train_contexts, "train_targets": train_targets,
        "valid_contexts": valid_contexts, "valid_targets": valid_targets
    }
    pickle.dump(data_obj, open("../data/data.pk", 'wb'))



def load_data(batch_size=None, preprocess=True, context_length=None, seed=None, 
                lemmatize=False, stem=False, remove_stopwords=True, library="spacy", **kwargs):
    
    if preprocess:
        
        if context_length is None:
            raise ValueError("Argument 'context_length' is None")
        
        if lemmatize and stem:
            raise ValueError("Arguments 'lemmatize' and 'stem' can't both be 'True'")

        if not library in ["spacy", "nltk"]:
            raise ValueError(f"Argument 'library' expected either 'spacy' or 'nltk', but got '{libary}''")
        
        preprocess_data(context_length=context_length, seed=seed, 
                        lemmatize=lemmatize, stem=stem, remove_stopwords=remove_stopwords, library=library)
        

    # loading data object
    data_obj = pickle.load(open("../data/data.pk", 'rb'))
    vocab = data_obj['vocab']
    train_contexts, train_targets = data_obj['train_contexts'], data_obj['train_targets']
    valid_contexts, valid_targets = data_obj['valid_contexts'], data_obj['valid_targets']

    # error checking
    if (not context_length is None) and data_obj['context_length'] != context_length:
        raise ValueError(f"Dataset has context_length = {data_obj['context_length']} from file, but you asked for {context_length}")

    #print("Total data:", len(train_targets) + len(valid_targets))

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