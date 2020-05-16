import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

# Ignore warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AdultDataset(Dataset):

    def __init__(self, df, label_feature, label_mapping):
        
        self.label_feature = label_feature
        self.label_mapping = label_mapping

        # separating data and labels
        self.data = df.loc[:, df.columns != label_feature]
        self.labels = df[label_feature]
        
        # we need to integer encode the labels using the manual mapping
        self.labels = self.labels.transform(lambda value: label_mapping[value])

        # converting to tensors
        self.data = torch.tensor( self.data.values )
        self.labels = torch.tensor( self.labels.values )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, :], self.labels[index]     


def clean(df):
    """ 
    TODO: Removes rows with bad data
        param: pd.DataFrame
        return: pd.DataFrame
    """
    # Note: "bad data" is data in the dataset that have entries '?'
    raise NotImplementedError

def balance(df, label_feature, seed=None):
    """ 
    TODO: Remove rows such that there is an equal number of rows with each feature_label value
        param: pd.DataFrame, str, int
        return: pd.DataFrame
    """
    # Steps: 
    # 1) Get set of all values in in the column "label_feature"
    #    (for the label "incomes", these values will be "<= 50K" and "> 50K")
    # 2) count the number of times each value appears in the column and get the minimum such value
    # 3) iterate through all data with each value, and cut the number down to this minimum value
    raise NotImplementedError

def get_Metadata(df_train, df_valid):
    # needed for later
    df_full = df_train.append( df_valid )
    
    # creating metadata dictionary
    continuous_feats = [ feature for feature in df_full.columns if df_full[feature].dtype in [int, float] ]
    categorical_feats = [ feature for feature in df_full.columns if df_full[feature].dtype in ["object", bool] ]
    
    Metadata = { feature : {"type": "continuous"} for feature in continuous_feats }
    Metadata.update({ feature : {"type": "categorical"} for feature in categorical_feats })

    # Saving metadata of train data. This is important, we don't do this for the validation data
    # we need to pretend as if all our model knows is the training data. So we normalize and encode
    # the validation data w.r.t. the values obtained from the training data
    for feature in continuous_feats:
        Metadata[feature]["mean"] = df_train[feature].mean()
        Metadata[feature]["std"] = df_train[feature].std()
    
    # because the data is sparse in some categories, we need to make sure we get every possible value
    # so our label encoder doesn't miss anything. Therefore we use df_full here
    label_encoder = LabelEncoder()
    for feature in categorical_feats:
        label_encoder.fit_transform(df_full[feature])
        Metadata[feature]["mapping"] = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))

    return Metadata

def transform(df, label_feature, Metadata):
    # we do not transform the label feature. We deal with that separately than the input data
    continuous_feats = [feature for feature in Metadata if feature != label_feature and Metadata[feature]["type"] == "continuous"]
    categorical_feats = [feature for feature in Metadata if feature != label_feature and Metadata[feature]["type"] == "categorical"]
    
    # normalizing continuous data using statistics in Metadata
    df_cont = df[continuous_feats].apply( lambda col: (col -  Metadata[col.name]["mean"]) / Metadata[col.name]["std"] )

    # label encode using Mmapping in Metadata
    df_cat = df[categorical_feats].apply( lambda col: col.transform( lambda value: Metadata[col.name]["mapping"][value]) )
    
    # Now, it might be the case that the validation data might not take on every possible value in each feature
    # and unfortunately, the onehot-encoder doesn't know how to deal with that, so we need to inject some dumby rows
    # I'm sorry this code is very messy...just don't look at it
    num_dumby_rows = 0
    max_index = max(df_cat.index.values)
    for feature in categorical_feats:
        all_values = set(Metadata[feature]["mapping"].values())
        present_values = set(df_cat[feature].values)
        missing_values = all_values - present_values
        if len(missing_values) != 0:
            dumby_row = np.zeros(len(categorical_feats))
            for value in missing_values:
                dumby_row[categorical_feats.index(feature)] = value
                index = max_index + num_dumby_rows + 1
                dumby_df = pd.DataFrame([dumby_row], columns=categorical_feats, index=[index])
                df_cat = df_cat.append(dumby_df)
                num_dumby_rows += 1

    # onehot encode data from label encoding
    onehot_encoder = OneHotEncoder(categories='auto')
    np_cat = onehot_encoder.fit_transform(df_cat).toarray()

    # removing dumby rows
    if num_dumby_rows != 0:
        np_cat = np_cat[:-num_dumby_rows, :]

    # putting the data back into a pandas dataframe
    header = [f"{feature}_{value}" for feature in categorical_feats for value in Metadata[feature]["mapping"]]
    df_cat_expanded = pd.DataFrame(np_cat, columns=header, index=df.index)

    # concatingating everything into one dataframe
    return pd.concat([df_cont, df_cat_expanded, df[label_feature]], axis=1)


def load_data(data_path, label_feature, label_mapping, preprocess=True, batch_size=None, seed=None):
    
    # getting dataset
    df = pd.read_csv(data_path)
    
    if preprocess:
        # cleaning removes bad or incomplete data
        print("Cleaning dataset...")
        df = clean(df)
        
        # we don't need to balance continuous data
        if not label_mapping is None:
            # balancing ensures there are an equal number of data points for each value in label_feature
            # which prevents our model from becoming biased
            print("Balancing dataset...")
            df = balance(df, label_feature, seed)

    # Splitting Data
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=seed)

    if preprocess:
        # getting the metadata of our data, which will help us in the transform function
        Metadata = get_Metadata(df_train, df_valid)
        
        # we need to normalize continuous data and one-hot encode categorical data
        print("Transforming dataset...")
        df_train = transform(df_train, label_feature, Metadata)
        df_valid = transform(df_valid, label_feature, Metadata)

        # writing to new csv
        df = df_train.append(df_valid)
        df = df.sort_index(axis=0)
        metadata_path = f"{data_path[:-4]}_preprocessed.csv"
        print(f"Saving preprocessed dataset to {metadata_path}")
        df.to_csv(metadata_path, index=False)
    
    train_dataset = AdultDataset(df_train, label_feature, label_mapping)
    batch_size = len(train_dataset) if batch_size is None else batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = AdultDataset(df_valid, label_feature, label_mapping)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))

    return train_loader, valid_loader