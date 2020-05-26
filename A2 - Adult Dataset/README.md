# Adult Income Dataset

We will be using the [Adult Income Dataset](https://www.cs.toronto.edu/~delve/data/adult/adultDetail.html) to attempt to classify a person’s income based on several other features:
* **Categorical Features**
  * education
  * gender
  * marital-status
  * native-country
  * occupation
  * race
  * relationship
  * workclass

* **Continuous Features**
  * age
  * capital-gain
  * capital-loss
  * educational-num
  * fnlwgt
  * hours-per-week

A person's income is a categorical feature that can be either "<= 50K" or "> 50K". This is the feature we are trying to predict.

## Preprocessing
Before we can do any classification, we need to **preprocess** the data, which is the one-time transformation of the data into a form that our model can use. Preprocessing involves the following steps:
  1. **Cleaning the Data**: remove missing or corrupt values
  2. **Balance the Data**: ensure there are an equal number of samples with income labels "<= 50K" and "> 50K"
  3. **Normalize** continuous features to have a mean of 0 and standard deviation of 1
  4. **Onehot encode** categorical features so all categorical inputs are binary

See L2 slides for graphics on normalizing and onehot encoding. The question you may have is, why do we need to do this? The answer is to prevent bias in our model.
  1. We need to clean the data since if the value of a feature is unknown, then it may bias our model towards the features where all values of known.
  2. We need to balance the data in order to prevent our model from being biased towards one income value. For example, consider a dataset with 90 "<= 50K" labels and 10 "> 50K" labels. The model would learn that if it always guesses "<= 50K", it will be correct 90% of the time. This is onbviously not desirable.
  3. Consider a model that takes in the amount of water a person consumes in a day in milliliters and the height of a person in meters. The former will be in the thousands and the latter will be somewhere between 1 and 2. Since all the weights of a model are usually initialized randomly between [-1, 1], the high values of the water input will dominate the values of the height values. Thus, the model will essentially disregard the information provided by the height data. In order to mitigate this problem, we normalize all data to have a mean of 0 and standard deviation of 1.
  4. If we were to just integer encode categorical data (for example, eye colors {"blue", "brown", "hazel"} --> {0, 1, 2}), then our data would suffer the same problem as in un-normalized data. The bigger values would slight bias the model. Further, the model would interpret these increasing numbers as a gradient between the feature values. In our example, the model would think that brown eye is between blue and hazel eyes. We don't want this, so the solution is to onehot encode the categorical data.
  
## Task

### Part 1
You will fill in some of the missing preprocessing code (found in `utils.py`). Specifically cleaning and balancing the data set. Normalizing and onehot encoding has been done for you.

```python
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
```

### Part 2
You will then create a simple MLP to predict a person’s income. For information on MLPs see L2 slides. 

```python
class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, output_size=1):
        super(MultiLayerPerceptron, self).__init__()
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError
```

You may implement a specific MLP in this class, or you can add arguments to the `__init__` so that the class can produce any arbitrary MLP. See the in the `Complete` directory for how to implement a general MLP

**Note**: the best non-neural networks can do is about 86% accuracy, so getting anything above an 80% validation accuracy is considered excellent.
