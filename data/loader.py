from dataclasses import dataclass

import numpy as np
import pandas as pd

from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

import os

if os.name != 'nt':
    from graph.graph import Graph

    def load_enron()->Graph:
        return Graph("./data/enron/enron_ed_ebc_edgesba.graphml", name="Enron", abbr="enron")

    def load_dblp()->Graph:
        return Graph("./data/dblp/dblp_ed_ebc.graphml", name="DBLP", abbr="dblp")

    def load_github()->Graph:
        return Graph("./data/github/github_ed_ebc_edgesba.graphml", name="Github", abbr="github")


NUMERICAL_ATT = 2
NOMINAL_ATT = 3

@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    n_classes: int
    n_attributes: int
    attr_types: list[int]
    n_nominal_values: dict[int, int]


def split_dataset(dataset: Dataset, test_size: float = 0.33, random_state: int = 0) -> tuple[Dataset, Dataset]:

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.X, dataset.y, test_size=test_size, random_state=random_state
    )
    return Dataset(X_train, y_train, dataset.n_classes, dataset.n_attributes, dataset.attr_types, dataset.n_nominal_values), \
            Dataset(X_test, y_test, dataset.n_classes, dataset.n_attributes, dataset.attr_types, dataset.n_nominal_values)


def load_pimas() -> Dataset:
    df = pd.read_csv("./data/tabular/diabetes.csv")
    for column in df.columns[:-1]:
        df[column] = (df[column] - df[column].min()) / (
            df[column].max() - df[column].min()
        )
    X = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy(dtype=int)
    n_classes = len(np.unique(y))
    n_attributes = X.shape[1]
    dataset = Dataset(X, y, n_classes, n_attributes, [NUMERICAL_ATT] * n_attributes, {})
    return dataset


def load_credit() -> Dataset:
    df = pd.read_csv("./data/tabular/credit.data.nonan.csv", header=None)
    attr_types = []
    n_nominal_values = {}
    for column in df.columns:
        if is_numeric_dtype(df[column]):
            df[column] = (df[column] - df[column].min()) / (
                df[column].max() - df[column].min()
            )
            attr_types.append(NUMERICAL_ATT)
        else:
            # transform unique values to int
            df[column] = df[column].astype("category").cat.codes
            unique_values = df[column].unique()
            n_nominal_values[column] = len(unique_values)
            attr_types.append(NOMINAL_ATT)

    X = df.iloc[:, :-1].to_numpy(dtype=np.float64)
    y = df.iloc[:, -1].to_numpy(dtype=int)
    n_classes = len(np.unique(y))
    n_attributes = X.shape[1]
    dataset = Dataset(X, y, n_classes, n_attributes, attr_types, n_nominal_values)
    return dataset

def load_nltcs():
    X = np.loadtxt('./data/tabular/nltcs.dat', dtype = int)
    i = 13
    y = X[:,i]
    X = np.delete(X, i, axis=1)
    n_classes = len(np.unique(y))
    n_attributes = X.shape[1]
    n_nominal_values = {i:2 for i in range(n_attributes)}
    dataset = Dataset(X, y, n_classes, n_attributes, [NOMINAL_ATT] * n_attributes, n_nominal_values)
    return dataset

def load_acs():
    X = np.loadtxt('./data/tabular/acs.dat', dtype = int)
    i = -13
    y = X[:,i]
    X = np.delete(X, i, axis=1)
    n_classes = len(np.unique(y))
    n_attributes = X.shape[1]
    n_nominal_values = {i:2 for i in range(n_attributes)}
    dataset = Dataset(X, y, n_classes, n_attributes, [NOMINAL_ATT] * n_attributes, n_nominal_values)
    return dataset

def load_adult():
    data = pd.read_csv("./data/tabular/adult_full.csv")
    data = data[data["workclass"] != "?"]
    data = data[data["occupation"] != "?"]
    data = data[data["native-country"] != "?"]
    data.replace(['Divorced', 'Married-AF-spouse', 
              'Married-civ-spouse', 'Married-spouse-absent', 
              'Never-married','Separated','Widowed'],
             ['not married','married','married','married',
              'not married','not married','not married'], inplace = True)

    category_col =['workclass', 'race', 'education','marital-status', 'occupation',
               'relationship', 'gender', 'native-country', 'income'] 

    for col in category_col:
        _, c = np.unique(data[col], return_inverse=True) 
        data[col] = c
    
    predictors = ['age','workclass','educational-num',
              'marital-status', 'occupation','relationship','race','gender',
              'capital-gain','capital-loss','hours-per-week', 'native-country']

    y = data["income"]
    X = data[predictors]

    cont_atts = np.array([0,8,9,10])
    X = X.astype(np.float64)
    for i in cont_atts:
        X.iloc[:,i] = (X.iloc[:,i] - X.iloc[:,i].min()) / (X.iloc[:,i].max() - X.iloc[:,i].min())

    X = X.to_numpy()
    y = y.to_numpy()

    n_classes = len(np.unique(y))
    n_attributes = X.shape[1]

    attr_types = [NOMINAL_ATT] * n_attributes
    for i in cont_atts:
        attr_types[i] = NUMERICAL_ATT

    n_nominal_values = {}
    for i in range(n_attributes):
        if attr_types[i] == NOMINAL_ATT:
            X[:,i] = X[:,i] - X[:,i].min()
            n_nominal_values[i] = len(np.unique(X[:,i]))
    
    return Dataset(X, y, n_classes, n_attributes, attr_types, n_nominal_values)

if __name__ == "__main__":
    # dataset = load_credit()
    # dataset = load_enron()
    # dataset = load_nltcs()
    # dataset = load_acs()
    dataset = load_adult()

    print()
