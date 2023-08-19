from utils import convert_tree_to_json
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
# import pandas as pd

X, y = make_classification(n_samples=100, n_features=5, n_informative=2, n_classes=2)

def test_convert_tree_to_json():
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    tree_as_json = convert_tree_to_json(clf)
