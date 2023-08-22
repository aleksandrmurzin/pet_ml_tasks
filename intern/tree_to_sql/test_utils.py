from utils import convert_tree_to_json, generate_sql_query
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

X, y = make_classification(n_samples=100, n_features=5, n_informative=2, n_classes=2)
X = pd.DataFrame(X, columns=["a","b","c", "d", "e"])
features = X.columns.to_list()
clf = DecisionTreeClassifier()
clf.fit(X, y)

# def test_convert_tree_to_json():
#     clf = DecisionTreeClassifier()
#     clf.fit(X, y)
#     tree_as_json = convert_tree_to_json(clf)



def test_generate_sql_query():
    tree_as_json = convert_tree_to_json(clf)
    sql_query = generate_sql_query(tree_as_json, features)