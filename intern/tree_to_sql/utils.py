import json
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return np.float32(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def convert_tree_to_json(tree: DecisionTreeClassifier) -> str:
    def rec(tree, node_index=0):
        if (tree.tree_.children_left[node_index] == -1
            and tree.tree_.children_right[node_index] == -1):
                class_label = int(tree.tree_.value[node_index].argmax())
                return {"Class Label:": class_label}

        feature_index = int(tree.tree_.feature[node_index])
        threshold = float(tree.tree_.threshold[node_index])
        left_child_index = tree.tree_.children_left[node_index]
        right_child_index = tree.tree_.children_right[node_index]

        return {"feature_index": feature_index,
                "threshold": np.round(threshold, 4),
                "left": rec(tree, left_child_index),
                "right": rec(tree, right_child_index)
                }

    tree_as_dict = rec(tree)
    tree_as_json = json.dumps(tree_as_dict, cls=NpEncoder)

    return tree_as_json