import json
from sklearn.tree import DecisionTreeClassifier


def convert_tree_to_json(tree: DecisionTreeClassifier) -> str:
    """_summary_

    Args:
        tree (DecisionTreeClassifier): _description_

    Returns:
        str: _description_
    """
    def rec(tree, node_index=0):
        """_summary_

        Args:
            tree (_type_): _description_
            node_index (int, optional): _description_. Defaults to 0.

        Returns:
            _type_: _description_
        """
        if (tree.tree_.children_left[node_index] == -1
            and tree.tree_.children_right[node_index] == -1):
            class_label = int(tree.tree_.value[node_index].argmax())
            return {"class": class_label}

        feature_index = int(tree.tree_.feature[node_index])
        threshold = float(tree.tree_.threshold[node_index])
        left_child_index = tree.tree_.children_left[node_index]
        right_child_index = tree.tree_.children_right[node_index]

        return {"feature_index": feature_index,
                "threshold": round(threshold, 4),
                "left": rec(tree, left_child_index),
                "right": rec(tree, right_child_index),
                }

    tree_as_dict = rec(tree)
    tree_as_json = json.dumps(tree_as_dict, indent=" ")

    return tree_as_json


def generate_sql_query(tree_as_json: str, features: list) -> str:
    """_summary_

    Args:
        tree_as_json (str): _description_
        features (list): _description_

    Returns:
        str: _description_
    """
    tree_as_dict = json.loads(tree_as_json)

    def rec(node, n=1):
        if "class" in node:
            return f"{node['class']}"
        feature_name = features[node["feature_index"]]
        return f"CASE\n\tWHEN {feature_name} > {node['threshold']} THEN {rec(node['right'])}\n\tELSE {rec(node['left'])} END"

    sql_query = rec(tree_as_dict)
    return f"SELECT \n {sql_query} AS CLASS_LABEL"
