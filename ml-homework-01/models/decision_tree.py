import math

import pandas as pd

"""
Author: Wang Jianghan<ssYanhuo@foxmail.com>
Decision Tree Models using ID3, C.5 and CART methods.
"""


class Node:
    def __init__(self, feature='', judgements=None, is_leaf=False, label=''):
        if judgements is None:
            judgements = dict()
        self.feature = feature
        self.judgements = judgements.copy()
        self.is_leaf = is_leaf
        self.label = label

    def as_leaf(self, label):
        self.is_leaf = True
        self.label = label

    def add_judgement(self, value):
        child = Node()
        self.judgements[value] = child
        return child

    def add_leaf(self, value, label):
        child = Node()
        child.is_leaf = True
        child.label = label
        self.judgements[value] = child
        return child

    def set_feature(self, feature):
        self.feature = feature

    def match(self, data):
        return self.judgements[data[self.feature]]


class DecisionTree:
    def __init__(self, x_features, y_label):
        self.root = Node()
        self.x = x_features
        self.y = y_label

    def get_root(self):
        return self.root


def get_most_label(data: pd.DataFrame, y: str):
    return data[y].mode()[0]


def check_feature_consistency(data: pd.DataFrame, features: list):
    for feature in features:
        if len(data[feature].unique()) > 1:
            return False
    return True


def entropy(data: pd.DataFrame, feature: str or None, value: str or None, y: str):
    if feature is not None and value is not None:
        data = data[data[feature] == value]
    y_values = data[y].unique()
    ent = 0
    for y_value in y_values:
        p = data[data[y] == y_value].shape[0] / data.shape[0]
        ent -= p * math.log2(p)
    return ent


def entropy_gain(data: pd.DataFrame, feature: str, y: str):
    gain = entropy(data, None, None, y)
    for value in data[feature].unique():
        gain -= data[data[feature] == value].shape[0] / data.shape[0] * entropy(data, feature, value, y)
    return gain


def id3(d, a, y):
    gain_max = 0
    best_feature = ''
    for feature in a:
        gain = entropy_gain(d, feature, y)
        if gain >= gain_max:
            gain_max = gain
            best_feature = feature
    return best_feature


def iv(data: pd.DataFrame, feature: str):
    iva = 0
    for value in data[feature].unique():
        p = data[data[feature] == value].shape[0] / data.shape[0]
        iva -= p * math.log2(p)
    return iva


def c45(d, a, y):
    gain_ratio_max = 0
    best_feature = ''
    for feature in a:
        if iv(d, feature) == 0:
            return feature
        gain_ratio = entropy_gain(d, feature, y) / iv(d, feature)
        if gain_ratio >= gain_ratio_max:
            gain_ratio_max = gain_ratio
            best_feature = feature
    return best_feature


def gini(data: pd.DataFrame, feature: str or None, value: str or None, y: str):
    data = data[data[feature] == value]
    gin = 1
    for y_value in data[y].unique():
        gin -= (data[data[y] == y_value].shape[0] / data[y].shape[0]) ** 2

    return gin


def gini_index(data: pd.DataFrame, feature: str, y: str):
    gi = 0
    for value in data[feature].unique():
        gi += data[data[feature] == value].shape[0] / data.shape[0] * gini(data, feature, value, y)
    return gi


def cart(d, a, y):
    gii_min = math.inf
    best_feature = ''
    for feature in a:
        gii = gini_index(d, feature, y)
        if gii <= gii_min:
            gii_min = gii
            best_feature = feature
    return best_feature


class BaseDecisionTree:
    def __init__(self, method):
        self.data = None
        self.x_features = None
        self.y_label = None
        self.method = method
        self.tree = None
        self.origin_data = None

    def generate_tree(self, d: pd.DataFrame, a: list, tree: DecisionTree, node: Node):
        if tree is None:
            tree = DecisionTree(self.x_features, self.y_label)
            node = tree.get_root()

        if len(d[self.y_label].unique()) <= 1:
            node.as_leaf(get_most_label(d, self.y_label))
            return

        if len(a) == 0 or check_feature_consistency(d, a):
            node.as_leaf(get_most_label(d, self.y_label))
            return

        best_feature = self.method(d, a, self.y_label)

        node.set_feature(best_feature)
        for value in self.origin_data[best_feature].unique():
            if d[d[best_feature] == value].shape[0] <= 0:
                node.add_leaf(value, d[self.y_label].mode()[0])
                continue
            child = node.add_judgement(value)
            next_a = a.copy()
            next_a.remove(best_feature)
            self.generate_tree(d[d[best_feature] == value], next_a, tree, child)

        return tree

    def test_tree(self, data, node: Node):
        if node.is_leaf:
            return node.label
        return self.test_tree(data, node.match(data))

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series):
        self.data = pd.concat([x_train, y_train], axis=1)
        self.x_features = x_train.columns.to_list()
        self.y_label = y_train.name
        self.tree = DecisionTree(self.x_features, self.y_label)
        self.origin_data = self.data.copy()
        self.generate_tree(self.data, self.x_features, self.tree, self.tree.get_root())

    def predict(self, data: pd.DataFrame):
        result = []
        for _, row in data.iterrows():
            result.append(self.test_tree(row, self.tree.get_root()))

        return result


class ID3DecisionTree(BaseDecisionTree):
    def __init__(self):
        super().__init__(id3)


class C45DecisionTree(BaseDecisionTree):
    def __init__(self):
        super().__init__(c45)


class CARTDecisionTree(BaseDecisionTree):
    def __init__(self):
        super().__init__(cart)
