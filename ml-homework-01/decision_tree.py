import pandas as pd

from models.decision_tree import ID3DecisionTree, C45DecisionTree, CARTDecisionTree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/decision_tree.csv')
x_data = data.iloc[:, 1: 15]
y_data = data.iloc[:, 15]
seed = 202015005
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=seed)


model_id3 = ID3DecisionTree()
model_id3.fit(x_train, y_train)
pred_id3 = model_id3.predict(x_test)
print("ID3:", accuracy_score(y_test, pred_id3))

model_c45 = C45DecisionTree()
model_c45.fit(x_train, y_train)
pred_c45 = model_c45.predict(x_test)
print("C4.5:", accuracy_score(y_test, pred_c45))

model_cart = CARTDecisionTree()
model_cart.fit(x_train, y_train)
pred_cart = model_cart.predict(x_test)
print("CART:", accuracy_score(y_test, pred_cart))
