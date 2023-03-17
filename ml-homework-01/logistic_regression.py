import pandas as pd

from models.logistic_regression import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/logistic_regression.csv')
x_data = data.iloc[:, 1: 31]
y_data = data.iloc[:, 31]
seed = 202015005
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=seed)


model = LogisticRegression()
model.fit(x_train, y_train)
pred = model.predict(x_test)
print(accuracy_score(y_test, pred))
