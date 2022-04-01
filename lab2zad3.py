import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("iris.csv")

all_inputs = df[['sepal.length', 'sepal.width', 'petal.length', 'petal.width']].values
all_classes = df['variety'].values

(train_inputs, test_inputs, train_classes, test_classes) = train_test_split(all_inputs, all_classes, train_size=0.7, random_state=51254)

print(train_inputs)
print(test_inputs)

dtc = DecisionTreeClassifier()
dtc = dtc.fit(train_inputs, train_classes)
dtc = dtc.score(test_inputs, test_classes)

print(dtc * 100)