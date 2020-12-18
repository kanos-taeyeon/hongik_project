import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import core
#%matplotlib inline
import pickle
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], iris.target, test_size=0.25, stratify=iris.target, random_state=123456)


modelpath="/home/server1/finalized_model.sav"
with open(modelpath, 'rb') as f:
	model = pickle.load(open(modelpath, 'rb'))

print(X_test)
print(X_train)
print(y_test)
print(model)


#predicted = model.predict(X_test)
#accuracy = accuracy_score(y_test, predicted)
