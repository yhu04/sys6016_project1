import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import KFold
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import random
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
import random

# csv_path = "/Users/zverham/Documents/School/Masters/SemesterTwo/SYS6016/sys6016_project1/data/titanic_preprocessed.csv"
csv_path = "~/Desktop/UVA Spring 2016/Machine Learning | SYS 6016/HW/Titanic Project/titanic_preprocessed.csv"

#features from our preprocessed dataset we want to consider
feature_names = ["pclass", "survived", "sex", "sibsp", "parch", "age.bin"]

#load data, split into attributes / classifications
titanic = pd.read_csv(csv_path)
X = titanic[feature_names]
y = titanic.embarked

#Here, we define the parameters of our tree
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth=18, random_state=0)

kf = KFold(len(X.index),5)

mean_accuracy = 0 
for train_index, test_index in kf:
	X_train, X_test = X[train_index], X[test_index]
	y_train, y_test = y[train_index], y[test_index]
	clf.fit(X_train,y_train)
	y_pred=clf.predict(X_test)
	mean_accuracy = accuracy_score(y_test,y_pred) + mean_accuracy
	fpr_RF,tpr_RF,thresholds_RF = roc_curve(y_test,y_pred)
	roc_auc_RF = auc(fpr_RF,tpr_RF)
	plt.plot(fpr_RF, tpr_RF, lw=1, label='ROC-tree (area = %0.3f)' % (roc_auc_RF))
mean_accuracy = mean_accuracy/5
print mean_accuracy

plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig('ROC-Decision Tree.pdf')
