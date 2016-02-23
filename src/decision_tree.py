import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import random
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import random



#Constants
# csv_path = "/Users/zverham/Documents/School/Masters/SemesterTwo/SYS6016/sys6016_project1/data/titanic_preprocessed.csv"
csv_path = "~/Desktop/UVA Spring 2016/Machine Learning | SYS 6016/HW/Titanic Project/titanic_preprocessed.csv"

#features from our preprocessed dataset we want to consider
feature_names = ["pclass", "survived", "sex", "sibsp", "parch", "age.bin", "fare"]

#load data, split into attributes / classifications
titanic = pd.read_csv(csv_path)
X = titanic[feature_names]
y = titanic.embarked

accuracies = []

#for j in range(0, 3):
    
#    X_train = None
#    X_test = None
#    y_train = None
#    y_test = None
    
# We, then, split the data into testing and training sets, using 30% of the data for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

#scaling doesn't seem to do anything...need to confirm this.
X_train_std = X_train
X_test_std = X_test

#Here, we define the parameters of our tree
clf = DecisionTreeClassifier(criterion = 'entropy', max_depth=18, random_state=0)

# We then fit the tree to our training data 
clf.fit(X_train_std, y_train.astype(int))

# Let's make a prediction
y_pred=clf.predict(X_test_std)

# Now we calculate our accuracy and create a confusion matrix of our results
print('Accuracy: %.2f' % accuracy_score(y_test,y_pred))
confmat=confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)
    
    #accuracies.append(accuracy_score(y_test,y_pred))

#print('Mean accuracy: %.2f' % (sum(accuracies)/len(accuracies)))

# Now we visualize our tree
export_graphviz(clf, out_file='titanic.dot',feature_names=feature_names)
