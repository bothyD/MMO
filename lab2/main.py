import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def makeTestTrainMas(df, longS, longAll):
    X_test = df['MrotInHour'][0:longS].values
    Y_test = df['Salary'][0:longS].values
    class_test = df['Class'][0:longS].values
    

    X_train = df['MrotInHour'][longS:longAll].values
    Y_train = df['Salary'][longS:longAll].values
    class_train = df['Class'][longS:longAll].values
    

    XY_test = np.vstack((X_test, Y_test,class_test))
    XY_train = np.vstack((X_train, Y_train,class_train))

    return XY_test, XY_train

X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = DecisionTreeClassifier()
clf.fit(X, y)
#???????????????????????????????simpleInputer

def main():
    df = pd.read_csv('heart_data.csv')
    longAll = len(df['age'])
    longS = int(longAll/3)
    XY_test, XY_train = makeTestTrainMas(df, longS, longAll)

    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 0])
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

if __name__ =="__main__":
    main()