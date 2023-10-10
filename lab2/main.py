import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn import tree

def impute(dataset):
    imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
    imputer.fit(dataset)
    return imputer.transform(dataset)


def main():
    data = pd.read_csv('heart_data.csv')
    data = data.replace('?', np.NaN)
    print(data)

    predictors = ['age','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
    outcome = ['goal']
    xx = data.loc[1:, predictors]
    yy = data.loc[1:, outcome]
    X = impute(xx)
    y = impute(yy)

    # разбиение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    best_acc=0
    for i in range(15,25,2):

        clf = DecisionTreeClassifier(max_depth=i-5, max_leaf_nodes=i)
        clf.fit(X_train,y_train)
        
        prediction = clf.predict(X_train)
        acc = accuracy_score(y_train, prediction)
        print("deph = ", i-5, "leaves = ",i,", Accuracy: {:.4%}".format(acc))
        if best_acc<acc:
            best_acc=acc
            best_index= i
            best_deph = i-5
    print("depth = ",best_deph,", n_leaves = ",best_index,", Best Accuracy: {:.4%}".format(best_acc))
    # plt.figure(figsize=(12,8))
    # tree.plot_tree(clf, rounded=True, filled=True, feature_names=prediction)
    # plt.show()
    # plt.savefig('tree.png')
    # или
    # text_rep = tree.export_text(clf)
    # print(text_rep)

    # print("depth: ", clf.get_depth())
    # print("Leaves: ", clf.get_n_leaves())
    
if __name__ =="__main__":
    main()