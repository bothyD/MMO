# Использовать модель LARS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lars
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

def impute(dataset):
    imputer = SimpleImputer(missing_values=np.NaN, strategy='mean')
    imputer.fit(dataset)
    return imputer.transform(dataset)

def make_tagertFeatures(data, target, features):
    targetAll = data.loc[1:, target]
    features = data.loc[1:, features]
    targetAll = impute(targetAll)
    featuresAll = impute(features)
    return targetAll, featuresAll

def checkProcenTrue(y_test, y_pred):
    count_right=0
    for i in range(len(y_test)):
        if abs(y_test[i]-y_pred[i])<=0.8:
            count_right +=1
    print(f"Процент верных предсказаний - ", count_right/len(y_test)* 100, "%")

def makeGraphic(predicted_values, real_values ):
    residuals = real_values - predicted_values
    plt.scatter(predicted_values, residuals)
    plt.xlabel('Предсказанные значения')
    plt.ylabel('Остатки')
    plt.title('График остатков')
    plt.axhline(y=0, color='r', linestyle='--') # Горизонтальная линия на нулевом уровне
    plt.show()

def makeLarsModel(features, target, strinName):
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features_normalized, target, test_size=0.3, random_state=196)
    model = Lars()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    mean_score = np.mean(score)
    print(f"\tДля {strinName}:\t")
    print(f"Среднее качесвто регрессии - ", mean_score)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'Средняя абсолютная ошибка (MAE): {mae}')
    checkProcenTrue(y_test, y_pred)

    y_graph = []
    for i in range(len(y_test)):
        y_graph.append(y_test[i][0])
    # makeGraphic(y_pred, y_graph)

def main(): 
    data = pd.read_csv('winequalityN.csv')
    column_features = data.columns[:12].tolist()
    column_target = data.columns[12:13].tolist()
    data['type'].replace({'red':1, 'white':0}, inplace=True)

    data_all = data.replace('?', np.NaN)
    data_white = data_all[data_all['type'] == 0]  
    data_red = data_all[data_all['type'] == 1]  
    # для всех 
    targetAll, featuresAll = make_tagertFeatures(data_all, column_target, column_features)

    makeLarsModel(featuresAll, targetAll, "всех вин")
    # для RED 
    targetRed, featuresRed = make_tagertFeatures(data_red, column_target, column_features)
    makeLarsModel(featuresRed, targetRed, "красных вин")
    # для WHITE
    targetWhite, featuresWhite = make_tagertFeatures(data_white, column_target, column_features)
    makeLarsModel(featuresWhite, targetWhite, "белых вин")
    
if __name__=="__main__":
    main()