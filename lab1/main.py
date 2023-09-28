#       2. Метод парзеновского окна с фиксированным h       #
#       2. T – треугольное K(x) = (1-r)[r<=1]               #

import numpy as np
import pandas as pd

def evklidR(x1,y1,x2,y2):
    h_dist = np.sqrt((y1-x1)^2 + (y2-x2)^2)
    return h_dist

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

def findCorAdv(XY_test, h_nowDist, longS):
    now_index = 0
    correct_adv = 0
    while now_index != (longS-1):
        class_0=0
        class_1=0
        for i in range(now_index, 0,-1):
            h_now = evklidR(XY_test[0][now_index],XY_test[1][now_index], XY_test[0][i-1],XY_test[1][i-1])
            if h_now<h_nowDist:
                if XY_test[2][i-1] == 1:
                    class_1+=1
                else:
                    class_0+=1

        for i in range(now_index, longS-1,1):
            h_now = evklidR(XY_test[0][now_index],XY_test[1][now_index], XY_test[0][i+1],XY_test[1][i+1])
            if h_now<h_nowDist:
                if XY_test[2][i+1] == 1:
                    class_1+=1
                else:
                    class_0+=1
            
        if class_1>class_0:
            predict = 1
        else:
            predict = 0
        if XY_test[2][now_index]==predict:
            correct_adv+=1
        now_index+=1
    return correct_adv

def main():
    df = pd.read_csv('data2.csv')
    longAll = len(df['Salary'])
    longS = int(longAll/3)
    XY_test, XY_train = makeTestTrainMas(df, longS, longAll)

    best_cor_adv_now = 0
    print("поиск лучшего h на тренировочных данных:")
    for h_Dist in range(1, 21, 3):
        correct_advForH = findCorAdv(XY_train, h_Dist, 2*longS)
   
        correct_advForH = correct_advForH/(2*longS) *100
        print("точность - ",correct_advForH, "% при h = ", h_Dist)

        if correct_advForH > best_cor_adv_now:
            best_cor_adv_now = correct_advForH
            h_best = h_Dist
    print("\nИтоги обучения:\n точность - ", best_cor_adv_now,"% \t h_best = ", h_Dist)

    correct_advForH = findCorAdv(XY_test, h_best, longS)
    correct_advForH = correct_advForH/longS *100
    print("\n точность обученой нейронки стоавляет - ", correct_advForH,"% при h = ", h_best)

if __name__ == '__main__':
    main()
