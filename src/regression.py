import sys
import matplotlib.pyplot as plt
from scipy.linalg import inv
from sklearn.datasets import make_regression, make_classification
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import json
from pandas.io.json import json_normalize
import os
import datetime as dt

sys.path.insert(1, '..\src')
import preprocessing

kwargs = dict(random_state=42)

def linearRegression(df_singlePeriod):
    X=df_singlePeriod.index.values
    Y=df_singlePeriod['inter_pol'].values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.00001 , shuffle=True, **kwargs)
    X= X.reshape(-1, 1)
    X_train= X_train.reshape(-1, 1)
    y_train= y_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    plt.xticks(fontsize=8, rotation=90)
    plt.title("Data plot for single period")
    plt.scatter(X_train, y_train)
    plt.show()
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    plt.scatter(X_train, y_train)
    plt.plot(X, lr.predict(X), color="green")
    plt.scatter(X_test, y_test, color="red")
    for i in range(len(X_test)):
        plt.plot([X_test[i]]*2, [y_test[i], y_pred[i]], color="red")
    
    # calculate mean squared error and mean absolute error
    for name, metric in zip(("MSE", "MAE"), (mean_squared_error, mean_absolute_error)):
        print(f"{name}: {metric(y_test, y_pred)}")
    
    print(y_pred)
    print(lr.coef_)


'''Funktionen die mithilfe der Steigungsfunktion, versuchen die orginalen Inputdaten zu approximieren. Nehmen einen preprocessden DataFrame
der auf Tagesniveau gruppiert wurde und geben die Predictens als Liste zurück
Dabei wird der Füllstand für die Prediction = mx + b - wobei b der Füllstand nach der vorherigen Leerung ist.'''
    
def predictValues_general(test, temp_list, temp):
    counter = 1
    pred_list = list()
    # pred_list.append(temp_list[0])
    for k in range(test[0]): 
        y = -3.62261628 * k + temp_list[0]
        pred_list.append(y) 
    for timeinterval in test:
        if counter < len(test):
            length = test[counter] - timeinterval
            #print(test[counter+1])
            #pred_list.append(temp_list[timeinterval])
            counter +=1    
            for i in range(length):
               #print(temp_list[timeinterval])
                y = -3.62261628 * i + temp_list[timeinterval]
                pred_list.append(y)
            addition =  temp.shape[0] - test[-1]
    for j in range(addition): 
        y = -3.62261628 * j + temp_list[test[-1]]
        pred_list.append(y)    
    temp['linearReg'] = pred_list
    rmse = mean_squared_error(temp_list, pred_list, squared=False)
    print("Root Squared Mean Error: " + str(rmse))
    print("Values have been predicted!")


def predictValues_clust0(input):
    counter = 1
    pred_list = list()
    temp_list = input['inter_pol'].tolist()

    emptie_checkpoints =  preprocessing.calculate_empties_adaptive(input, 1.015)

    for k in range(emptie_checkpoints[0]):
        y = -3.62261628 * k + temp_list[0]
        pred_list.append(y)
    
    for timeinterval in emptie_checkpoints:
        if counter < len(emptie_checkpoints):
            length = emptie_checkpoints[counter] - timeinterval
            counter = counter + 1

            for i in range(length):
                y = -3.62261628 * i + temp_list[timeinterval]
                pred_list.append(y)
            
            addition = input.shape[0] - emptie_checkpoints[-1]
    
    for j in range(addition):
        y = -3.62261628 * temp_list[emptie_checkpoints[-1]]
        pred_list.append(y)
    
    print('values have been predicated')
    rmse = mean_squared_error(temp_list, pred_list, squared=False)
    print("Root Squared Mean Error:  " + str(rmse))
    return pred_list


def predictValues_clust1(input):
    df = input[['time_stamp', 'inter_pol']]
    counter = 1
    pred_list = list()
    temp_list = input['inter_pol'].tolist()

    emptie_checkpoints =  preprocessing.calculate_empties_adaptive(input, 1.015)

    for k in range(emptie_checkpoints[0]):
        y = -3.63041747 * k + temp_list[0]
        pred_list.append(y)
    
    for timeinterval in emptie_checkpoints:
        if counter < len(emptie_checkpoints):
            length = emptie_checkpoints[counter] - timeinterval
            counter = counter + 1

            for i in range(length):
                y = -3.62261628 * i + temp_list[timeinterval]
                pred_list.append(y)
            
            addition = input.shape[0] - emptie_checkpoints[-1]
    
    for j in range(addition):
        y = -3.62261628 * temp_list[emptie_checkpoints[-1]]
        pred_list.append(y)
    
    print('values have been predicated')
    rmse = mean_squared_error(temp_list, pred_list, squared=False)
    print("Root Squared Mean Error:  " + str(rmse))
    return pred_list


def predictValues_clust2(input):
    counter = 1
    pred_list = list()
    temp_list = input['inter_pol'].tolist()

    emptie_checkpoints =  preprocessing.calculate_empties_adaptive(input, 1.015)

    for k in range(emptie_checkpoints[0]):
        y = -3.62261628 * k + temp_list[0]
        pred_list.append(y)
    
    for timeinterval in emptie_checkpoints:
        if counter < len(emptie_checkpoints):
            length = emptie_checkpoints[counter] - timeinterval
            counter = counter + 1

            for i in range(length):
                y = -3.62261628 * i + temp_list[timeinterval]
                pred_list.append(y)
            
            addition = input.shape[0] - emptie_checkpoints[-1]
    
    for j in range(addition):
        y = -3.38830135 * temp_list[emptie_checkpoints[-1]]
        pred_list.append(y)
    
    print('values have been predicated')
    rmse = mean_squared_error(temp_list, pred_list, squared=False)
    print("Root Mean Squared Error" + str(rmse))
    return pred_list


def linearRegressionPlot(temp, test, cluster):
    plt.figure(figsize=(30,8))
    plt.ylim((0,200))
    plt.title(cluster)

    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=10, fontweight='bold')
    plt.plot(temp['inter_pol'])
    plt.plot(temp['linearReg'])
    plt.legend(['Raw', 'moving average', 'mov_avg_in'], loc='upper left')

    for i in test: 
        plt.vlines(i, color="green", ymin=0, ymax=200)
    plt.show()



def linearRegressionPlot_pred(input, cluster):
    if cluster < 0 or cluster > 4:
        print('Cluster has to be 0, 1, 2')
        return
    plt.figure(figsize=(30,8))
    plt.ylim((0,200))
    plt.title('Abweichung')

    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=10, fontweight='bold')
    plt.plot(input['inter_pol'])
    
    if cluster == 0:
        predicted = predictValues_clust0(input)
        empties = preprocessing.calculate_empties_adaptive(input, 1.015)
    elif cluster == 1:
        predicted = predictValues_clust1(input)
        empties = preprocessing.calculate_empties_adaptive(input, 1.015)
    else:
        predicted = predictValues_clust2(input)
        empties = preprocessing.calculate_empties_adaptive(input, 1.015)

    plt.plot(predicted)
    plt.legend(['Preprocessed', 'Predictions'], loc='upper left')

    for i in empties: 
        plt.vlines(i, color="green", ymin=0, ymax=200)
    plt.show()