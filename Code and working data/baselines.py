#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 16:41:23 2020

@author: anamika
"""
import time
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
import pandas as pd
#import pmdarima as pm
#import csv

start_time = time.time()

def combine_daily_data (load_data, T):
    # load_data is expected to be 3D [no_consumers x no_days x readings_per_day], T = no_of_days
    # the ouput is a 2D array [timeseries_samples x AMI_readings_for_T_days] 
    discard_days = load_data.shape[1]%T
    if (discard_days):
        load_data = load_data[:,:-discard_days,:]   # discard the remainder days 
    splits = int(load_data.shape[1] / T)
    data_T = []
    for i in range (load_data.shape[0]):                                #   for each consumer
        split_arrays = np.array_split(load_data[i], splits, axis = 0)   #   split the monthly data into groups of 'T days' 
        for arr in split_arrays:
            arr_flat = arr.flatten()            # flatten that T days' data into one vector
            data_T.append(arr_flat) 
    data_T = np.asarray(data_T)
    return (data_T)

# Average MAPE barring samples with zero values:
def mape_(forecast, actual):
    mape=[]
    for i in range (actual.shape[0]):
        mape.append(np.mean(np.abs(forecast[i] - actual[i])/np.abs(actual[i])))
    mape = np.asarray(mape)
    mape = mape[mape<100]       #removes all inf elements/rows
    avg_mape = np.mean(mape)    #avearges mape values for all training samples
    return (avg_mape)

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mae = np.mean(np.abs(forecast - actual))  # MAE
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    #mape = mape_(forecast,actual)  # MAPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({'mae': mae, 'mape':mape, 'rmse':rmse})

#############################################################
#
#  Load and Divide the data into PV and non-PV consumers  
#
############################################################

dat_load = pickle.load(open('dat_totalload.pkl','rb'), encoding='latin1')
data_load_PV = dat_load[150:]           #   [150:300] reserved as PV consumers
data_load_NPV = dat_load[:150]          #  [0:150] reserved as non-PV consumers

############################################################
#
#  Regroup the timeseries into samples of 'T' days 
#
############################################################
for T in ([14]):
    order = '2 0 2'
    p, d, q = 2, 0, 2
    #T = 7
    T_time = time.time()
    data_T = combine_daily_data (data_load_PV, T)   
    print("\n T = ", T,"days") 
    
    ############################################################
    #
    #  Normalize the T-length timeseries samples
    #
    ############################################################
    
    scaler = MinMaxScaler(feature_range = (0,1))    #   MinMax Normalization
    dataT_norm = scaler.fit_transform(data_T)       #    rescales data to [0,1]
    
    # ############################################################
    # #
    # #   LSTM
    # #
    # ############################################################
    # BATCH = 8
    # EPOCHS = 40
    # blocks = 5
    
    # #------- Reshaping the data into 3D format: [batches, timeseries_samples, no_features] --------#
    # rnn_data = np.reshape(dataT_norm, (dataT_norm.shape[0], dataT_norm.shape[1],1))
    
    # #------- Train-Test Split using TimeSeriesSplit from sklearn --------#
    
    # #training_data = rnn_data
    # #tscv = TimeSeriesSplit(n_splits=3)
    # #abc = tscv.split(training_data)     #   splits consumer-wise: 39 train, then 76 train then 113 train (adding test in sizes of 37, 3 times)
    # #pqr = tscv.split(training_data[0])  #   splits timestamp-wise: 3 increments of 372 test data points to training set. [0,371]+[372,743]; [:743]+[744, 1115]; [:1115]+[1116, 1487]
    
    # #------- Train-Val-Test Split consumerwise manually --------#
    # train_p, val_p, test_p = 70, 15, 15                       
    # split1 = int(rnn_data.shape[0] * train_p/100)         #   timeseries samples' train-val-test split
    # split2 = int(rnn_data.shape[0] * (train_p+val_p)/100)
    
    # print("\nTrain - Val - Test split on timseries: {0}% - {1}% - {2}%\nTraining range: [0 :{3}] \nValidation Range: [{4} : {5}] \nTest range: [{6} : {7}]\n" .format( int(train_p), int(val_p), int(test_p), split1, split1, split2, split2, rnn_data.shape[0]))
    
    # train, val, test = np.split(rnn_data, [split1, split2])      #   splitting sample-wise (not considering each consumer's records together)
    
    # #------- X and Y Split --------#
    # trainX , trainY = train[:,:-1,:] , train[:,1:,:]
    # valX , valY = val[:,:-1,:] , val[:,1:,:]
    # testX , testY = test[:,:-1,:] , test[:,1:,:]
    
    # #------- Define/Build Model: Adding Layers (Input Dimension = (48*T, 1)) --------#
    # model = Sequential()
    # model.add(LSTM(blocks, input_shape = (trainX.shape[1] , trainX.shape[2]), return_sequences = True))     # batch_size is flexible
    # model.add(Dense(1))
    # model.compile(loss = 'mae', optimizer = 'adam')
    
    # #------- Train Model: Feeding input to the structure (Input Dimension = (48*T,1)) --------#
    # history = model.fit(trainX, trainY, epochs = EPOCHS, batch_size = BATCH, validation_data = (valX, valY), verbose=0, shuffle=False)
    
    # #------- Plot training and testing progress --------#
    # plt.figure()
    # plt.plot(history.history['loss'], label = 'training loss')
    # plt.plot(history.history['val_loss'], label = 'validation loss')
    # plt.legend()
    # # plt.figure()
    # # plt.plot(model.predict(trainX)[0][:100], label = 'predicted train')
    # # plt.plot(trainX[0][:100], label = 'actual train')
    # # plt.legend()
    # # plt.figure()
    # # plt.plot(model.predict(testX)[0][:100], label = 'predicted test')
    # # plt.plot(testX[0][:100], label = 'actual test')
    # # plt.legend()
    
    # #------- Evaluation on training and test set --------#
    # print('\nTraining error...')    
    # print(forecast_accuracy(model.predict(trainX), trainX))
    # print('\nTesting error...')
    # print(forecast_accuracy(model.predict(testX), testX))
    # #   Calculating program execution time
    # exec_time = int(time.time()-T_time)
    # print("Execution time for T =",T,"days:", exec_time,"sec")
    
    # train_accuracy=forecast_accuracy(model.predict(trainX), trainX)
    # test_accuracy=forecast_accuracy(model.predict(testX), testX)
    # f = open("results_lstm.csv","a", newline = '')
    # writer = csv.writer(f, delimiter=' ')
    # writer.writerow("\n")
    # writer.writerow([T,blocks, EPOCHS, BATCH, '70:15:15', 'Training', round(train_accuracy['mae'],3),round(train_accuracy['mape']*100,2),round(train_accuracy['rmse'],3),exec_time])
    # writer.writerow([T,blocks, EPOCHS, BATCH, '70:15:15', 'Test', round(test_accuracy['mae'],3),round(test_accuracy['mape']*100,2),round(test_accuracy['rmse'],3)])
    # f.close()
    
    ############################################################
    #
    #   ARIMA
    #
    ############################################################
    
    #------- Testing timeseries stationarity using ad fuller test --------#
    
    count = 0
    for timeseries in dataT_norm:
        p_value = adfuller(timeseries)[1]
        if (p_value > 0.05):
            count +=1
    print("Number of non-stationary samples : ", count)    
    
    #------- ARIMA model training --------#
    
    consumer = 0
    dataT_norm = [dataT_norm[consumer], dataT_norm[10], dataT_norm[20], dataT_norm[50]]    #-------$$$$ TEST
    p, d, q = 2, 0, 2
    
    for timeseries in dataT_norm:
        # PACF Analysis
        
        # ACF Analysis
        
    #    # Autoarima model: ARMA._fit_start_params
    #    model = pm.auto_arima(timeseries, start_p=1, start_q=1, test='adf', max_p=6, max_q=6, m=1, d=None, seasonal=False, start_P=0, D=0, trace=True, error_action='ignore', supress_warnings=True, stepwise=True)
    #    print(model.summary())
    #    
        # Model fit
        #model = ARIMA(timeseries, order = (p,d,q))      #-------$$$$ TUNE
        #results = model.fit(disp=-1)
        #print(results.summary())
        # Aggregate AIC
        
        # Training prediction error - Do we need this?
        
    #    # Plot Actual vs Fitted Values
    #    results.plot_predict(dynamic=False)
    #    plt.show()
        
        # Out-of-sample error OR Forecast error
        
        #Train-Test Split
        train_p = 60                                    #-------$$$$ TUNE
        split = int(timeseries.shape[0] * (train_p/100))
        print("\nTrain - Test split: {0}% - {1}%" .format( int(train_p), int(100-train_p)))
        train = timeseries[:split]
        test = timeseries[split:]
        
        #Build Model
        model = ARIMA(train, order = (p,d,q))
        fitted = model.fit(disp=-1)
    #    model = pm.auto_arima(train, start_p=1, start_q=1, test='adf', max_p=6, max_q=6, m=1, d=None, seasonal=False, start_P=0, D=0, trace=True, error_action='ignore', supress_warnings=True, stepwise=True)
    #    print(model.summary())
    #    fitted=model
        
        #Forecast
        for npred in ([15]):    # number of prediction days          #-------$$$$ TUNE
            fc, se, conf = fitted.forecast(npred, alpha = 0.05)    #95% confidence
            
            # Make as pandas series
            pred_index = [i for i in range(split,split+npred)]
            test_index = [i for i in range(split,timeseries.shape[0])]
            fc_series = pd.Series(fc, index = pred_index)
            lower_series = pd.Series(conf[:, 0], index = pred_index)
            upper_series = pd.Series(conf[:, 1], index = pred_index)
            test = pd.Series(test, index = test_index)
            
            # Plot
            plt.figure(figsize=(12,5), dpi=100)
            plt.plot(train, label='training')
            plt.plot(test, label='actual')
            plt.plot(fc_series, label='forecast')
            plt.fill_between(lower_series.index, lower_series, upper_series, 
                            color='k', alpha=.15)
            plt.title('Forecast vs Actuals')
            plt.legend(loc='upper left', fontsize=8)
            plt.show()
            
            # Evaluation Metrics
            forecast_acc = forecast_accuracy(fc, test[:npred])
            print(forecast_accuracy(fc, test[:npred]))
            
            # Rolling Forecast
            history = [x for x in train]
            test_rolling = [x for x in test[:npred]]        # converting series to list
            prediction = list()
            for i in range (len(test_rolling)):
                model = ARIMA(history , order = (p,d,q))
                model_fit = model.fit(disp = 0)
                output = model_fit.forecast()       # forecast() performs one step forecast on trained model
                y_pred = output[0]
                prediction.append(y_pred)
                history.append(test_rolling[i])      # update the history with the actual (n+1)th value to predict further values
            
            #  Evaluation Metrics
            prediction = np.asarray(prediction)
            test_rolling = np.asarray(test_rolling)
            rolling_forecast_acc = forecast_accuracy(prediction, test_rolling)
            print(forecast_accuracy(prediction, test_rolling))
            plt.figure(2)
            plt.plot(test_rolling , label = 'Actual Series')    
            plt.plot(prediction, color = 'red' , label = 'Predicted Series')
            plt.legend(loc = 'best')
            plt.xlabel('Timestamps')
            plt.title('Plots')
            plt.show()
            
            #   Calculating program execution time
            exec_time = int(time.time()-T_time)
            print("Execution time for T =",T,"days:", exec_time,"sec")
            
            ##   Print metrics - mae, mape, rmse - in an excel file
            # f1 = open("results_arima.csv","a", newline = '')
            # writer = csv.writer(f1, delimiter=' ')
            # writer.writerow("\n")
            # writer.writerow([consumer,T,order,npred,round(forecast_acc['mae'],3),round(forecast_acc['mape']*100,2),round(forecast_acc['rmse'],3),exec_time])
            # writer.writerow([consumer,T,order,npred,round(rolling_forecast_acc['mae'],3),round(rolling_forecast_acc['mape']*100,2),round(rolling_forecast_acc['rmse'],3)])
            # f1.close()
            
            # For all consumer average
            #mape = np.append(mape)
            
        break

#   Calculating program execution time
#print("Execution time: ", int(time.time()-start_time), "sec")






   