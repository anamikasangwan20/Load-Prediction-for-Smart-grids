
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
import csv

start_time = time.time()

def combine_daily_data (load_data, T):
    # load_data is expected to be 3D [no_customers x no_days x readings_per_day], T= no_of_days
    # the ouput is a 2D array [timeseries_samples x AMI_readings_for_T_days] 
    discard_days = load_data.shape[1]%T
    if (discard_days):
        load_data = load_data[:,:-discard_days,:]   # discard the remainder days 
    splits = int(load_data.shape[1] / T)
    data_T = []
    for i in range (load_data.shape[0]):                                #   for each customer
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
    #mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    mape = mape_(forecast,actual)  # MAPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    return({'mae': mae, 'mape':mape, 'rmse':rmse})

#############################################################
#
#  Load and Divide the data (daily timeseries) into PV and non-PV customers  
#
############################################################

dat_load = pickle.load(open('dat_totalload.pkl','rb'), encoding='latin1')
data_load_PV = dat_load[150:]           #   [150:300] reserved as PV customers
data_load_NPV = dat_load[:150]          #  [0:150] reserved as non-PV customers

############################################################
#
#  Regroup the timeseries into samples of 'T' days 
#
############################################################
for T in ([14]):
    #order = '2 0 2'
    #p, d, q = 2, 0, 2
    T = 14
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
    
    ############################################################
    #
    #   LSTM
    #
    ############################################################
    BATCH = 2
    EPOCHS = 10
    blocks = 50
    lookback = 5            # no_features in lstm = lookback
    
    #------- Reshaping the data into 3D format: [batches, timeseries_samples, no_features] --------#
    
    arr = np.zeros((dataT_norm.shape[0], lookback+1))           # lookback + 1 for X and Y in the same array
    rnn_data = np.zeros((dataT_norm.shape[0], dataT_norm.shape[1]-lookback , lookback+1))
    for i in range(dataT_norm.shape[1]-lookback):
        arr = dataT_norm[:, i : i+lookback+1]
        rnn_data[:,i,:] = arr
    
# #_____ Correct Version using one ouput at the end of Tx48 series as Y -------#
    
#     arr = np.zeros((dataT_norm.shape[0], lookback))           # lookback + 1 for X and Y in the same array
#     rnn_data = np.zeros((dataT_norm.shape[0], dataT_norm.shape[1]-lookback , lookback))
#     for i in range(dataT_norm.shape[1]-lookback):
#         arr = dataT_norm[:, i : i+lookback]
#         rnn_data[:,i,:] = arr
    #rnn_data = np.reshape(dataT_norm, (dataT_norm.shape[0], dataT_norm.shape[1], lookback))
    
    #------- Train-Test Split using TimeSeriesSplit from sklearn --------#
    
    #training_data = rnn_data
    #tscv = TimeSeriesSplit(n_splits=3)
    #abc = tscv.split(training_data)     #   splits customer-wise: 39 train, then 76 train then 113 train (adding test in sizes of 37, 3 times)
    #pqr = tscv.split(training_data[0])  #   splits timestamp-wise: 3 increments of 372 test data points to training set. [0,371]+[372,743]; [:743]+[744, 1115]; [:1115]+[1116, 1487]
    
    #------- Train-Val-Test Split customerwise manually --------#
    train_p, val_p, test_p = 70, 15, 15                       
    split1 = int(rnn_data.shape[0] * train_p/100)         #   timeseries samples' train-val-test split
    split2 = int(rnn_data.shape[0] * (train_p+val_p)/100)
    
    print("\nTrain - Val - Test split on timseries: {0}% - {1}% - {2}%\nTraining range: [0 :{3}] \nValidation Range: [{4} : {5}] \nTest range: [{6} : {7}]\n" .format( int(train_p), int(val_p), int(test_p), split1, split1, split2, split2, rnn_data.shape[0]))
    
    train, val, test = np.split(rnn_data, [split1, split2])      #   splitting sample-wise (not considering each customer's records together)
    
    #------- X and Y Split --------#
    
    trainX , trainY = train[:,:,:-1] , train[:,:,-1]
    valX , valY = val[:,:,:-1] , val[:,:,-1]
    testX , testY = test[:,:,:-1] , test[:,:,-1]
    
    # Reshape Y :
    trainY, valY, testY = np.reshape(trainY, (trainX.shape[0], trainX.shape[1], 1)), np.reshape(valY, (valX.shape[0], valX.shape[1], 1)), np.reshape(testY, (testX.shape[0], testX.shape[1], 1))
    
    # trainX , trainY = train[:,:-lookback,:] , train[:,lookback:,:]
    # valX , valY = val[:,:-lookback,:] , val[:,lookback:,:]
    # testX , testY = test[:,:-lookback,:] , test[:,lookback:,:]
    
    #------- Define/Build Model: Adding Layers (Input Dimension = (48*T, 1)) --------#
    model = Sequential()
    model.add(LSTM(blocks, input_shape = (trainX.shape[1] , trainX.shape[2]), return_sequences = True))     # batch_size is flexible
    model.add(Dense(1))
    model.compile(loss = 'mae', optimizer = 'adam')
    
    #------- Train Model: Feeding input to the structure (Input Dimension = (48*T,1)) --------#
    history = model.fit(trainX, trainY, epochs = EPOCHS, batch_size = BATCH, validation_data = (valX, valY), verbose=0, shuffle=False)
    model.save('models/lstm1_TLstmBatchEpoch_%d-%d-%d-%d.hdf5'%(T, blocks, BATCH, EPOCHS))
    
    #------- Plots --------#
    # Loss Curve over epochs
    plt.figure()
    plt.plot(history.history['loss'], label = 'training loss')
    plt.plot(history.history['val_loss'], label = 'validation loss')
    plt.legend()
    plt.title("Loss curve for T-lstm-batch-epochs = %d %d %d %d" % (T,blocks,BATCH,EPOCHS))
    plt.savefig('Plots_lstm1/Loss_curve_T%dL%dB%dE%dLook%d.png'%(T,blocks,BATCH,EPOCHS,lookback))
    
    # Peformance Curve 
    #model = load_model('models/lstm1_TLstmBatchEpoch_%d-%d-%d-%d.hdf5'%(T, blocks, BATCH, EPOCHS))
    sequence = 0
    plt.figure()
    input_index = [i for i in range (len(trainX[sequence][:,-1]))]
    input_series = pd.Series(trainX[sequence][:,-1], index = input_index)
    pred_index = [i for i in range( len(trainX[sequence][:,1]) , len(trainX[sequence])+1 )]
    prediction_series = pd.Series(model.predict(trainX)[sequence][-1], index = pred_index)
    future_series = pd.Series(trainY[sequence][-1], index = pred_index)
    
    plt.plot(input_series[-50:], '-', label = 'History')
    plt.plot(prediction_series, 'go', label = 'Model Prediction')
    plt.plot(future_series, 'rx', label = 'True Future')
    plt.legend()
    plt.title("LSTM Model for sequence #1 | T-lstm-batch-epochs = %d %d %d %d" % (T,blocks,BATCH,EPOCHS) )
    plt.xlabel('timesteps')
    plt.savefig('Plots_lstm1/LSTM_Prediction_Curve_T%dL%dB%dE%dLook%d.png'%(T,blocks,BATCH,EPOCHS,lookback))
    
    # Checking for autoencoder-like learning pattern
    plt.figure()
    plt.plot(model.predict(trainX)[:50,-1,-1], label = 'one-step model prediction')
    plt.plot(trainY[:50,-1,-1], label = 'actual future value')             #or []:50,-1,:]
    plt.title('Predicted vs Actual Output for first 50 sequences')
    plt.xlabel('timeseries sequences')
    plt.legend()
    plt.figure()
    plt.plot(model.predict(trainX)[:50,-1,-1], label = 'one-step model prediction for each sequence')
    plt.plot(trainX[:50,-1,-1], label = 'input last timestep value for each sequence')
    plt.title('Comparing similarity between prediction and input values for first 50 sequences')
    plt.xlabel('timeseries sequences')
    plt.legend()
    plt.figure()
    plt.plot(model.predict(trainX)[:50,-1,-1], label = 'one-step model prediction for each sequence')
    plt.plot(trainX[:50,-1,-2], label = 'input -2 timestep value for each sequence')
    plt.title('Comparing similarity between prediction and input values')
    plt.xlabel('timeseries sequences')
    plt.legend()
    plt.figure()
    plt.plot(model.predict(trainX)[:50,-1,-1], label = 'one-step model prediction for each sequence')
    plt.plot(trainX[:50,-1,-3], label = 'input -3 timestep value for each sequence')
    plt.title('Comparing similarity between prediction and input values')
    plt.xlabel('timeseries sequences')
    plt.legend()
    plt.figure()
    plt.plot(model.predict(trainX)[:50,-1,-1], label = 'one-step model prediction for each sequence')
    plt.plot(trainX[:50,-1,-4], label = 'input -4 timestep value for each sequence')
    plt.title('Comparing similarity between prediction and input values')
    plt.xlabel('timeseries sequences')
    plt.legend()
    plt.figure()
    plt.plot(model.predict(trainX)[:50,-1,-1], label = 'one-step model prediction for each sequence')
    plt.plot(trainX[:50,-1,-5], label = 'input -5 timestep value for each sequence')
    plt.title('Comparing similarity between prediction and input values')
    plt.xlabel('timeseries sequences')
    plt.legend()
    
    # #------- Plot training and testing progress --------#
    # plt.figure()
    # plt.plot(history.history['loss'], label = 'training loss')
    # plt.plot(history.history['val_loss'], label = 'validation loss')
    # plt.legend()
    # plt.figure()
    # plt.plot(model.predict(trainX)[:50], label = 'predicted trainY')
    # plt.plot(trainY[:50], label = 'actual trainY')
    # plt.legend()
    # plt.figure()
    # plt.plot(model.predict(testX)[:50], label = 'predicted testY')
    # plt.plot(testY[:50], label = 'actual testY')
    # plt.legend()
    
    # #------- Plot individual series history and future --------#
    
    # plt.figure()
    # plt.plot(trainX[0], '-', label = 'History')
    # prediction_series = pd.Series(model.predict(trainX)[0], index = [len(trainX[0])])
    # future_series = pd.Series(trainY[0], index = [len(trainX[0])])
    # plt.plot(prediction_series, 'go', label = 'Model Prediction')
    # plt.plot(future_series, 'rx', label = 'True Future')
    # plt.legend()
    # plt.title('LSTM One-step Model')
     
     
    #------- Evaluation on training and test set --------#
    print('\nTraining error...')    
    print(forecast_accuracy(model.predict(trainX), trainY))
    print('\nTesting error...')
    print(forecast_accuracy(model.predict(testX), testY))
    #   Calculating program execution time
    exec_time = int(time.time()-T_time)
    print("Execution time for T =",T,"days:", exec_time,"sec")
    
    # #------- Evaluation w.r.t the time-shifted signal (erroneous) --------#
    
    # plt.figure()
    # plt.plot(model.predict(trainX)[1][:50], label = 'predicted trainY')
    # plt.plot(dataT_norm[1][lookback-1:50], label = 'actual trainY shifted 1')
    # plt.legend()
    # plt.figure()
    # plt.plot(model.predict(testX)[1][:50], label = 'predicted test')
    # plt.plot(testX[1][:50], label = 'testX')
    # plt.legend()
    # plt.figure()
    # plt.plot(model.predict(testX)[1][:50], label = 'predicted test')
    # plt.plot(testX[1][:50][:,-1], label = 'actual testY shifted 1')
    # plt.legend()
    
    # print('\nTraining error...')    
    # print(forecast_accuracy(model.predict(trainX)[:,:,-1], trainY[:,:,-1]))
    # print('\nTesting error...')
    # print(forecast_accuracy(model.predict(testX)[:,:,-1], testY[:,:,-1]))
    
    # print('\nTraining error...')    
    # print(forecast_accuracy(model.predict(trainX)[:,:,-1], trainX[:,:,-1]))
    # print('\nTesting error...')
    # print(forecast_accuracy(model.predict(testX)[:,:,-1], testX[:,:,-1]))
    
    
    #------- Writing Results in csv --------#
    train_accuracy=forecast_accuracy(model.predict(trainX), trainX)
    test_accuracy=forecast_accuracy(model.predict(testX), testX)
    f = open("results_lstm.csv","a", newline = '')
    writer = csv.writer(f, delimiter=' ')
    writer.writerow("\n")
    writer.writerow([T,blocks, EPOCHS, BATCH, '70:15:15', 'Training', round(train_accuracy['mae'],3),round(train_accuracy['mape']*100,2),round(train_accuracy['rmse'],3),exec_time])
    writer.writerow([T,blocks, EPOCHS, BATCH, '70:15:15', 'Test', round(test_accuracy['mae'],3),round(test_accuracy['mape']*100,2),round(test_accuracy['rmse'],3)])
    f.close()
    


#  Calculating program execution time
print("Execution time: ", int(time.time()-start_time), "sec") 






   