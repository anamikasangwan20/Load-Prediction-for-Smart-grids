import pandas as pd
import os
import sys
import numpy as np
import pickle
import scipy.io as sio
from datetime import date as DateTimeDate
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.utils.extmath import cartesian
import datetime

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

folder_name = 'ausgrid_data'
createFolder(folder_name)

#---------------------------#
#                           #
#   Loading data            #
#                           #
#---------------------------#
df1 = pd.read_csv("2010_2011_solar.csv",sep=',',skiprows=[0])
df2 = pd.read_csv("2011_2012_solar.csv",sep=',',skiprows=[0])
df3 = pd.read_csv("2012_2013_solar.csv",sep=',',skiprows=[0])
frames = [df1, df2, df3]

df = pd.concat(frames)
customerid = df.Customer.unique()

intervals = cartesian([np.array(range(24)).astype(str),['00','30']])
intervals = np.vstack((np.delete(intervals,0,0),['0','00']))

dat_length = len(customerid)

#---------------------------#
#                           #
#   Preprocessing data      #
#                           #
#---------------------------#

days_in_month = 29  #change this

dat_gencap = np.zeros((dat_length))
dat_solar = np.zeros((dat_length,days_in_month,48))
dat_totalload = np.zeros((dat_length,days_in_month,48))
for i in range(dat_length):
    temp1 = df.loc[df['Customer'] == customerid[i]]
    dat_gencap[i] = temp1['Generator Capacity'].iloc[0]
    tempCL = temp1.loc[temp1['Consumption Category']=='CL']
    tempGG = temp1.loc[temp1['Consumption Category']=='GG']
    tempGC = temp1.loc[temp1['Consumption Category']=='GC']
    dat_CL_available = np.zeros((days_in_month)).astype(bool)
    for day in range(days_in_month):
        #date_to_extract = str(day+1)+'-'+'Feb'+'-'+str(12)   #change the year and month here to modify month to extract
        if day+1 < 13:
            date_to_extract = str(day+1)+'/'+'2'+'/'+str(12)    #Use this for July 2011 and future i.e. files 2 and 3
        else:
            date_to_extract = str(day+1)+'/'+'02'+'/'+str(2012)
        tempCL_day = tempCL.loc[tempCL['date']==date_to_extract]
        if(tempCL_day.shape[0] != 0):
            dat_CL_available[day] = 1
        tempGC_day = tempGC.loc[tempGC['date']==date_to_extract]
        tempGG_day = tempGG.loc[tempGG['date']==date_to_extract]
        for j in range(48):
            timeInterval = intervals[j,0]+':'+intervals[j,1]
            dat_solar[i,day,j] = tempGG_day[timeInterval].iloc[0]
            if(dat_CL_available[day]):
                dat_totalload[i,day,j] = tempCL_day[timeInterval].iloc[0]+tempGC_day[timeInterval].iloc[0]-tempGG_day[timeInterval].iloc[0]
            else:
                dat_totalload[i,day,j] = tempGC_day[timeInterval].iloc[0]-tempGG_day[timeInterval].iloc[0]
                
pickle.dump(dat_solar, open('dat_solar.pkl','wb'))  #Solar data
pickle.dump(dat_totalload, open('dat_totalload.pkl','wb'))  #Net Load data
pickle.dump(dat_CL_available, open('dat_CL_available.pkl','wb'))
pickle.dump(dat_gencap, open('dat_gencap.pkl','wb'))
