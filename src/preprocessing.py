import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from mat4py import loadmat
import json
import datetime

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from scipy.signal import lfilter

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
    
# function loads the preprocessed data into DataFrames. 
def import_preprocessed_data():
    path_0 = '../data/preprocessed/clusters/0/'
    path_1 = '../data/preprocessed/clusters/1/'
    path_2 = '../data/preprocessed/clusters/2/'

    dfs_c_0 = []
    dfs_c_1 = []
    dfs_c_2 = []
    dfs_c_0_grouped = []
    dfs_c_1_grouped = []
    dfs_c_2_grouped = []

    csv_files_0 = [csv for csv in os.listdir(path_0) if csv.endswith('.csv')]
    csv_files_1 = [csv for csv in os.listdir(path_1) if csv.endswith('.csv')]
    csv_files_2 = [csv for csv in os.listdir(path_2) if csv.endswith('.csv')]

    for file in csv_files_0:
        # import DataFrame
        df = pd.read_csv(path_0 + file)

        if file.startswith('g_'):
          dfs_c_0_grouped.append(df)
        else:
          dfs_c_0.append(df)

    for file in csv_files_1:
        # import DataFrame
        df = pd.read_csv(path_1 + file)

        if file.startswith('g_'):
          dfs_c_1_grouped.append(df)
        else:
          dfs_c_1.append(df)

    for file in csv_files_2:
        # import DataFrame
        df = pd.read_csv(path_2 + file)

        if file.startswith('g_'):
          dfs_c_2_grouped.append(df)
        else:
          dfs_c_2.append(df)
    
    return dfs_c_0, dfs_c_1, dfs_c_2, dfs_c_0_grouped, dfs_c_1_grouped, dfs_c_2_grouped


'''für jedes Cluster unterschiedliche calculate empties funktionen da die Container ihren durchschnittlichen Füllstand 
an sehr unterschiedlichen Bereichen haben '''
# function calculates the empties of a container.
def calculate_empties(df, threshold):
    empties_indices = []
    for idx,val in enumerate(df['inter_pol']):
        if idx != 0 and (df['inter_pol'][idx-1]*threshold < (df['inter_pol'][idx])) and (df['inter_pol'][idx] > 120): 
            empties_indices.append(idx)
            
    # filter out double values
    for idx,val in enumerate(empties_indices): 
        if idx !=0 and (empties_indices[idx] - empties_indices[idx-1]) < 3: 
            empties_indices.pop(idx)
    return empties_indices

def calculate_empties_adaptive(df, threshold):
    top_thresh = df.inter_pol.mean() + (df.inter_pol.std() * 0.75)
    empties_indices = []
    for idx,val in enumerate(df['inter_pol']):
        if idx != 0 and (df['inter_pol'][idx-1]*threshold < (df['inter_pol'][idx])) and (df['inter_pol'][idx] > top_thresh): 
            empties_indices.append(idx)
            
    # filter out double values
    for idx,val in enumerate(empties_indices): 
        if idx !=0 and (empties_indices[idx] - empties_indices[idx-1]) < 3: 
            empties_indices.pop(idx)
    return empties_indices

# function calculation empties of container for cluster 0
def calculate_empties_0(df, threshold):
    empties_indices = []
    for idx,val in enumerate(df['inter_pol']):
        if idx != 0 and (df['inter_pol'][idx-1]*threshold < (df['inter_pol'][idx])) and (df['inter_pol'][idx] > 120): 
            empties_indices.append(idx)
            
    # filter out double values
    for idx,val in enumerate(empties_indices): 
        if idx !=0 and (empties_indices[idx] - empties_indices[idx-1]) < 3: 
            empties_indices.pop(idx)
    return empties_indices


# function calculation empties of container for cluster 1
def calculate_empties_1(df, threshold):
    empties_indices = []
    for idx,val in enumerate(df['inter_pol']):
        if idx != 0 and (df['inter_pol'][idx-1]*threshold < (df['inter_pol'][idx])) and (df['inter_pol'][idx] > 60): 
            empties_indices.append(idx)
            
    # filter out double values
    for idx,val in enumerate(empties_indices): 
        if idx !=0 and (empties_indices[idx] - empties_indices[idx-1]) < 3: 
            empties_indices.pop(idx)
    return empties_indices



# function calculation empties of container for cluster 2
def calculate_empties_2(df, threshold):
    empties_indices = []
    for idx,val in enumerate(df['inter_pol']):
        if idx != 0 and (df['inter_pol'][idx-1]*threshold < (df['inter_pol'][idx])) and (df['inter_pol'][idx] > 65): 
            empties_indices.append(idx)
            
    # filter out double values
    for idx,val in enumerate(empties_indices): 
        if idx !=0 and (empties_indices[idx] - empties_indices[idx-1]) < 3: 
            empties_indices.pop(idx)
    return empties_indices



# function adds the calculated emptied to the container DataFrame.
def add_empties_column(df, values):
    df_final = df
    df_final['empties'] = 0
    for i in range(0, len(df), 1): 
        if i in values: 
            df_final['empties'][i] = 1
    return df_final


# function parses JSON data into csv file format. Only the important information is saved in csv
def create_csv_files():
    path_to_json = '../data/raw/data/' 
    json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.txt')]
    csv_folder = '../data/preprocessed/CSV/'

    for filename in json_files:
        f = open(path_to_json + filename,)
        data = json.load(f)
        f.close()
        json_objects = data[1]

        # create important columns
        columns = ['id','deveui', 'unix_time', 'client_id', 'created_at', 'Status', 'Sensor ID', 'Events', 'Height', 'Voltage', 'Temperature', 'Tilt', 'Tx Event', 'Messagetype']
        df = pd.DataFrame(columns=columns)

        # import all json data into dataframes
        for i in range(0, len(json_objects)):
            new_data = {'id': int(json_objects[i]['id']),'deveui':str(json_objects[i]['deveui']), 'unix_time':int(json_objects[i]['unix_time']), 'client_id':str(json_objects[i]['client_id']), 'created_at':str(json_objects[i]['created_at']), 'Status':json_objects[i]['decoded_data']['sensor_data']['Status'], 'Sensor ID':str(json_objects[i]['decoded_data']['sensor_data']['Sensor ID']), 'Events':str(json_objects[i]['decoded_data']['sensor_data']['Events']), 'Height':str(json_objects[i]['decoded_data']['sensor_data']['Height 1']), 'Voltage':str(json_objects[i]['decoded_data']['sensor_data']['Voltage']), 'Temperature':str(json_objects[i]['decoded_data']['sensor_data']['Temperature']), 'Tilt':str(json_objects[i]['decoded_data']['sensor_data']['Tilt']), 'Tx Event':str(json_objects[i]['decoded_data']['sensor_data']['Tx Event'])}
            df = df.append(new_data, ignore_index=True)

        df = df[::-1] #reverse values
        
        # parse data into correct data types
        df['Height'] = df['Height'].apply(lambda x: str(x).split(' ')[0])
        df['Voltage'] = df['Voltage'].apply(lambda x: str(x).split(' ')[0])
        df['Temperature'] = df['Temperature'].apply(lambda x: str(x).split(' ')[0])
        df['Tilt'] = df['Tilt'].apply(lambda x: str(x).split(' ')[0])
        df['Tx Event'] = df['Tx Event'].apply(lambda x: str(x).split(' ')[0])

        df['Height'] = df['Height'].astype('int')
        df['Voltage'] = df['Voltage'].astype('int')
        df['Temperature'] = df['Temperature'].astype('int')
        df['Tilt'] = df['Tilt'].astype('int')
        df['Tx Event'] = df['Tx Event'].astype('int')

        # save DataFrames in csv files 
        filename = df['deveui'][0]
        filename = filename + ".csv"
        df.to_csv(csv_folder + filename)
        
        
        
# function will assign holiday value the given container data frame
def assign_holidays(df, df_holiday):
    df['holiday'] = 0

    for i in range(0,len(df), 1):
        form = "%Y-%m-%d"
        d1 = datetime.datetime.strptime(df['time_stamp'][i], form)

        #get holiday value
        for j in range(0, len(df_holiday), 1):
            d2 = datetime.datetime.strptime(df_holiday['Date'][j], form)
            if d1 == d2: 
                df['holiday'][i] = df_holiday['Holiday'][j]
                
                
                
# function will assign weather data of Frankfurt airport to container data frame
def assign_weather(df, df_weather):
    df['temp avg'] = 0.0
    df['temp min'] = 0.0
    df['temp max'] = 0.0
    df['rainfall sum'] = 0.0
    df['snowfall sum'] = 0.0
    df['sunshine minutes'] = 0

    for i in range(0,len(df), 1):
        form = "%Y-%m-%d"
        d1 = datetime.datetime.strptime(df['time_stamp'][i], form)

        #get weather values
        for j in range(0, len(df_weather), 1):
            d2 = datetime.datetime.strptime(df_weather['date'][j], form)
            if d1 == d2: 
                df['temp avg'][i] = df_weather['tavg'][j]
                df['temp min'][i] = df_weather['tmin'][j]
                df['temp max'][i] = df_weather['tmax'][j]
                df['rainfall sum'][i] = df_weather['prcp'][j]
                df['snowfall sum'][i] = df_weather['snow'][j]
                df['sunshine minutes'][i] = df_weather['tsun'][j]


'''brauchbare Werte werden in Integer umgewandelt, Messfehler [Height > 190 & Temperatur > 100] werden bereineigt. 
Wobei Messfehler in Bezug auf die Höhe gelöscht werden und Temperatur Messfehler mit NaN Werten überschrieben werden '''
def cleaning_del(df):
    
    new = pd.DataFrame()
    #setting device id to join multiple bins later in the process
    new['device_id'] = df['deveui']
    #seeting time stamp
    new['time_stamp'] = pd.to_datetime(df['created_at'], format='%Y-%m-%d')    
    #deleting rows with values height > 190 -> Measurement errors
    new['Height'] = df['Height'].str.replace('cm', '').astype(int)
    new = new[new.Height < 190]
    
    #casting temperature to int and replace values > 100 with NaN
    new['Temperature'] = df['Temperature'].str.replace('C', '').astype(int)
    new.loc[(new.Temperature > 100), 'Temperature'] = np.nan
    #casting Tilt to int
    new['Tilt'] = df['Tilt'].str.replace('Degree', '').astype(int)
    #sorting values acording to time_stamp
    new.sort_values(by=['time_stamp'], ascending=True, inplace = True)

    return new


''' unterschiedliche Smoothing verfahren werden angewandt - einmal auf die ungruppierten    Daten und einmal auf die gruppierten Daten. Als aggregationsfunktion wird .mean() verwendet. Als Level wird auf eintägig guppiert. 
NaN Temperaturwerte werden mit dem durchschnittlichen Temperaturwert überschrieben - gleiches gilt für Tilt.
beim mov_avg auf height werden NaN durch Interpolation gefüllt
'''

def smoothing_fillingNaN(df):
     
    #smooth with lfilter
    n = 60  # the larger n is, the smoother curve will be
    b = [1.0 / n] * n
    a = 1
    df['lfilter'] = lfilter(b,a, df['Height'])

    #moving average smoothing
    df['mov_avg'] = df['Height'].rolling(30).mean()

    #minimum moving average
    df['min_avg'] = df['Height'].rolling(30).min()

    
    #creating new DataFrame on Level Daily with aggregation max()
    daily = df.groupby(pd.Grouper(key='time_stamp', axis=0, 
                        freq='1D', sort=True)).mean()
    #add device_id
    daily['device_id'] = df.iloc[1, 0]

    #further smoothing on daily level with rolling mean window 2
    daily['mov_avg'] = daily['Height'].rolling(2).mean()

    #fill missing values with interpolation
    df['inter_pol'] = df['mov_avg'].interpolate(limit_direction = 'both')
    daily['inter_pol'] = daily['mov_avg'].interpolate(limit_direction = 'both')
    daily.inter_pol.fillna(method = 'backfill', inplace = True)

    #fill missing NaN in DeviceID
    daily['device_id'].fillna(method = 'bfill', inplace=True)

    #fill Temperature NaN with average
    daily.Temperature.fillna(value = df.Temperature.mean(), inplace = True)
    #fill Tilt NaNs with mean
    daily.Tilt.fillna(value = daily.Tilt.mean(), inplace = True)

    return df, daily



