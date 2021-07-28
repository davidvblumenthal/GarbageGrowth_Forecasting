import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset

#Utlity functions import
import sys
from pathlib import Path
# in jupyter (lab / notebook), based on notebook path
module_path = str(Path.cwd().parents[0] / "src")

if module_path not in sys.path:
    sys.path.append(module_path)
from preprocessing import cleaning_del
from preprocessing import smoothing_fillingNaN

'''funktion aus dem notebook preprocessing_clustering.ipynb
liest alle zeitreihen ein, wandelt diese in numpy arrays um skaliert diese berechnet die Ähnlichkeit mit dynamic time warping
und und kmeans
FUNKTION ÜBERNIMMT KEINE ARGUMENTE - CLUSTER SIND AUF N = 3 FESTGELEGT, DA SICH DIESE ALS AM SINNVOLSTEN ERWIESEN HABEN
erstellt einen Ordner clusters in data/preprocessed und darunter dann 0, 1, 2 in denen die dazugehörigen Zeitreihen einsortiert werden '''

def kmeans_cluster():
    csv_folder = '../data/preprocessed/good_csv/'
    csv_files = [csv for csv in os.listdir(csv_folder) if csv.endswith('.csv')]
    count = 1
    all_series = pd.DataFrame()
    #df_list = list()

    for file in csv_files:
        #print(file)
            
        # import DataFrame
        df = pd.read_csv('../data/preprocessed/good_csv/' + file)
            
        df = cleaning_del(df)
        df, grouped = smoothing_fillingNaN(df)

        #write the interpolated hight into columns
        all_series[file] = grouped['inter_pol']
        count = count + 1
    
    #transform dataframe to numpy array
    time_series = to_time_series_dataset(all_series)
    #turn into float
    time_series = time_series.astype(float)
    #roate array into correct shape
    a = np.rot90(time_series, k = 3)
    a.shape

    #scale values
    train = TimeSeriesScalerMeanVariance().fit_transform(a)
    #initialise cluster algorithm
    dba_km = TimeSeriesKMeans(n_clusters = 3,  metric = "dtw", max_iter = 50, random_state=1)
    #predict clusters and save
    pred = dba_km.fit_predict(train)

    sz = train.shape[1]
    #print sample of the different clusters
    for yi in range(3):
        plt.subplot(3, 3, 4 + yi)
        for xx in a[pred == yi]:
            plt.plot(xx.ravel(), "k-", alpha=.2)
        plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
        plt.xlim(0, sz)
        plt.ylim(-4, 4)
        plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
                transform=plt.gca().transAxes)
        if yi == 1:
            plt.title("DBA $k$-means")

    cluster_count = [0, 1, 2]
    path_name = '../data/preprocessed/clusters/'

    for i in cluster_count:
        dirName = path_name + str(i)
        #creating folders based on clusters and export files
        # Create target Directory if don't exist
        if not os.path.exists(dirName):
            os.mkdir(dirName)
            print("Directory " , dirName ,  " Created ")
        else:    
            print("Directory " , dirName ,  " already exists") 

        print('Cluster ' + str(i))

        for series, clust in clusters.items():

            if clust == i:          
                df = pd.read_csv('../data/preprocessed/good_csv/' + series)

                df = cleaning_del(df)
                df, grouped = smoothing_fillingNaN(df)

                df.to_csv(dirName + '/' + series)
                grouped.to_csv(dirName + '/g_' + series)

                plt.figure(figsize=(30,8))
                plt.ylim((0,200))
                plt.title(series)

                plt.xticks(fontsize=8, rotation=90)
                plt.yticks(fontsize=10, fontweight='bold')
                plt.plot(grouped['inter_pol'])

                plt.show()