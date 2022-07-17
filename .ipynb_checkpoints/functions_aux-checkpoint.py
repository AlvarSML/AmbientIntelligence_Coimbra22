import zipfile
import os
import glob
import re
import pandas as pd
import numpy as np
import struct
import math
import statistics as stats
from colorama import Fore, Back, Style
import geopandas as gpd
from sqlalchemy import create_engine
from geoalchemy2 import Geometry
from scipy import stats as sc_stats
from scipy.signal import find_peaks
from numpy.fft import fft, ifft

##### File transformation #####

def unzip_files(folderIn = "./InputFiles", folderOut = "./OutputFiles"):
    """
    Unzips all de files in a given folder to another
    args:
    - folderIn -- input folder 
    - folderOut -- output folder
    """
    files = os.listdir(folderIn)
    for file in files:
        name, extension = os.path.splitext(file)
        if (extension == ".zip"):
            with zipfile.ZipFile((folderIn+"/"+file), 'r') as zip_ref:
                zip_ref.extractall((folderOut+"/"+name))

def get_sensor_files(folder = "./OutputFiles/sensemycity_46008_BUS_USER469"):
    """
    Gets a list of sensor files inside a folder
    args:
    - folder -- folder containing sensemycity files
    returns:
    - list of files
    """
    files = glob.iglob(folder+"/SenseMyCity/Session*/*.csv",recursive=True)
    return files

def read_sensors(files):
    """
    Obtains sensors dataframes from files
    args:
    - files -- lists of files
    returns
    - set of sensor pandas dataframes: acc, gyro, gps, mag, cell
    """
    
    pattern = '[\w-]+?(?=\.)'
    acc = gyro = gps = None
    for file in files:
        print("Processing file: "+file)
        
        
        name = re.search(pattern, file).group()
        if "accelerometer" in name:
            acc = pd.read_csv(file,sep=';')
        elif "gyroscope" in name:
            gyro = pd.read_csv(file,sep=';')
        elif "gps" in name:
            gps = pd.read_csv(file,sep=';')
        
    if (gyro is None):
        gyro = pd.DataFrame()
        print(Fore.RED + file + ": No tiene GYRO")
        print(Style.RESET_ALL)
        
    if (acc is None):
        acc = pd.DataFrame()
        print(Fore.RED +file + ": No tiene ACC")
        print(Style.RESET_ALL)
        
    if (gps is None):
        acc = pd.DataFrame()
        print(Fore.RED +file + ": No tiene GPS")
        print(Style.RESET_ALL)
    

    return acc,gyro,gps

##### Data pre-process #####

def clean_hex(dataframe,columns):
    """
    DEPRECATED
    Deletes the scape characters in a hex array
    """
    for col in columns:
        dataframe[col] = dataframe[col].str[2:]
        
def get_arr_acc(sethex):
    """
    Obtains a array of floats for acceleration from a hex 2B/value string
    args:
    - sethex -- set of hex values
    returns:
    - list of acceletarion values in m/s
    """
    values = []
    for i in range(2,len(sethex),4):
        val = int(sethex[i:i+4],base=16)

        if val > pow(2,15):
            val = val-pow(2,16)
        
        values.append(val/256)
    return values

def get_acc_df(acc):
    """
    Generates a dataframe with the important data of the dataframe
    args:
    - acc -- acceleration dataframe
    returns
    - transformed dataframe
    """
    cols_acc = ["idsession","second","accx","accy","accz"]
    data = pd.DataFrame(columns=cols_acc)
    for index,row in acc.iterrows():
        row = pd.Series({'idsession':row.session_id, 'second':row.seconds, 'accx':get_arr_acc(row.accx), 'accy':get_arr_acc(row.accy), 'accz':get_arr_acc(row.accz)})
        data = data.append(row,ignore_index = True)   
    return data

##### ACCELEROMETER #####
def get_acc_features(accx,accy,accz):
    """
    Obtanins a series of features from the 3 components of acceleration
    args:
    - accx -- acceleration X axis
    - accy -- acceleration Y axis
    - accz -- acceleration Z axis
    returns:
    - - start second
    - x = sum of components[]
    - z = sum of sqares[]
    - media of x
    - media of z
    - standard deviation
    - 99 pecentile
    """
    x = [] #Suma de componentes menos gravedad
    z = [] #Suma de cuadrados
    for i in range(len(accx)):
        x.append(accx[i] + accy[i] + accz[i] - 9.8)
        z.append(math.sqrt(pow(accx[i],2) + pow(accy[i],2) + pow(accz[i],2))-9.8)        
    return x,z,stats.mean(x),stats.mean(z),np.std(x),np.std(z),np.percentile(x,99),np.percentile(z,99)

def merge_axis_acc(accx,accy,accz):
    """
    Merges the 3 axis into a one dimension array, for linear acceleration data
    args:
    - accx -- acceleration X axis
    - accy -- acceleration Y axis
    - accz -- acceleration Z axis
    returns
    - x = sum of components[]
    - z = sum of sqares[]
    """
    x = [] #Suma de componentes menos gravedad
    z = [] #Suma de cuadrados
    for i in range(len(accx)):
        x.append(accx[i] + accy[i] + accz[i] - 9.8)
        z.append(math.sqrt(pow(accx[i],2) + pow(accy[i],2) + pow(accz[i],2))-9.8)
    return x,z

def merge_axis(accx,accy,accz):
    """
    Merges the 3 axis into a one dimension array, for any data
    args:
    - accx -- acceleration X axis
    - accy -- acceleration Y axis
    - accz -- acceleration Z axis
    returns
    - x = sum of components[]
    - z = sum of sqares[]
    """
    x = [] #Suma de componentes menos gravedad
    z = [] #Suma de cuadrados
    for i in range(len(accx)):
        x.append(accx[i] + accy[i] + accz[i])
        z.append(math.sqrt(pow(accx[i],2) + pow(accy[i],2) + pow(accz[i],2)))
    return x,z
        
##### GYROSCOPE #####

def get_gyro_df(gyro):
    """
    Generates a dataframe with the important data of the dataframe
    args:
    - gyro -- acceleration dataframe
    returns
    - transformed dataframe
    """
    cols_acc = ["idsession","second","accx","accy","accz"]
    data = pd.DataFrame(columns=cols_acc)
    for index,row in gyro.iterrows():
        row = pd.Series({'idsession':row.session_id, 'second':row.seconds, 'accx':get_arr_acc(row.gyrx), 'accy':get_arr_acc(row.gyry), 'accz':get_arr_acc(row.gyrz)})
        data = data.append(row,ignore_index = True)
    return data

##### WINDOW EXTRACTION #####

def window_extraction(dataframe,seconds=10,offset=0):
    """
    Generates an array of segmens of n length, the length is not guaranteed for the last element
    args:
    - dataframe -- df of origin
    - seconds -- length of the segment, default = 10
    - offset -- start point of the segments, default = 0
    returns:
    - arr of segments
    """
    segments = []
    for i in range(offset,len(dataframe),seconds):
        segments.append(dataframe[i:i+seconds])
    return segments

def get_window_features(window):
    """
    Obtains features from a window of acceleration data
    args:
    - window -- single window of n length
    returns:
    - processed features
    """
    x,z,avgx,avgz,stdx,stdz,p99x,p99z = get_acc_features(window["accx"].sum(),window["accy"].sum(),window["accz"].sum())
    row={"second":window["second"].iloc[0],"sum_values":x,"square_values":z,"media_x":avgx,"media_z":avgz,"std_x":stdx,"std_z":stdz,"p99_x":p99x,"p99_z":p99z}
    return row
    

# generate dataset of windows
    """
    Generated dataframe from windows of features
    args:
    - windows list
    returns:
    - dataframe of features
    """
def dataset_generate(windows):
    cols_dat = {"second","sum_values","square_values","media_x","media_z","std_x","std_z","p99_x","p99_z"}
    df = pd.DataFrame(columns=cols_dat)
    for w in windows:
        window = get_window_features(w)
        df = df.append(window,ignore_index=True)
    return df

##### VISUAL #####
import matplotlib.pyplot as plt
def axis_plot(x,y,z):
    time = range(0,len(x))

    plt.subplot(3, 1, 1)
    plt.plot(time, x, '.-')
    plt.title('Acceleration by axis')
    plt.ylabel('X acceleration')

    plt.subplot(3, 1, 2)
    plt.plot(time, y, '.-')
    plt.xlabel('time (s)')
    plt.ylabel('Y acceleration')

    plt.subplot(3, 1, 3)
    plt.plot(time, z, '.-')
    plt.xlabel('time (s)')
    plt.ylabel('Z acceleration')
    
##### DATABASE #####
def instant_lines(gps,acc,gyro,cols,currid):
    """
    Returns a line of data for each secod of recordings
    args:
    gps - dataset of the gps file
    acc - dataset from the accelerometer file
    gyro - dataset from the gyroscope data
    cols - column name, array
    currid - id of the files
    returns:
    data - dataset with all the columns    
    """
    accdata = get_acc_df(acc)
    datagy = get_gyro_df(gyro)
    data = pd.DataFrame(columns=cols)
    for index,line in gps.iterrows():
        seconds = line.seconds
        accrow = accdata[accdata.second == seconds]
        gyrorow = datagy[datagy.second == seconds]
        # Ver si hay una linea para ese segundo
        if len(accrow) == 0:
            print(Back.RED+"!! No hay acc data para el segundo: "+ str(seconds))
            print(Style.RESET_ALL)
        elif len(gyrorow) == 0 :
            print(Back.RED+"!! No hay gyro data para el segundo: "+ str(seconds))
            print(Style.RESET_ALL)
        else:
            row = {"trajectory_id":currid,
                   "lat":line["lat"],
                   "lon":line["lon"],
                   "gps_position":line["geo"],
                   "altitude":line["alt"],
                   "gps_accuracy":line["acc"],
                   "speed":line["speed"],
                   "nsats":line["nsats"],
                   "accx":accrow["accx"].tolist()[0],
                   "accz":accrow["accz"].tolist()[0],
                   "accy":accrow["accy"].tolist()[0],
                   "gyrx":gyrorow["accx"].tolist()[0],
                   "gyrz":gyrorow["accz"].tolist()[0],
                   "gyry":gyrorow["accy"].tolist()[0],
                   "second":line.seconds}
            row = pd.Series(row)
            data = data.append(row,ignore_index = True)
    return data
    
def insert_trayectorie(conn,data):
    """
    Inserts a trajectorie into the database, only the tag and times
    args:
    conn - connection to the db
    data - dataset with the metadata
    returns:
    idret - id generated for the trajectory
    """
    cols = "tag,start_timestamp,elapsed_seconds"
    sql = "INSERT INTO trayectories({0}) VALUES({1}) RETURNING session_id;"
    try:
        cur = conn.cursor()
        values = str(data["session_id"])+",'"+data["tag"]+"','"+str(data["start_timestamp"])+"',"+str(data["elapsed_seconds"])
        columns = str(list(data.keys())).replace("[","").replace("]","").replace("'","")

        sql = sql.format(columns,values)
        #print(sql)

        cur.execute(sql)    
        idret = cur.fetchone()[0]

    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        conn.commit()
        cur.close()
    return idret

def insert_instant(conn,data):
    """
    Inserts a trajectorie into the database, only the tag and times
    args:
    conn - connection to the db
    data - dataset with all the columns
    """
    conn_string = 'postgresql://postgres:root@127.0.0.1/proyect_ai'  
    engine = create_engine(conn_string)
    conn2 = engine.connect()
    gdf = gpd.GeoDataFrame(data)
    gdf = gdf.set_geometry(gpd.points_from_xy(gdf.lon,gdf.lat))
    gdf = gdf.rename(columns={'geometry':'geom'}).set_geometry('geom')
    gdf.crs = 'epsg:4326'

    try:
        gdf.to_postgis("instant",engine,if_exists='append',dtype={'geom': Geometry(geometry_type='POINT', srid= 4326)})
    except (Exception) as error:
        print(error)
        
##### FEATURE EXTRACTION #####

def stat_features(window=[],label="stat"):
    """
    Generates a dictionary of statistical features over a window of data, must contain arrays
    args:
    window - bidimensional array of any recording
    label - base to name the columns
    returns:
    dic - dictionary with all the features
    """
    dic = {}
    dic[label+"_avg"] = np.mean(window)
    dic[label+"_std"] = np.std(window)
    dic[label+"_adif"] = np.mean(np.absolute(window - np.mean(window)))
    dic[label+"_min"] = np.min(window)
    dic[label+"_max"] = np.max(window)
    dic[label+"_mmdiff"] = np.max(window) - np.min(window) 
    dic[label+"_median"] = np.median(window)
    #print("window: "+label+" -> "+str(window))
    dic[label+"_quart"] = np.percentile(window, 75) - np.percentile(window, 25)
    dic[label+"_neg"] = sum(i < 0 for i in window)
    dic[label+"_pos"] = sum(i > 0 for i in window)
    
    dic[label+"_negper"] = sum(i < 0 for i in window) / len(window)
    dic[label+"_posper"] = sum(i > 0 for i in window) / len(window)
    
    dic[label+"_aavg"] = np.sum(window > np.mean(window))
    dic[label+"_peaks"] = len(find_peaks(window)[0])
    dic[label+"_skew"] = sc_stats.skew(window)
    return dic
    
def get_amplitudes_oneside(signal,cut):
    """
    Returns one side of the symetric frequencies domain with a cut as a percentage
    """
    fft_sig = np.abs(fft(signal))    
    return fft_sig[1:math.floor(len(signal)/2*cut)]
