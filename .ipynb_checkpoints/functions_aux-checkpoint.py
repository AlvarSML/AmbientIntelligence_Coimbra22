import zipfile
import os
import glob
import re
import pandas as pd
import numpy as np
import struct
import math
import statistics as stats

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
    files = glob.iglob(folder+"/SenseMyCity/*/*.csv",recursive=True)
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
    for file in files:
        name = re.search(pattern, file).group()
        if "accelerometer" in name:
            acc = pd.read_csv(file,sep=';')
        elif "gyroscope" in name:
            gyro = pd.read_csv(file,sep=';')
        elif "gps" in name:
            gps = pd.read_csv(file,sep=';')
        elif "magnetometer" in name:
            mag = pd.read_csv(file,sep=';')
        elif "cellular" in name:
            cell = pd.read_csv(file,sep=';')
    return acc,gyro,gps,mag,cell

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