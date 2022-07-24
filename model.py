# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 17:31:42 2022

@author: Moksha Sri
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import mysql.connector
from autots import AutoTS
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import statsmodels.api as sm

################## SQL connection ##########################
def mysqlc():
    mydb = mysql.connector.connect(host="localhost",user="root",password="Amma@143",database = "ds70")
    mycursor = mydb.cursor()

    mycursor.execute("select datum,M01AB,M01AE,N02BA,N02BE,N05B,N05C,R03,R06 from salesdaily")
    result = mycursor.fetchall()

# convert to dataframe
    data = pd.DataFrame(result)

#Define column names
    data.columns =['datum', 'M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03','R06']
    
    return mysqlc()

############### Pre processing the Data ###################################
def preprocessing():
    #Convert to datetime
    data["datum"]=pd.to_datetime(data["datum"]) 
   #  Dropping Duplicate data 
    data.drop_duplicates()

   #  Dropping missing data 
    data.isna().sum()
    data.dropna(axis =0,inplace= True)
    
    # outlier treatment
    col = data.columns[1:]

     # finding IQR
    IQR = data.quantile(0.75)-data.quantile(0.25)
    lower_limit = data.quantile(0.25) - (IQR * 1.5)
    upper_limit = data.quantile(0.75) + (IQR * 1.5)

for i in col:
     # if outlier < lower_limit1 replace with lower_limit1 , if outlier > upper_limit1 replace with upper_limit1 else no change
    out_replace = np.where(data[i]>upper_limit[i],upper_limit[i],np.where(data[i]<lower_limit[i],lower_limit[i],data[i]))
     # create dataframe of replaced data
data[i] = pd.DataFrame(out_replace)

    return preprocessing()
################ Model building the data with Auto TS #######################

def model_ml(ab):
    
    
    
    
    train = data[-365:]
    model_M01AB = AutoTS(forecast_length=7,
                     frequency='infer',
                     prediction_interval=0.95,
                     ensemble=None,
                     model_list="fast",
                     transformer_list="fast",
                     drop_most_recent=1,
                     max_generations=4,
                     num_validations=2,
                     validation_method="backwards")

    model_M01AB = model_M01AB.fit(train,
                              date_col="datum" ,
                              value_col='M01AB',
                              id_col=None)

## saving the model into folder

    pickle.dump(model_M01AB,open('model_M01AB.pkl','wb'))

    model_M01AB = pickle.load(open('model_M01AB.pkl','rb'))

#########  M01AE     ############################

    model_M01AE = AutoTS(forecast_length=7,
                     frequency='infer',
                     prediction_interval=0.95,
                     ensemble=None,
                     model_list="fast",
                     transformer_list="fast",
                     drop_most_recent=1,
                     max_generations=4,
                     num_validations=2,
                     validation_method="backwards")

    model_M01AE = model_M01AE.fit(train,
                              date_col="datum" ,
                              value_col='M01AE',
                              id_col=None)

    pickle.dump(model_M01AE,open('model_M01AE.pkl','wb'))

    model_M01AE = pickle.load(open('model_M01AE.pkl','rb'))


########### N02BA  #############

    model_N02BA = AutoTS(forecast_length=7,
                     frequency='infer',
                     prediction_interval=0.95,
                     ensemble=None,
                     model_list="fast",
                     transformer_list="fast",
                     drop_most_recent=1,
                     max_generations=4,
                     num_validations=2,
                     validation_method="backwards")

    model_N02BA = model_N02BA.fit(train,
                              date_col="datum" ,
                              value_col='N02BA',
                              id_col=None)

    pickle.dump(model_N02BA,open('model_N02BA.pkl','wb'))
    
    model_N02BA = pickle.load(open('model_N02BA.pkl','rb'))
    
    ########## N02BE #############################################
    
    model_NO2BE = AutoTS(forecast_length=7,
                         frequency='infer',
                         prediction_interval=0.95,
                     ensemble=None,
                     model_list="fast",
                     transformer_list="fast",
                     drop_most_recent=1,
                     max_generations=4,
                     num_validations=2,
                     validation_method="backwards")

    model_NO2BE = model_NO2BE.fit(train,
                                  date_col="datum" ,
                                  value_col='N02BE',
                                  id_col=None)
    
    
    pickle.dump(model_NO2BE,open('model_N02BE.pkl','wb'))
    
    model_N02BE = pickle.load(open('model_N02BE.pkl','rb'))

#################### N05B ###########################################

    model_NO5B = AutoTS(forecast_length=7,
                         frequency='infer',
                         prediction_interval=0.95,
                         ensemble=None,
                         model_list="fast",
                         transformer_list="fast",
                         drop_most_recent=1,
                         max_generations=4,
                         num_validations=2,
                         validation_method="backwards")
    
    model_NO5B = model_NO5B.fit(train,
                                  date_col="datum" ,
                                  value_col='N05B',
                                  id_col=None)
    
    pickle.dump(model_NO5B,open('model_N05B.pkl','wb'))
    
    model_N05B = pickle.load(open('model_N05B.pkl','rb'))


################# N05C  ##########################

    model_NO5C = AutoTS(forecast_length=7,
                         frequency='infer',
                         prediction_interval=0.95,
                         ensemble=None,
                         model_list="fast",
                         transformer_list="fast",
                         drop_most_recent=1,
                         max_generations=4,
                         num_validations=2,
                         validation_method="backwards")
    
    model_NO5C = model_NO5C.fit(train,
                                  date_col="datum" ,
                                  value_col='N05C',
                                  id_col=None)
    
    pickle.dump(model_NO5C,open('model_N05C.pkl','wb'))
    
    model_N05C = pickle.load(open('model_N05C.pkl','rb'))
    
    ####################  R03  ####################################
    
    model_R03 = AutoTS(forecast_length=7,
                         frequency='infer',
                         prediction_interval=0.95,
                         ensemble=None,
                         model_list="fast",
                         transformer_list="fast",
                         drop_most_recent=1,
                         max_generations=4,
                         num_validations=2,
                         validation_method="backwards")
    
    model_R03 = model_R03.fit(train,
                                  date_col="datum" ,
                                  value_col='R03',
                                  id_col=None)

    pickle.dump(model_R03,open('model_R03.pkl','wb'))
    
    model_R03 = pickle.load(open('model_R03.pkl','rb'))
    
    ##################### R06 ##################################
    
    model_R06 = AutoTS(forecast_length=7,
                         frequency='infer',
                         prediction_interval=0.95,
                         ensemble=None,
                         model_list="fast",
                         transformer_list="fast",
                         drop_most_recent=1,
                         max_generations=4,
                         num_validations=2,
                         validation_method="backwards")
    
    model_R06 = model_R06.fit(train,
                                  date_col="datum" ,
                                  value_col='R06',
                                  id_col=None)
    
    
    pickle.dump(model_R06,open('model_R06.pkl','wb'))
    
    model_R06 = pickle.load(open('model_R06.pkl','rb'))

    return model_M01AB,model_M01AE,model_N02BA,model_N02BE,model_N05B,model_N05C,model_R03,model_R06

pipe = Pipeline(['preprocess',preprocessing(),
                 'model_ml',model_ml("M01AB"),
                 'model_ml',model_ml("M01AE"),
                 'model_ml',model_ml("N02BA"),
                 'model_ml',model_ml("N02BE"),
                 'model_ml',model_ml("N05B"),
                 'model_ml',model_ml("N05C"),
                 'model_ml',model_ml("R03"),
                 'model_ml',model_ml("R06")])









