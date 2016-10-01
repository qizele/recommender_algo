#data readin
import numpy as np
import pandas as pd
from scipy import sparse
import math
import random
from sklearn.utils.extmath import randomized_svd
from scipy import linalg

def data_readin(filename, sep=','):
    matrix=np.genfromtxt(filename,delimiter=',',dtype=None, names=None)
    outfile=pd.DataFrame(matrix, dtype=float)
    return outfile

rating_1= data_readin(str('F:\Careers\Showcase\joke_rating\jester-data-1.csv'))
rating_2= data_readin(str('F:\Careers\Showcase\joke_rating\jester-data-2.csv'))
rating_3= data_readin(str('F:\Careers\Showcase\joke_rating\jester-data-3.csv'))

rating_fnl=pd.concat([rating_1, rating_2, rating_3], axis=0, ignore_index=True)
rating_fnl.rename(columns=lambda x: 'i'+str(x+1), inplace=True)
rating_fnl.rename(index=lambda x: 'u'+str(x+1), inplace=True)

def data_sampling(datafile, ratio):
    nrows=ratio
    rows=random.sample(list(datafile.index),nrows)
    sample_data=datafile.loc[rows,:]
    return sample_data

sample_data=data_sampling(rating_fnl, 2000)

#data imputation (cold start)
def data_imputation(df_name, null_value, impute_method='mean', direction='user'):
    #first change the particular value to np.nan
    df_out=df_name.replace(to_replace=null_value, value=np.nan)
    if impute_method=='mean':
        if direction=='user':
            df_out=df_out.T.fillna(df_out.mean(axis=1)).T
        if direction=='item':
            df_out.fillna(df_out.mean(), inplace=True)
    if impute_method=='median':
        if direction=='user':
            df_out=df_out.T.fillna(df_out.median(axis=1)).T
        if direction=='item':
            df_out.fillna(df_out.median(), inplace=True)
    return df_out

#SVD CF recommender (in user dimension)
def svd_cf(data_file, N, null_value):
    #N should be less than min(N_item, N_user)
    #create a new matrix by imputing the null value
    data_file_imputed=data_imputation(data_file, null_value=null_value, impute_method='mean')
    df_null=data_file.replace(to_replace=null_value, value=np.nan)
    user_mean=df_null.mean(axis=1)
    data_file_imputed=data_file_imputed.sub(user_mean, axis=0)
    data_file_predict=data_file_imputed.copy()

    #SVD decomposition with N components
    U, S, V = randomized_svd(data_file_imputed, n_components=N, n_iter=10, random_state=None)
    Sigma=np.diag(S)
    Sigma_sqrt=linalg.sqrtm(Sigma)
    left=np.dot(U, Sigma_sqrt)
    right=np.dot(Sigma_sqrt, V)
    for i in range(0,len(left)):
        pred=user_mean.iloc[i]+np.dot(left[i,:], right)
        data_file_predict.iloc[i]=pred
    return data_file_predict

svd_predict=svd_cf(sample_data, 20, 99)

#Normalized Mean Absolute Error
def cf_nmae(data_file, null_value, data_file_predict, vmax, vmin):
    #Here null value in origiinal dataset should be covert to np.nan
    df_out=data_file.replace(to_replace=null_value, value=np.nan)
    pred_error=abs(data_file_predict-df_out)
    nmae=pred_error.sum().sum()/(pred_error.count().sum()*(vmax-vmin))
    return nmae

nmae=cf_nmae(sample_data, 99, svd_predict, 10, -10)
print(nmae)
