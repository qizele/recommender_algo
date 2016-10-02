#data readin
import numpy as np
import pandas as pd
from scipy import sparse
import math
import random
from sklearn.decomposition import PCA
from scipy import linalg
from sklearn.cluster import KMeans 

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

#PCA CF recommender (in user dimension)
def pca_cf(data_file, N, M, null_value):
    #N is the PCA dimension, should be less than 10 normally
    #M is the # of clusters
    #create a new matrix by imputing the null value
    data_file_imputed=data_imputation(data_file, null_value=null_value, impute_method='mean', direction='item')
    df_null=data_file.replace(to_replace=null_value, value=np.nan)
    item_mean=df_null.mean(axis=0)
    item_std=df_null.std(axis=0)
    data_file_normalized=data_file_imputed.sub(item_mean, axis=1).divide(item_std, axis=1)
    data_file_predict=data_file_imputed.copy()
    corr_matrix=1/len(data_file_normalized)*data_file_normalized.T.dot(data_file_normalized)

    #PCA decomposition with N components
    pca = PCA(n_components=N)
    pca.fit(corr_matrix)
    pca_comp=pca.components_
    x=data_file_imputed.dot(pca_comp.T)

    #perform Kmeans clustering based on N components in each user
    kmeans = KMeans(n_clusters=M, random_state=0).fit(x)
    cluster_info=pd.DataFrame(kmeans.labels_, index=data_file_imputed.index, columns=['cluster'])

    #make recommendation based on the given cluster
    for i in data_file_predict.index:
        cluster_number=cluster_info.loc[i,'cluster']
        users_in_same_cluster=cluster_info[(cluster_info.cluster==cluster_number) & (cluster_info.index!=i)].index
        #use average value for recommendation based on the users in given cluster
        pred=data_file_imputed.loc[users_in_same_cluster, :].mean(axis=0)
        data_file_predict.loc[i,:]=pred
    return data_file_predict
pca_predict=pca_cf(sample_data, 5, 20, 99)

#Normalized Mean Absolute Error
def cf_nmae(data_file, null_value, data_file_predict, vmax, vmin):
    #Here null value in origiinal dataset should be covert to np.nan
    df_out=data_file.replace(to_replace=null_value, value=np.nan)
    pred_error=abs(data_file_predict-df_out)
    nmae=pred_error.sum().sum()/(pred_error.count().sum()*(vmax-vmin))
    return nmae

nmae=cf_nmae(sample_data, 99, pca_predict, 10, -10)
print(nmae)
