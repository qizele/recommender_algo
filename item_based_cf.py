#data readin
import numpy as np
import pandas as pd
from scipy import sparse
import math
import random

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

#item-based CF recommender (KNN)
def item_based_cf(data_file, N, null_value):
    #create a new matrix by imputing the null value
    data_file_imputed=data_imputation(data_file, null_value=null_value, impute_method='mean', direction='item')
    data_file_predict=data_file_imputed.copy()
    #Calculate consine similarity and predict for each items (for small RAM)
    for i in data_file_imputed.columns:
        # calculate the cosine similarity for two vector
        cos_list=[]
        for j in data_file_imputed.columns:
            v1=data_file_imputed.loc[:,i]
            v2=data_file_imputed.loc[:,j]
            prod = np.dot(v1, v2)
            len1 = math.sqrt(np.dot(v1, v1))
            len2 = math.sqrt(np.dot(v2, v2))
            cos_list.append(prod / (len1 * len2))
        cosine_similarity=pd.DataFrame(cos_list, index=data_file_imputed.columns, columns=['similarity'])
        cosine_similarity.sort_values(by=['similarity'], ascending=False, inplace=True)
        #Top N    
        KNN_similarity=cosine_similarity[1:N+1]
        #Predict Values in User i
        KNN_matrix=data_file_imputed.loc[:,list(KNN_similarity.index)]
        nom=KNN_matrix.multiply(np.array(list(KNN_similarity['similarity'])), axis=1).sum(axis=1)
        denom=KNN_similarity['similarity'].abs().sum()
        pred=nom.divide(denom)
        data_file_predict.loc[:,i]=pred

    return data_file_predict

item_based_predict=item_based_cf(sample_data, 20, 99)

#Normalized Mean Absolute Error
def cf_nmae(data_file, null_value, data_file_predict, vmax, vmin):
    #Here null value in origiinal dataset should be covert to np.nan
    df_out=data_file.replace(to_replace=null_value, value=np.nan)
    pred_error=abs(data_file_predict-df_out)
    nmae=pred_error.sum().sum()/(pred_error.count().sum()*(vmax-vmin))
    return nmae

nmae=cf_nmae(sample_data, 99, item_based_predict, 10, -10)
print(nmae)
