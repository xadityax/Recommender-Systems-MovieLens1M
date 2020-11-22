# -*- coding: utf-8 -*-
"""
Created on sat Nov 21 18:09:37 2020

@author: Jalaj
"""
"""
    A Recommender System model based on the Collaborative filtering concepts.

    An Item-Item based collaborative filtering is used to find similar items which then
    is used to predict rating a user might give to a movie/item based on the ratings he
    gave to similar items. Also calculates rating deviations of users to the form of
    the mean of ratings to handle strict and generous raters.
 """
import numpy as np
import time
from Stats import for_getting_metrics

def for_corrMatrix_building(data, name_of_file):
    """
    For building correlation matrix
    
    parameters:
        data: 
            training data
            
        name_of_file:
            name of the file to be passed
            
    Return:
        corrMatrix:
            correlation matrix
            
    """
    users_number = len(data)
    corrMatrix = np.corrcoef(data)[:users_number+1, :users_number+1]
    np.save(name_of_file, corrMatrix)
    return corrMatrix



def basic_collaborative(data_train, testData, corrMatrix, K):
    """
    Basic method is used to compute the reconstructed matrix

     Parameters
     ----------
     data_train : 
         data present in train.csv file
    testdata:
        data in test.csv file
     corrMatrix : 
         correlation matrix
     K : 
         amount of most similar k users

     Returns
     -------
     train_reconstructed_y :
         reconstructed matrix

     """
    users_number = len(data_train)
    items_number = len(data_train[0])
    
    train_reconstructed_y = np.zeros((users_number, items_number))
    for userToPredict in range(users_number):
        USERS_closest = (-corrMatrix[userToPredict]).argsort()[:K]
        for item in range(items_number):
            if testData[userToPredict, item] == 0:
                continue
            sum_corr = 0
            for closeUser in USERS_closest:
                if data_train[closeUser, item] != 0:
                    train_reconstructed_y[userToPredict, item] += corrMatrix[userToPredict, closeUser] * data_train[closeUser, item]
                    sum_corr += corrMatrix[userToPredict, closeUser]
            if sum_corr != 0:
                train_reconstructed_y[userToPredict, item] /= sum_corr
    return train_reconstructed_y



def collaborative_baseline(data_train, testData, corrMatrix, K):
    """
    Baseline method is used to compute the reconstructed matrix

    Parameters
    ----------
    data_train : 
        data present in train.csv file
        
    testData :
        data in test.csv file
        
    corrMatrix :
        correlation matrix
    K : 
        amount of most similar k users

    Returns
    -------
    train_reconstructed_y : 
        reconstructed matrix

    """
    users_number = len(data_train)
    items_number = len(data_train[0])
    globalMean = 0
    
    # rating deviation for each user/item
    deviation_rating_user, deviation_rating_item = np.zeros(users_number), np.zeros(items_number)
    
    # number of ratings per user/item
    ratings_number_user, ratings_number_items = np.zeros(users_number), np.zeros(items_number)
    
    for user in range(users_number):
        for item in range(items_number):
            if data_train[user, item] == 0:
                continue
            else:
                deviation_rating_user[user] += data_train[user, item]
                deviation_rating_item[item] += data_train[user, item]
                globalMean += data_train[user, item]
                ratings_number_user[user] += 1
                ratings_number_items[item] += 1
    
    # handle cases where a user/item has not rated/been rated (to avoid divide-by-zero)
    for user in range(users_number):
        if ratings_number_user[user] == 0:
            ratings_number_user[user] = 1
    for item in range(items_number):
        if ratings_number_items[item] == 0:
            ratings_number_items[item] = 1
    
    # calculate global mean and rating deviations
    globalMean /= np.sum(ratings_number_user)
    deviation_rating_user = np.divide(deviation_rating_user, ratings_number_user) # avg rating of any user
    deviation_rating_user -= globalMean # subtract global mean
    deviation_rating_item = np.divide(deviation_rating_item, ratings_number_items) # avg rating of any item
    deviation_rating_item -= globalMean # subtract global mean
    
    # calculate baselines for each user,item pair
    baseline = np.zeros((users_number, items_number))
    for user in range(users_number):
        for item in range(items_number):
            baseline[user, item] = globalMean + deviation_rating_user[user] + deviation_rating_item[item]
    
    # compute reconstructed matrix
    train_reconstructed_y = np.zeros((users_number, items_number))
    for userToPredict in range(users_number):
        USERS_closest = (-corrMatrix[userToPredict]).argsort()[:K]
        for item in range(items_number):
            if testData[userToPredict, item] == 0:
                continue
            sum_corr = 0
            for closeUser in USERS_closest:
                if data_train[closeUser, item] != 0:
                    train_reconstructed_y[userToPredict, item] += corrMatrix[userToPredict, closeUser] * (data_train[closeUser, item] - baseline[closeUser, item])
                    sum_corr += corrMatrix[userToPredict, closeUser]
            if sum_corr != 0:
                train_reconstructed_y[userToPredict, item] /= sum_corr
                train_reconstructed_y[userToPredict, item] += baseline[userToPredict, item]
    return train_reconstructed_y


def main():
    K = 50
    data_train = np.load('train.npy')
    testData = np.load('test.npy')
    
    # UNCOMMENT BELOW AND REMOVE FURTHER BELOW FOR FASTER PERFORMANCE ON MULTIPLE RUNS
    print("[INFO] Building Correlation Matrix")
    try:
        corrMatrix = np.load('correlation_matrix.npy')
    except FileNotFoundError:
        corrMatrix = for_corrMatrix_building(data_train, 'correlation_matrix.npy')
    
    t0 = time.process_time()
    
    # REMOVE BELOW FOR FASTER PERFORMANCE ON MULTIPLE RUNS, AND UNCOMMENT ABOVE
    # corrMatrix = for_corrMatrix_building(data_train, 'correlation_matrix.npy')
    
    print("[INFO] Running")
    reconstructedTrainBasic = basic_collaborative(data_train, testData, corrMatrix, K)
    RMSEbasic, SRCbasic, precisionTopKbasic = for_getting_metrics(reconstructedTrainBasic, testData)
    t1 = time.process_time()
    print('basic:    RMSE = {}; SRC = {}; Precision on top K = {}; time taken = {}'.format(RMSEbasic, SRCbasic, precisionTopKbasic, t1-t0))
    
    # REMOVE BELOW FOR FASTER PERFORMANCE ON MULTIPLE RUNS, AND UNCOMMENT ABOVE
    # corrMatrix = for_corrMatrix_building(data_train, 'correlation_matrix.npy')
    
    Baseline_reconstructed_train = collaborative_baseline(data_train, testData, corrMatrix, K)
    baseline_of_RMSE, baseline_of_SRC, baseline_topk_precision = for_getting_metrics(Baseline_reconstructed_train, testData)
    t2 = time.process_time()
    print('baseline: RMSE = {}; SRC = {}; Precision on top K = {}; time taken = {}'.format(baseline_of_RMSE, baseline_of_SRC, baseline_topk_precision, t2-t1))


if __name__ == '__main__':
    main()
