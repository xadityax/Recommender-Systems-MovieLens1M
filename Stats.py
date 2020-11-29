# -*- coding: utf-8 -*-
"""
Created on sat Nov 21 18:09:37 2020

@author: Aditya
"""

def for_getting_metrics(for_train_reconstruction, Data_test):
    """
    It calculates root mean squared error, spearman rank correlation coeffecient and the precision on top k 

    Parameters
    ----------
    for_train_reconstruction :
        train data
        
    Data_test : 
        test data

    Returns
    -------
    RMSE : 
        root mean squared error
        
    SRC :
        spearman rank correlation coeffecient
        
    PRECISION_TOP_K : 
        precision on top k

    """
    users_number = len(for_train_reconstruction)
    items_number = len(for_train_reconstruction[0])
    Error_squared, num_test = 0, 0
    for user in range(users_number):
        for item in range(items_number):
            if Data_test[user, item] == 0:
                continue
            else:
                Error_squared += (Data_test[user, item] - for_train_reconstruction[user, item])**2
                num_test += 1
    
    SRC = 1 - ((6 * Error_squared) / (num_test * (num_test**2 - 1)))
    RMSE = (Error_squared / num_test) ** 0.5

    PRECISION_TOP_K = 0
    K = 10
    THRESHOLD = 3.5
    num_movies_rated = {}
    i = 0
    for user in Data_test:
        num_movies_rated[i] = user[user > 0].size
        i = i + 1
    i = 0
    all_precisions = []
    for user in for_train_reconstruction:
        if num_movies_rated[i] < K:
            i = i + 1
            continue
        top_k_indices = (-user).argsort()[:K]
        top_k_values = [(index, user[index]) for index in top_k_indices]
        recommended = []
        for (index, user[index]) in top_k_values:
            if(user[index] >= 3.5):
                recommended.append((index, user[index]))
                if len(recommended) == K:
                    break
        count = 0
        for tup in recommended:
            if Data_test[i][tup[0]] >= THRESHOLD:
                count = count + 1
        if len(recommended) > 0:
            precision = count/len(recommended)
            all_precisions.append(precision)
        i = i + 1
    PRECISION_TOP_K = sum(all_precisions) / len(all_precisions)
    return RMSE, SRC, PRECISION_TOP_K
