# -*- coding: utf-8 -*-
"""
Created on sat Nov 21 18:09:37 2020

@author: Aditya
"""
"""
    A Recommender System model based on the Singular Value Decomposition concepts.

    The 0 values in each user row are replaced by the mean rating of each user.
    SVD factorizes the utility matrix into U(m x m), Sigma(m X n) and V-transpose(n X n)
    Dimensionality reduction reduces the dimensions of each matrix to k dimensions.
    The dot product U.Sigma.vt in the reduced form gives the prediction matrix.
    U is an m X m unitary matrix.
    Sigma is an m X n rectangular diagonal matrix, with each diagonal element as the
    singular values of the utility matrix.
    vt is an n X n unitary matrix.
 """
import numpy as np
from math import sqrt
from Colabs import for_corrMatrix_building
from Colabs import basic_collaborative
from Stats import for_getting_metrics
import time


def building_matrix_svd_for(g):
    """
    Normalizes the Utility matrix consisting of users, movies and their ratings by
    replacing 0s in a row by their row mean.
    Performs SVD on the normalized utility matrix and factorizes it into u, vt and sigma
    
    Parameters
    ----------
    g : 
        train data

    Returns
    -------
    u : 
        m X n unitary matrix
        
    vt :
        n X n unitary matrix
        
    sigma :
        m X n rectangular diagonal matrix

    """
    at = np.transpose(g)
    a_at = np.matmul(g, at)
    user_for_nb = g.shape[0]
    movie_for_nb = g.shape[1]
    at_a = np.matmul(at, g)
    del g
    del at
    u_eigenvalue, u_eigenvector = np.linalg.eigh(a_at)
    v_eigenvalue, v_eigenvector = np.linalg.eigh(at_a)
    u_pos_eigen = []
    v_pos_eigen = []
    for val in u_eigenvalue.tolist():
        if(val > 0):
            u_pos_eigen.append(val)
    for val in v_eigenvalue.tolist():
        if(val > 0):
            v_pos_eigen.append(val)
    u_pos_eigen.reverse()
    v_pos_eigen.reverse()
    u_eigen_root = [sqrt(val) for val in u_pos_eigen]
    u_eigen_root = np.array(u_eigen_root)
    masig = np.diag(u_eigen_root)
    len_masig = masig.shape[0]
    ut = np.zeros(shape = (len_masig, user_for_nb))
    vt = np.zeros(shape = (len_masig, movie_for_nb))
    i = 0
    for val in u_pos_eigen:
        ut[i] = u_eigenvector[u_eigenvalue.tolist().index(val)]
        i = i + 1
    i = 0
    for val in v_pos_eigen:
        vt[i] = v_eigenvector[v_eigenvalue.tolist().index(val)]
        i = i + 1
    u = np.transpose(ut)
    del ut
    return u, vt, masig
    

def energy_90_top(u, vt, masig):
    """
    Performs SVD with 90% retained energy on the normalized utility matrix and factorizes it into u, vt and sigma

    Parameters
    ----------
    u : 
       m X n unitary matrix calculated from svd.
       
    vt :
        n X n unitary matrix calculated from svd.       
        
    masig : 
        m X n rectangular diagonal matrix calculated from svd  with each diagonal element as the
        singular values

    Returns
    -------
    new_u :
        new m X n unitary matrix with 90% energy
        
    new_vt : 
        new n X n unitary matrix with 90% energy
        
    new_sigma : 
        new m X n rectangular diagonal matrix with 90% energy

    """
    len_masig = masig.shape[0]
    sum_tot = 0
    for_eigen_req = np.zeros(len_masig)
    for i in range(len_masig):
        sum_tot += masig[i][i] * masig[i][i]
    tot_curr = 0
    for i in range(len_masig):
        tot_curr += masig[i][i] * masig[i][i]
        for_eigen_req[i] = masig[i][i]
        if (tot_curr/sum_tot) >= 0.9:
            i = i + 1
            break
    for_eigen_req = for_eigen_req[for_eigen_req > 0]
    new_sigma = np.diag(for_eigen_req)
    new_u = np.transpose(np.transpose(u)[:new_sigma.shape[0]])
    new_vt = vt[:new_sigma.shape[0]]
    return new_u, new_vt, new_sigma


def main():
    
    K = 50
    data_train = np.load('train.npy')
    data_test = np.load('test.npy') 
    t0 = time.process_time() 
    # REMOVE BELOW FOR FASTER PERFORMANCE ON MULTIPLE RUNS, AND UNCOMMENT ABOVE
    u, vt, masig = building_matrix_svd_for(data_train)
    space_svd_users = np.matmul(u, masig)
    svd_corrMatrix = for_corrMatrix_building(space_svd_users, 'svd_correlation_matrix.npy')
    of_result_svd = basic_collaborative(data_train, data_test, svd_corrMatrix, K)
    RMSE_svd, SRC_svd, precisionTopK_svd = for_getting_metrics(of_result_svd, data_test)
    del of_result_svd
    del svd_corrMatrix
    t1 = time.process_time()
    print('SVD:     RMSE = {}; SRC = {}; Precision on top K = {}; time taken = {}'.format(RMSE_svd, SRC_svd, precisionTopK_svd, t1-t0))
    t2 = time.process_time()
    # REMOVE BELOW FOR FASTER PERFORMANCE ON MULTIPLE RUNS, AND UNCOMMENT ABOVE
    new_u, new_vt, new_sigma = energy_90_top(u, vt, masig)
    users_in_svd_90_space = np.matmul(new_u, new_sigma)
    svd_90_corrMatrix = for_corrMatrix_building(users_in_svd_90_space, 'svd_90_correlation_matrix.npy')
    result_svd_90 = basic_collaborative(data_train, data_test, svd_90_corrMatrix, K)
    RMSE_svd_90, SRC_svd_90, precisionTopK_svd_90 = for_getting_metrics(result_svd_90, data_test)
    del result_svd_90
    del svd_90_corrMatrix
    t3 = time.process_time()
    print('SVD 90%: RMSE = {}; SRC = {}; Precision on top K = {}; time taken = {}'.format(RMSE_svd_90, SRC_svd_90, precisionTopK_svd_90, t3-t2))
    

if __name__ == '__main__':
    main()
