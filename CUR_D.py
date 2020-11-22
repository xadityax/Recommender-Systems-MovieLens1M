# -*- coding: utf-8 -*-
"""
Created on sat Nov 21 18:09:37 2020

@author: Jalaj

"""

import numpy as np
from math import sqrt
from SV_D import building_matrix_svd_for
from SV_D import energy_90_top
from Stats import for_getting_metrics
import timeit
from sklearn.metrics import mean_squared_error

def C_1_U_1_R(k):
    """
    performs the CUR decomposition on the user matrix and stores them as numpy arrays

    Parameters
    ----------
    k :
        number of rows and columns taken
        

    Returns
    -------
    None.

    """
    matrix_of_user_movie = np.load('train.npy')
    ssq_sum = 0 #sum of squares of all elements
    users_of_numbers = matrix_of_user_movie.shape[0]
    movies_of_numbers = matrix_of_user_movie[0].size
    for i in range(users_of_numbers):
        for j in range(movies_of_numbers):
            ssq_sum = ssq_sum + matrix_of_user_movie[i][j]*matrix_of_user_movie[i][j]
    users_probabilities = []
    movies_probabilities = []
    for i in range(users_of_numbers):
        ssq_of_row = 0 #
        for j in range(movies_of_numbers):
            ssq_of_row = ssq_of_row + matrix_of_user_movie[i][j]*matrix_of_user_movie[i][j]
        users_probabilities.append(ssq_of_row/ssq_sum)#computing user probabilities
    for j in range(movies_of_numbers):
        ssq_of_column = 0
        for i in range(users_of_numbers):
            ssq_of_column = ssq_of_column + matrix_of_user_movie[i][j]*matrix_of_user_movie[i][j]
        movies_probabilities.append(ssq_of_column/ssq_sum)#computing movie probabilties
    users_that_are_top = np.random.choice(len(users_probabilities),k, replace=False, p=users_probabilities) #sampling rows
    movies_that_are_top = np.random.choice(len(movies_probabilities),k, replace=False, p=movies_probabilities) #sampling columns
    movies_that_are_top.sort()
    users_that_are_top.sort()
    C = []
    R = []
    for i in users_that_are_top:
        R.append(list(matrix_of_user_movie[i]/sqrt(k*users_probabilities[i])))
    for j in movies_that_are_top:
        C.append(list(matrix_of_user_movie[:,j]/sqrt(k*movies_probabilities[j])))
    Ct = np.transpose(C)
    W = []
    for i in users_that_are_top:
        X=[]
        for j in movies_that_are_top:
            X.append(matrix_of_user_movie[i][j])#intersection of sampled rows and columns
        W.append(np.array(X))
    W = np.array(W)
    x,yt,sigma = building_matrix_svd_for(W)#SVD of intersection
    sigm_pinv = np.linalg.pinv(sigma) #Moore Penrose Pseudo Inverse
    sig_sq = np.linalg.matrix_power(sigm_pinv, 2)#square of pseudo-inverse
    y = np.transpose(yt)
    xt = np.transpose(x)
    U = np.matmul(y, sig_sq)
    U = np.matmul(U, xt)    #reconstructing U
    np.save('cur_ct.npy', Ct)
    np.save('cur_r.npy', R)
    new_x, new_yt, new_sigma = energy_90_top(x,yt,sigma)
    pinv_new_sigma = np.linalg.pinv(new_sigma)
    new_sig_sq = np.linalg.matrix_power(pinv_new_sigma, 2)
    y = np.transpose(new_yt)
    xt = np.transpose(new_x)
    U = np.matmul(y, new_sig_sq)
    U = np.matmul(U, xt)
    np.save('cur_u.npy', U)
    
def C_1_U_90_1_R(k):
    """
    performs the CUR decomposition on the user matrix with 90% retained energy and stores them as numpy arrays

    Parameters
    ----------
    k :
        number of rows and columns and taken

    Returns
    -------
    None.
    """
    matrix_of_user_movie = np.load('train.npy')
    #[[1,1,1,0,0],[3,3,3,0,0],[4,4,4,0,0],[5,5,5,0,0],[0,0,0,4,4],[0,0,0,5,5],[0,0,0,2,2]]
    ssq_sum = 0 #sum of squares of all elements
    users_of_numbers = matrix_of_user_movie.shape[0]
    movies_of_numbers = matrix_of_user_movie[0].size
    for i in range(users_of_numbers):
        for j in range(movies_of_numbers):
            ssq_sum = ssq_sum + matrix_of_user_movie[i][j]*matrix_of_user_movie[i][j]
    users_probabilities = []
    movies_probabilities = []
    for i in range(users_of_numbers):
        ssq_of_row = 0 #
        for j in range(movies_of_numbers):
            ssq_of_row = ssq_of_row + matrix_of_user_movie[i][j]*matrix_of_user_movie[i][j]
        users_probabilities.append(ssq_of_row/ssq_sum)#computing user probabilities
    for j in range(movies_of_numbers):
        ssq_of_column = 0
        for i in range(users_of_numbers):
            ssq_of_column = ssq_of_column + matrix_of_user_movie[i][j]*matrix_of_user_movie[i][j]
        movies_probabilities.append(ssq_of_column/ssq_sum)#computing movie probabilties
    users_that_are_top = np.random.choice(len(users_probabilities),k, replace=False, p=users_probabilities) #sampling rows
    movies_that_are_top = np.random.choice(len(movies_probabilities),k, replace=False, p=movies_probabilities) #sampling columns
    movies_that_are_top.sort()
    users_that_are_top.sort()
    C = []
    R = []
    for i in users_that_are_top:
        R.append(list(matrix_of_user_movie[i]/sqrt(k*users_probabilities[i])))
    for j in movies_that_are_top:
        C.append(list(matrix_of_user_movie[:,j]/sqrt(k*movies_probabilities[j])))
    Ct = np.transpose(C)
    W = []
    for i in users_that_are_top:
        X=[]
        for j in movies_that_are_top:
            X.append(matrix_of_user_movie[i][j])#intersection of sampled rows and columns
        W.append(np.array(X))
    W = np.array(W)
    x,yt,sigma = building_matrix_svd_for(W)#SVD of intersection
    sigm_pinv = np.linalg.pinv(sigma) #Moore Penrose Pseudo Inverse
    sig_sq = np.linalg.matrix_power(sigm_pinv, 2)#square of pseudo-inverse
    y = np.transpose(yt)
    xt = np.transpose(x)
    U = np.matmul(y, sig_sq)
    U = np.matmul(U, xt)    #reconstructing U
    np.save('cur_ct_90.npy', Ct)
    np.save('cur_r_90.npy', R)
    new_x, new_yt, new_sigma = energy_90_top(x,yt,sigma)#SVD with top 90% energy
    pinv_new_sigma = np.linalg.pinv(new_sigma)
    new_sig_sq = np.linalg.matrix_power(pinv_new_sigma, 2)
    y = np.transpose(new_yt)
    xt = np.transpose(new_x)
    U = np.matmul(y, new_sig_sq)
    U = np.matmul(U, xt)
    np.save('cur_u_90.npy', U) 
    
def srcr(matrix,final):
  """
    calculates spearman rank correlation coefficient
    
    Parameters
    ----------
    matrix :
        train values matrix
    final :
        C*U*R

    Returns
    -------
    values :
        Spearman Rank Correlation.

  """
  freq=0
  sum=0
  for i in range(0,len(matrix)):
    for j in range(0,len(matrix[i])):
      sum=sum+(matrix[i][j]-final[i][j])**2
      freq=freq+1
  sum=6*sum
  flag=(freq**3)-freq
  values=1-(sum/flag)
  return values
  

def cur_ponk_precision(mat, final):
  """
    calculating precision on top k for CUR

    Parameters
    ----------
    mat : 
        train values matrix.
    final :
        C*U*R

    Returns
    -------
    Precision/100:
        precision on top K
        

  """
  k_mat=final.tolist()
  freq=0.00
  dart=0.00
  for i in range(0,len(mat)):
    for j in range(0,len(mat[i])):
      freq=freq+1
      a=int(round(mat[i][j]))
      b=int(round(k_mat[i][j]))
      if (a==b):
        dart=dart+1
  precision=(dart*100)/freq
  return precision/100

def main():
    start=timeit.default_timer()
    C_1_U_1_R(600)
    print("Time taken")
    stop=timeit.default_timer()
    print("%s seconds" %(stop-start))
    C_1_U_90_1_R(600)
    print("Time taken for 90%")
    stop=timeit.default_timer()
    print("%s seconds" %(stop-start))
    Ct = np.load('cur_ct.npy')
    A = np.load('train.npy')
    #[[1,1,1,0,0],[3,3,3,0,0],[4,4,4,0,0],[5,5,5,0,0],[0,0,0,4,4],[0,0,0,5,5],[0,0,0,2,2]]
    R = np.load('cur_r.npy')
    U = np.load('cur_u.npy')
    final = np.matmul(Ct, U)
    final = np.matmul(final, R)
    rmse_err=sqrt(mean_squared_error(A, final))
    print("RMSE error is :")
    print(rmse_err)
    print("Precision on top k is :")
    ans=cur_ponk_precision(A, final)
    print(ans)
    answer = srcr(A, final)
    print("Spearman Rank Correlation is ", answer)
    Ct_90 = np.load('cur_ct_90.npy')
    R_90 = np.load('cur_r_90.npy')
    U_90 = np.load('cur_u_90.npy')
    final_90 = np.matmul(Ct_90, U_90)
    final_90 = np.matmul(final_90, R_90)
    rmse_err_90=sqrt(mean_squared_error(A, final_90))
    print("RMSE error for 90% is :")
    print(rmse_err_90)
    print("Precision on top k for 90% is :")
    ans_90=cur_ponk_precision(A, final_90)
    print(ans_90)
    answer_90 = srcr(A, final_90)
    print("Spearman Rank Correlation for 90%  is ", answer_90)
    del A
        
if __name__ == '__main__':
    main()