# -*- coding: utf-8 -*-
"""
@author: Aditya Agarwal
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def read_TT_Split():
    '''
    Read ML-1M dataset.
    Train test split 80:20
    Save data as csv
    '''
    ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None,
                      engine = 'python', encoding = 'latin-1')
    training_data, test_data = train_test_split(ratings, test_size = 0.2, random_state = 0)
    training_data.to_csv('train.csv', header = None, index = False)
    test_data.to_csv('test.csv', header = None, index = False)

def NP_ARR(df, start, end, n_mov):
    '''
    Conversion from csv file to numpy arrays for faster access and retrieval
    '''
    L_d = []
    for ids in range(start, end+1):
        movID = df[:, 1][df[:, 0] == ids]
        ratID = df[:, 2][df[:, 0] == ids]
        rats = np.zeros(n_mov)
        rats[movID - 1] = ratID
        L_d.append(list(rats))
    return L_d

      
def main_next():
    '''
    Read the train and test data stored as csv
    Call NP_ARR to get list of lists
    Save the data as numpy arrays/.npy files
    '''
    print("Read CSVs from ML-1M")
    all_train = pd.read_csv('train.csv')
    all_test = pd.read_csv('test.csv')
    all_train = np.array(all_train, dtype='int')
    all_test = np.array(all_test, dtype='int')
    n_user = int(max(max(all_train[:,0]), max(all_test[:,0])))
    n_mov = int(max(max(all_train[:,1]), max(all_test[:,1])))
    start_ID = 1
    end_ID = n_user
    all_train = NP_ARR(all_train, start_ID, end_ID, n_mov)
    all_test = NP_ARR(all_test, start_ID, end_ID, n_mov)
    all_train = np.array([np.array(x) for x in all_train])
    all_test = np.array([np.array(x) for x in all_test])
    print("Save train and test as .npy files")
    #print(all_test[0])
    np.save('train.npy',all_train)
    np.save('test.npy', all_test)
    
if __name__ == '__main__':
    # read_TT_Split()
    main_next()
    
    
    
    
    