from dataset import PyTorchDataset
from tcn import Model
import numpy as np
import torch 
import random

def split_dataset(X,Y,metadata, train_frac = 0.7  , val_frac = 0.15 ): 
    if train_frac + val_frac >= 1:
        raise ValueError (f"the sum of train_frac and val_frac is {train_frac + val_frac} but it should be less than 1")
    N = len(X)
    indices =  [n for n in range(N)]
    random.shuffle(indices)
    num_train  = int(train_frac * N)
    num_val = int(N * val_frac)
    
    train_indices = indices[0: num_train]
    val_indices = indices[num_train: num_train + num_val]
    test_indices = indices[num_train + num_val:]



    X_train = np.array([X[i] for i in train_indices])
    Y_train = np.array([Y[i] for i in  train_indices])
    metadata_train = [metadata[i] for i in train_indices]
    X_val = np.array([X[i] for i in val_indices])
    Y_val = np.array([Y[i] for i in val_indices])
    metadata_val = [metadata[i] for i in val_indices]
    X_test = np.array([X[i] for i in test_indices])
    Y_test = np.array([Y[i] for i in test_indices])
    metadata_test = [metadata[i] for i in test_indices]

    if len(X_train) == len(Y_train) == len(metadata_train) and len(X_val) == len(Y_val) == len(metadata_val) and len(X_test) == len(Y_test) == len(metadata_test):
            
        print("validated dataset lengths")
    else:
        raise ValueError ("inconsistent lengths for one or all datasets with train, val and test")
    if len(X_train) + len(X_val) + len(X_test) != N: 
            raise ValueError(f"The sum of the different sample classes don't add up to total number of samples: number of training samples = {len(X_train)}, number of validation samples = {len(X_val)}, and the number of test samples = {len(X_test)} ")
    
    return (X_train, Y_train, metadata_train, 
            X_val, Y_val, metadata_val, 
            X_test, Y_test, metadata_test
    )


def normalize():

def train_one_epoch():


def validate():


def main():


if __name__ = "__main__":
    main()