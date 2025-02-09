import numpy as np
import pandas as pd
from sklearn.linear_model import  LogisticRegression
import os

def load_waterbirds_data(path, frac=1.0, label_noise=0.):
    
    
    features_path, metadata_path = path + 'resnet18_1layer.npy', path + 'metadata.csv'
    TRAIN, VAL, TEST = (0, 1, 2)
    # del VAL
    features = np.load(features_path)
    metadata = pd.read_csv(metadata_path)
    # Train
    train_mask = metadata['split']==TRAIN
    train_y = metadata[train_mask]['y'].values
    if label_noise > 0.:
        train_y = np.logical_xor(train_y, np.random.binomial(1, label_noise, size=len(train_y)))
    train_x = features[train_mask,:]
    train_g = 2*metadata[train_mask]['y'].values + metadata[train_mask]['place'].values
    if frac < 1:
        idx = np.random.choice(np.arange(train_y.size), int(frac*(train_y.size)))
        train_x = train_x[idx,:]
        train_y = train_y[idx]
        train_g = train_g[idx]
    # Test
    test_mask = metadata['split']==TEST
    test_y = metadata[test_mask]['y'].values
    if label_noise > 0.:
        test_y = np.logical_xor(test_y, np.random.binomial(1, label_noise, size=len(test_y)))
    test_x = features[test_mask,:]
    test_g = 2*metadata[test_mask]['y'].values + metadata[test_mask]['place'].values
    
    
    # val
    val_mask = metadata['split']==VAL
    val_y = metadata[val_mask]['y'].values
    if label_noise > 0.:
        val_y = np.logical_xor(val_y, np.random.binomial(1, label_noise, size=len(val_y)))
    val_x = features[val_mask,:]
    val_g = 2*metadata[val_mask]['y'].values + metadata[val_mask]['place'].values
    
    
    test_x = np.concatenate((test_x, val_x), axis = 0)
    test_y, test_g = np.hstack((test_y, val_y)), np.hstack((test_g, val_g))
    
    
    return ((train_x, train_y, train_g), (test_x, test_y, test_g)), 4


def load_waterbirds_data_full(path, frac=1.0, label_noise=0.):
    
    features_path, metadata_path = path + 'resnet18_1layer.npy', path + 'metadata.csv'
    TRAIN, VAL, TEST = (0, 1, 2)
    
    # del VAL
    features = np.load(features_path)
    metadata = pd.read_csv(metadata_path)
    
    # Train
    train_mask = metadata['split']==TRAIN
    train_y = metadata[train_mask]['y'].values
    if label_noise > 0.:
        train_y = np.logical_xor(train_y, np.random.binomial(1, label_noise, size=len(train_y)))
    train_x = features[train_mask,:]
    train_g = 2*metadata[train_mask]['y'].values + metadata[train_mask]['place'].values
    if frac < 1:
        idx = np.random.choice(np.arange(train_y.size), int(frac*(train_y.size)))
        train_x = train_x[idx,:]
        train_y = train_y[idx]
        train_g = train_g[idx]
        
    # Test
    test_mask = metadata['split']==TEST
    test_y = metadata[test_mask]['y'].values
    if label_noise > 0.:
        test_y = np.logical_xor(test_y, np.random.binomial(1, label_noise, size=len(test_y)))
    test_x = features[test_mask,:]
    test_g = 2*metadata[test_mask]['y'].values + metadata[test_mask]['place'].values
    
    # val
    val_mask = metadata['split']==VAL
    val_y = metadata[val_mask]['y'].values
    if label_noise > 0.:
        val_y = np.logical_xor(val_y, np.random.binomial(1, label_noise, size=len(val_y)))
    val_x = features[val_mask,:]
    val_g = 2*metadata[val_mask]['y'].values + metadata[val_mask]['place'].values
    
    test_x = np.concatenate((test_x, val_x), axis = 0)
    test_y, test_g = np.hstack((test_y, val_y)), np.hstack((test_g, val_g))
    
    return ((train_x, train_y, train_g), (test_x, test_y, test_g)), 4



