# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 19:09:45 2021

@author: Ranak Roy Chowdhury
"""
import math
import numpy as np
import os
import torch
import shutil
import sys
import warnings
import argparse
import tarnet_utils
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='AF')
parser.add_argument('--batch', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--emb_size', type=int, default=64)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--task_rate', type=float, default=0.5)
parser.add_argument('--masking_ratio', type=float, default=0.15)
parser.add_argument('--lamb', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--ratio_highest_attention', type=float, default=0.5)
parser.add_argument('--avg', type=str, default='macro')
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--nhid_task', type=int, default=128)
parser.add_argument('--nhid_tar', type=int, default=128)
parser.add_argument('--task_type', type=str,
                    default='classification', help='[classification, regression]')
args = parser.parse_args()


def main():
    prop = tarnet_utils.get_prop(args)
    path = './data/' + prop['dataset'] + '/'
    train_file = path + prop['dataset'] + '_TRAIN.ts'
    test_file = path + prop['dataset'] + '_TEST.ts'
    norm = 'standard'

    print('Data loading start...')
    X_train, y_train = tarnet_utils.load_from_tsfile_to_dataframe(train_file)
    X_test, y_test = tarnet_utils.load_from_tsfile_to_dataframe(test_file)
    print('Data loading complete...')

    print('Data preprocessing start...')
    min_len = np.inf
    for i in range(len(X_train)):
        x = X_train.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)
    for i in range(len(X_test)):
        x = X_test.iloc[i, :]
        all_len = [len(y) for y in x]
        min_len = min(min(all_len), min_len)
    x_train = tarnet_utils.process_data(
        X_train, normalise=norm, min_len=min_len)
    x_test = tarnet_utils.process_data(X_test, normalise=norm, min_len=min_len)

    x_train, y_train, x_test, y_test = tarnet_utils.preprocess(
        prop, x_train, y_train, x_test, y_test)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    print('Data preprocessing complete...')

    prop['nclasses'] = torch.max(y_train).item(
    ) + 1 if prop['task_type'] == 'classification' else None
    prop['dataset'], prop['seq_len'], prop['input_size'] = prop['dataset'], x_train.shape[1], x_train.shape[2]
    prop['device'] = torch.device(
        'cuda:0' if torch.cuda.is_available() else "cpu")

    print('Initializing model...')
    model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer = tarnet_utils.initialize_training(
        prop)
    print('Model intialized...')

    print('Training start...')
    tarnet_utils.training(model, optimizer, criterion_tar, criterion_task, best_model,
                          best_optimizer, x_train, y_train, x_test, y_test, prop)
    print('Training complete...')


if __name__ == "__main__":
    main()
