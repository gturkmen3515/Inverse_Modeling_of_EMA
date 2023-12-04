# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:20:49 2020

@author: gokmenatakanturkmen@gmail.com
"""
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import joblib

class EmaModelReader:
    def __init__(self, model_prefix='ematwoexp', model_extension='.sav', data_file='doubleema1/double_exp_model.txt'):
        self.model_prefix = model_prefix
        self.model_extension = model_extension
        self.data = self.load_data(data_file)
        self.le = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.mean_val = np.float64([4.095931357497794, 6.143897036596352, 8.1918627154697, 10.239828394341824, 12.287794073213034, 14.335759752256555, 16.383725430933115, 18.43169110982901, 20.479656788671473])
        self.mean_val = np.reshape(self.mean_val, (len(self.mean_val), 1))

    def load_data(self, file_path):
        with open(file_path) as f:
            triplets = f.read().split()
        for i in range(len(triplets)):
            triplets[i] = triplets[i].split(',')
        A = np.asarray(triplets)
        data = A.reshape(int(len(A) / 9), 9)
        data = np.array(data)
        data = np.float64(data)
        d1 = data[:, 0:1]
        d2 = data[:, 2:3]
        d3 = np.concatenate((d1, d2), axis=1)
        d4 = data[:, 5:6]
        d5 = np.concatenate((d3, d4), axis=1)
        d6 = data[:, 7:]
        return np.concatenate((d5, d6), axis=1)

    def preprocess_data(self, x_val):
        val = float(x_val[:, 2:3]) * np.ones(((len(self.mean_val), 1)))
        min_dist = abs(self.mean_val - val)
        index = int(np.where(min_dist == np.min(min_dist))[0])
        return index

    def load_model(self, index):
        current = f'{self.model_prefix}{index}{self.model_extension}'
        return joblib.load(open(current, 'rb'))

    def make_prediction(self, model, x_val):
        y_pred = model.predict(x_val)
        current_rand = y_pred[:, 0:1] * 0.5 + 1
        d_rand = y_pred[:, 1:2] * 5 + 50
        return current_rand, d_rand

    def predict(self, x_val):
        index = self.preprocess_data(x_val)
        model = self.load_model(index)
        return self.make_prediction(model, x_val)

if __name__ == "__main__":
    ema_model_reader = EmaModelReader()
    x_val = np.array([[1.0, 2.0, 3.0]])  # Example input values
    current_rand, d_rand = ema_model_reader.predict(x_val)
    print(current_rand, d_rand)
