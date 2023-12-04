# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 12:20:49 2022

@author: gokmenatakanturkmen@gmail.com
"""
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import numpy as np
import joblib

class EmaRegressor:
    def __init__(self, file_path, gain=1):
        self.data = self.load_data(file_path, gain)
        self.le = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.mean_val = []
        self.error = []
        self.oob = []
        self.error_test = []
        self.n_est = []

    def load_data(self, file_path, gain):
        with open(file_path) as f:
            triplets = f.read().split()
        for i in range(len(triplets)):
            triplets[i] = triplets[i].split(',')
        A = np.asarray(triplets)
        data = A.reshape(int(len(A) / 9), 9)
        data = np.array(data, dtype=np.float64)
        d1 = data[:, 0:1]
        d2 = data[:, 2:3]
        d3 = np.concatenate((d1, d2), axis=1)
        d4 = data[:, 3:6] * gain
        d5 = np.concatenate((d3, d4), axis=1)
        d6 = data[:, 6:]
        return np.concatenate((d5, d6), axis=1)

    def preprocess_data(self, current_data_value):
        current_index = np.where(self.data[:, 6] == current_data_value)[0]
        current_data = self.data[current_index[0]:current_index[-1], :]
        x = current_data[:, 0:6]
        y = current_data[:, 6:8]
        mean_value = np.mean(x[:, 2:3])
        return x, y, mean_value

    def train_model(self, x_train, y_train, n_estimators):
        rf_class = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion='gini',
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features='auto',
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=1000,
            verbose=1,
            warm_start=True,
            class_weight='balanced_subsample',
            min_samples_split=2,
            max_depth=None
        )
        rf_class.fit(x_train, y_train)
        return rf_class

    def evaluate_model(self, model, x_test, y_test):
        y_pred = model.predict(x_test)
        mae_model = mean_absolute_error(y_test, y_pred)
        mse_model = mean_squared_error(y_test, y_pred)
        oob_error = 1 - model.oob_score_
        return mae_model, mse_model, oob_error

    def train_and_evaluate_models(self, n_estimators_values):
        for i in range(9):
            x_train, y_train, _ = self.preprocess_data(1.0 * (i + 1))
            x_test, y_test, _ = self.preprocess_data(1.0 * (i + 1))

            model = self.train_model(x_train, y_train, 100)  # Using a fixed n_estimator=100
            mae, mse, oob_error = self.evaluate_model(model, x_test, y_test)

            self.error.append(mae)
            self.oob.append(oob_error)
            self.error_test.append(mse)
            self.n_est.append(40)  # Fixed n_estimator value

            filename = f'ematwo_{i}.sav'
            joblib.dump(model, filename)

    def plot_results(self):
        plt.figure()
        plt.plot(self.error)
        plt.ylabel('MAE')

        plt.figure()
        plt.plot(self.error_test)
        plt.xlabel("Number Of Estimator")
        plt.ylabel("Mean Square Error")

        plt.figure()
        plt.plot(self.oob)
        plt.ylabel('OOB')

if __name__ == "__main__":
    ema_regressor = EmaRegressor('ema_data/dEMA/dEMA.txt', gain=1)
    ema_regressor.train_and_evaluate_models(range(10, 110, 10))
    ema_regressor.plot_results()
    plt.show()





