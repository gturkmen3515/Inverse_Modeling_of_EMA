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
    def __init__(self, data_path='ema_data/sEMA/sEMA.txt'):
        self.data = self.load_data(data_path)
        self.le = LabelEncoder()
        self.scaler = MinMaxScaler()
        self.error = []
        self.oob = []
        self.error_test = []
        self.n_est = []

    def load_data(self, file_path):
        with open(file_path) as f:
            triplets = f.read().split()
        for i in range(len(triplets)):
            triplets[i] = triplets[i].split(',')
        A = np.asarray(triplets)
        data = A.reshape(int(len(A) / 9), 9)
        return np.float64(data)

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
        for i, n_estimators in enumerate(n_estimators_values, start=1):
            x_train, y_train, _ = self.preprocess_data(1.0 * i)
            x_test, y_test, _ = self.preprocess_data(1.0 * i)

            model = self.train_model(x_train, y_train, n_estimators)
            mae, mse, oob_error = self.evaluate_model(model, x_test, y_test)

            self.error.append(mae)
            self.oob.append(oob_error)
            self.error_test.append(mse)
            self.n_est.append(n_estimators)

            filename = f'emaone_{i}.sav'
            joblib.dump(model, filename)

    def plot_results(self):
        plt.figure()
        plt.plot(self.n_est, self.error, label='Mean Absolute Error')
        plt.xlabel("Number Of Estimator")
        plt.ylabel("Mean Absolute Error")
        plt.legend()

        plt.figure()
        plt.plot(self.n_est, self.error_test, label='Mean Squared Error')
        plt.xlabel("Number Of Estimator")
        plt.ylabel("Mean Squared Error")
        plt.legend()

        plt.figure()
        plt.plot(self.n_est, self.oob, label='OOB Error')
        plt.xlabel("Number Of Estimator")
        plt.ylabel("OOB Error")
        plt.legend()

if __name__ == "__main__":
    ema_regressor = EmaRegressor()
    ema_regressor.train_and_evaluate_models(range(10, 110, 10))
    ema_regressor.plot_results()
    plt.show()





