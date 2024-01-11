import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class Data:
    def __init__(self, url):
        self.data = pd.read_csv(url)

    def show_correlations(self):
        for col in self.data.columns[:-1]:
            sns.scatterplot(x=self.data[col], y=self.data["Concrete compressive strength(MPa, megapascals) "])
            plt.xlabel(col)
            plt.ylabel("Concrete compressive strength(MPa, megapascals) ")
            plt.title(f"{col} vs. Concrete compressive strength(MPa, megapascals) ")
            plt.show()

    def remove_rows_with_outliers(self, threshold=1.5):
        X = self.data.iloc[:, :-1]
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        lower_bounds = Q1 - threshold * IQR
        upper_bounds = Q3 + threshold * IQR
        outlier_mask = ((X < lower_bounds) | (X > upper_bounds)).any(axis=1)
        self.data = self.data[~outlier_mask]

    def split_data(self, test_size=0.2, random_state=5):
        X = self.data.drop(['Concrete compressive strength(MPa, megapascals) '], axis=1)
        Y = self.data['Concrete compressive strength(MPa, megapascals) ']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)

        X_train_scaled_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train_scaled))
        X_test_scaled_bias = np.hstack((np.ones((X_test.shape[0], 1)), X_test_scaled))

        return X_train_scaled_bias, Y_train, X_test_scaled_bias, Y_test

class Gradient:
    def __init__(self, learn_rate=0.1, n_iter=10000):
        self.learn_rate = learn_rate
        self.n_iter = n_iter

    def ssr_gradient(self, x, y, w):
        m = len(y)
        y_pred = x.dot(w)
        res = y_pred - y
        gradient = (2 / m) * x.T.dot(res)
        return gradient

    def gradient_descent(self, gradient, x, y, start, tolerance=1e-06):
        vector = start
        for _ in range(self.n_iter):
            diff = -self.learn_rate * np.array(gradient(x, y, vector))
            if np.all(np.abs(diff) <= tolerance):
                break
            vector += diff
        return vector

    def train_model(self, X_train, Y_train):
        weights = self.gradient_descent(self.ssr_gradient, X_train, Y_train, start=np.zeros(X_train.shape[1]))
        return weights

    def evaluate_model(self, X_train, Y_train, X_test, Y_test, weights):
        y_train_predict = X_train.dot(weights)
        y_test_predict = X_test.dot(weights)
        r2_train = r2_score(Y_train, y_train_predict)
        r2_test = r2_score(Y_test, y_test_predict)
        rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predict))
        rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predict))

        return {
            "R2_train": r2_train,
            "R2_test": r2_test,
            "RMSE_train": rmse_train,
            "RMSE_test": rmse_test
        }

if __name__ == "__main__":
    data_url = "https://raw.githubusercontent.com/ManyaBondada/CS4375.001_Assignment1/main/Concrete_Data.csv"
    data_handler = Data(data_url)
    data_handler.show_correlations() 
    
    # pre process data  
    data_handler.remove_rows_with_outliers()

    # prep data for training
    X_train, Y_train, X_test, Y_test = data_handler.split_data()
    
    # perform regression  
    gradient_handler = Gradient()
    weights = gradient_handler.train_model(X_train, Y_train)
    evaluation_result = gradient_handler.evaluate_model(X_train, Y_train, X_test, Y_test, weights)
    
    # output evaluation metrics
    print("Regression weights:")
    print("-------------------")
    for i in range (len(weights)):
        print(f"w{i}: {weights[i]}")

    print("The model performance for training set")
    print("--------------------------------------")
    print("RMSE is", evaluation_result["RMSE_train"])
    print("R2 score is", evaluation_result["R2_train"])
    print("\n")

    print("The model performance for testing set")
    print("--------------------------------------")
    print("RMSE is", evaluation_result["RMSE_test"])
    print("R2 score is", evaluation_result["R2_test"])