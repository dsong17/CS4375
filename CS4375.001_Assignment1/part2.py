import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor

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

        s = StandardScaler()
        X = pd.DataFrame(s.fit(X).fit_transform(X))
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

        X_train_scaled = s.fit_transform(X_train)
        X_test_scaled = s.fit_transform(X_test)

        return X_train_scaled, Y_train, X_test_scaled, Y_test

    def train_model(self, X_train, Y_train, alpha=0.001, eta0=0.001, max_iter=10000000, tol=0.0001):
        model = SGDRegressor(alpha=alpha, eta0=eta0, max_iter=max_iter, tol=tol)
        model.fit(X_train, Y_train)
        return model

    def evaluate_model(self, model, X_train, Y_train, X_test, Y_test):
        y_train_predict = model.predict(X_train)
        bias_weight = model.intercept_
        weights = model.coef_
        rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_predict))
        r2_train = r2_score(Y_train, y_train_predict)

        y_test_predict = model.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_predict))
        r2_test = r2_score(Y_test, y_test_predict)

        print("The model performance for training set")
        print("--------------------------------------")
        print("The model's coefficients are: ")
        print("w0:", bias_weight[0])
        for index, weight in enumerate(weights):
            print(f"w{index + 1}: {weight}")
        print('RMSE for training set:', rmse_train)
        print('R2 score for training set:', r2_train)
        print("\n")

        print("The model performance for testing set")
        print("--------------------------------------")
        print('RMSE for testing set:', rmse_test)
        print('R2 score for testing set:', r2_test)

if __name__ == "__main__":
    data_url = "https://raw.githubusercontent.com/ManyaBondada/CS4375.001_Assignment1/main/Concrete_Data.csv"
    data_handler = Data(data_url)
    data_handler.show_correlations() 
    
    # preprocess data
    data_handler.remove_rows_with_outliers()

    # split data
    X_train, Y_train, X_test, Y_test = data_handler.split_data()

    # train model
    model = data_handler.train_model(X_train, Y_train)

    # evaluate model
    data_handler.evaluate_model(model, X_train, Y_train, X_test, Y_test)

