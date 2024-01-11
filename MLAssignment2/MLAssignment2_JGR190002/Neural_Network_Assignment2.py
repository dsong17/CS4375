#!/usr/bin/env python
# coding: utf-8

# In[203]:


#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from itertools import product


class NeuralNet:
    def __init__(self, dataFile, header=True):
        self.raw_input = pd.read_csv(dataFile)

    def preprocess(self):
        self.processed_data = self.raw_input
        threshold = 1.5
        X = self.processed_data.iloc[:, :-1]

        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1

        lower_bounds = Q1 - threshold * IQR
        upper_bounds = Q3 + threshold * IQR

        outlier_mask = ((X < lower_bounds) | (X > upper_bounds)).any(axis=1)

        data_no_outliers = self.processed_data[~outlier_mask]

        return data_no_outliers
    
    def modelPerformanceMetrics(self, epoch, X_val, y_val, model,mae_history,r2_history,mse_history):
        # In this section we  calculate MSE,MAE, and R^2 scores on a section of partially trained data.
        
        # We make a temporary prediction based on the supplied training data we have so far.
        temp_y_pred = model.predict(X_val)
        temp_mae = mean_absolute_error(y_val, temp_y_pred)
        temp_r2 = r2_score(y_val, temp_y_pred)
        temp_mse = mean_squared_error(y_val, temp_y_pred)

        # Then we record these values in a array/list where we would use it to plot it against the epoch/iterations
        mae_history.append(temp_mae)
        r2_history.append(temp_r2)
        mse_history.append(temp_mse)  
        
        # For testing purposes, we print out the 3 metrics calculated at that point in time.
        # The purpose is to observe the behavior of the model as the epoch increases.
        if epoch % 100 == 0 and epoch > 0:
            print(f'Epoch {epoch}: MAE = {temp_mae:.4f}, R2 = {temp_r2:.4f}, MSE = {temp_mse:.4f}')
        
        return mae_history,r2_history,mse_history

        
    def train_evaluate(self, df):
        # We get all of the columns except 'Concrete compressive strength(MPa, megapascals) ' column, which is our target of prediction.
        X = df.drop(['Concrete compressive strength(MPa, megapascals) '], axis=1)
        y = df['Concrete compressive strength(MPa, megapascals) ']

        # We split into testing and training data with a 80/20 split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # These are the neural network parameters we are going to be adjusting and picking various combinations of.
        activations = ['logistic', 'tanh', 'relu']
        learning_rates = [0.1, 0.01]
        max_iters = [500, 1000]  # Iterations are also known as epoch
        hidden_layer_size = [(10, 15), (20, 35)]
        batch_size = 32
        
        # List declared to be used to be passed into a dataframe to display data into tuplar format.
        results_list = []

        # Using the method product from the itertools library we can efficiently loop through every combination of
        # parameters for the model without having 4 nested for loops within each other.
        
        # The way it works is that we specify the variables/lists we which to iterate through and we make only one for-loop
        # below it that will go through each one of these.
        param_combinations = product(
            hidden_layer_size, activations, learning_rates, max_iters
        )
        print("Visualizing the model history progression:\n")
        for params in param_combinations:
            hidden_layer_size, activation, learning_rate_init, max_iter = params
            
            # We declare 3 lists to hold the 3 performance metrics we wish to plot for. With r^2 being the testing score of the model.
            mae_history = []
            r2_history = []
            mse_history = []
            # This is the model we create to make the Neural Network using the MLPRregressor method
            model = MLPRegressor(
                # The four parameters we wish to modify
                hidden_layer_sizes=hidden_layer_size,
                activation=activation,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
            )
            # To keep track of the score/r^2 score and various performance metrics we have a nested for-loop that iterates according
            # to the current # iterations we are using in the model and we take batch size of data to partially train on, in this case 32 values.
            print("\nHidden layer size: ",hidden_layer_size,"Activation Func: ",activation,"Learning Rate: ",learning_rate_init,"Epoch/Iterations: ",max_iter)
            for epoch in range(max_iter):
                for i in range(0, len(X_train), batch_size):
                    model.partial_fit(X_train[i:i + batch_size], y_train[i:i + batch_size])
                # Then we call a method which records the various performace metrics with the partially train data. 
                # We return the history into the lists for printing the plots.
                mae_history,r2_history,mse_history=self.modelPerformanceMetrics(epoch, X_test, y_test,model,mae_history,r2_history,mse_history)
            #We call the plotting method to perform graph plottting of the current model history of accuracy and performance metrics.
            self.plottingMethod(mae_history,r2_history,mse_history)
            
            # We then calculate the various model performance metrics and record them in a dictionary.
            
            # Reason why we calc. tabular data here instead of the modelPerformanceMetrics method is because we are only
            # interested in the final performance of the model after it is fully trained.
            y_pred = model.predict(X_test)
            scoreTesting = r2_score(y_test, y_pred)
            scoreTraining = r2_score(y_train, model.predict(X_train))
            mseTesting = mean_squared_error(y_test, y_pred)
            mseTraining = mean_squared_error(y_train, model.predict(X_train))
            maeTesting = mean_absolute_error(y_test, y_pred)
            maeTraining = mean_absolute_error(y_train, model.predict(X_train))
            maperntage = mean_absolute_percentage_error(y_test, y_pred)
            
            # Kepping track of the parameters used and the various performance metric with the given parameters inside a dictironary.
            results = {
                'Hidden Layer Size': hidden_layer_size,
                'Activation Func.': activation,
                'Learning Rate #': learning_rate_init,
                'Epoch/Iterations': max_iter,
                'Testing Score': scoreTesting,
                'Training Score': scoreTraining,
                'MSE Testing': mseTesting,
                'MSE Training': mseTraining,
                'MAE Testing': maeTesting,
                'MAE Training': maeTraining,
                'Mean Abs. %': maperntage,
            }
            # From the dictory we append to the list to pass it though a dataframe for printing
            results_list.append(results)
        # Pass the list into datafram and print the final tabular data
        results_df = pd.DataFrame(results_list)
        print(results_df)
        return 0


    
    def plottingMethod(self,mae_history,r2_history,mse_history):
        # Plot the model history for MAE, R2, and MSE with all against Epoch, after fitting the model with the given trial
        # of parameters.
        
        # We adjust scatterplot dot size to size 20. To see where the dots are going
        dot_size = 10
        
        # Making a 16 by 9 ratio figure plot. Typically this is the ratio used not have heavily distorted graphs.
        plt.figure(figsize=(16, 9))
        
        # Making a 1 row by 3 column subplot, with 1 being the current plot we are at.
        # We plot the model history for mean absolute error for the specigic trial we have done.
        plt.subplot(1, 3, 1)
        plt.scatter(np.arange(1, len(mae_history) + 1), mae_history,s=dot_size)
        plt.title('Mean Absolute Error vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Absolute Error')

        # We plot the scatter plot for the test score or r^2.
        plt.subplot(1, 3, 2)
        plt.scatter(np.arange(1, len(r2_history) + 1), r2_history,s=dot_size)
        plt.title('R^2/Testing Accuracy vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('R^2')

        #Plotting the model history of mean squared error for the specigic trial we have done.
        plt.subplot(1, 3, 3)
        plt.scatter(np.arange(1, len(mse_history) + 1), mse_history,s=dot_size)
        plt.title('Mean Squared Error vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Mean Squared Error')

        # Showing the scatter polot.
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    neural_network = NeuralNet("https://utdallas.box.com/shared/static/b7woqah5sk2ooh7od6xw0sd163p0kajm.csv")
    df=neural_network.preprocess()
    neural_network.train_evaluate(df)


# In[ ]:




