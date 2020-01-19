# Simple Linear Regression

# Importing the libraries
import numpy as np #for calculations
import matplotlib.pyplot as plt
import pandas as pd #csv file read ke liye

# Importing the dataset
dataset = pd.read_csv('monthlyexp vs incom.csv') #path can be added as well to read csv
X = dataset.iloc[:, :1].values #separating values into matrix input can be converted into array by  
y = dataset.iloc[:, 1].values 
from sklearn.model_selection import train_test_split #to split training set intp 4 parts
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=1/10, random_state=0)
from sklearn.linear_model import LinearRegression #importing Lnear reg command
regressor=LinearRegression() #regressor is a variable
regressor.fit(X_train, y_train) #fitting (giving input) to regressor and training data (x and y train) onto regressor
plt.scatter(X_train,y_train, color = 'green') #shows traning data
plt.scatter(X_train, #plt.scatter shows dot dot graph
            regressor.predict(X_train), #predicts y using this, forms scattered graph of y predictions
            color = 'blue')
plt.plot(X_train, #plt.plot shows line graph
            regressor.predict(X_train), #predicts y using this lne plot of y predictions
            color = 'red')
plt.title('Months Experience Vs Income (Training set)')
plt.xlabel('Months Experience')
plt.ylabel('Income')
plt.show()

plt.scatter(X_test,y_test, color = 'green')
plt.scatter(X_train, #plt.scatter shows dot dot graph
            regressor.predict(X_train), #predicts y using this
            color = 'blue')
plt.plot(X_train, #plt.plot shows line graph
            regressor.predict(X_train), #predicts y using this
            color = 'red')
plt.title('Months Experience Vs Income (Testing set)')
plt.xlabel('Months Experience')
plt.ylabel('Income')
plt.show()

print(regressor.predict([[18]]))


