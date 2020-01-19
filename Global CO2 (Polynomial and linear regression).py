# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 11:12:02 2020

@author: Sana
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv('global_co2.csv')
X = dataset.iloc[:, :1].values
y = dataset.iloc[:, 1].values

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =9 )
X_poly = poly_reg.fit_transform(X) #fit OBTAIN ANSWER, TRANSFORM TRANSFORM INPUT AS O/P (LOOK UP)
poly_reg.fit(X_poly, y) 
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'pink')
plt.title('CO2 Emissions vs Years (Linear Regression)')
plt.xlabel('Years')
plt.ylabel('CO2 Emissions')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'blue')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'green')
plt.title('CO2 Emissions vs Years (Polynomial Regression)')
plt.xlabel('Years')
plt.ylabel('CO2 Emissions')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'violet')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'black')
plt.title('CO2 Emissions vs Years (Smoother Polynomial Regression)')
plt.xlabel('Years')
plt.ylabel('CO2 Emissions')
plt.show()

# Predicting a new result with Linear Regression
print ("Value for 2013 with Linear Regression=",lin_reg.predict([[2013]]))

# Predicting a new result with Polynomial Regression
print ("Value for 2013 with Polynomial Regression=",lin_reg_2.predict(poly_reg.fit_transform([[2013]])))