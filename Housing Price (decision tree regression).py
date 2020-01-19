# Decision Tree Regression used for classification problems

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics 

# Importing the dataset
dataset = pd.read_csv('housing price.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1:].values

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)



# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'purple')
plt.plot(X_grid, regressor.predict(X_grid), color = 'violet')
plt.title('ID vs Housing Price (Decision Tree Regression)')
plt.xlabel('ID')
plt.ylabel('Housing Price')
plt.show()
# Predicting a new result
print(regressor.predict([[1541]]))

