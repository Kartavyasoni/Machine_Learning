# Import Libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
diabetes = datasets.load_diabetes()

# Create X & Y Data Metrics
x = diabetes.data
y = diabetes.target
x.shape, y.shape

# Split Data into Train & Test.
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)
x_train.shape , y_train.shape
x_test.shape , y_test.shape

# Define Model
model = LinearRegression()

# Fit Data Into Model
model.fit(x_train, y_train)

#Apply Test Data Into Trained Model For Prediction
y_pred = model.predict(x_test)

# Model Performance
print("Coefficient:\n",model.coef_)
print("\nIntercept: %.2f" % model.intercept_)
print("\nMean Squared Error (MSE): %.2f" % mean_squared_error(y_test, y_pred))
print("\nCoefficient of Determination (r^2): %.2f" % r2_score(y_test, y_pred))

# Best Fit Line
coef = np.polyfit(y_pred, y_test, 1)
y_best_fit = np.polyval(coef, y_pred)
print("\nBest fit line: y = ",np.polynomial.Polynomial(coef))

# Scatteplot of Orignal Value vs Prediction Value With Best Fit Line
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=y_test, alpha=0.5, label='Result')
sns.lineplot(x=y_pred, y=y_best_fit, color='red', label='Best Fit Line')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Predicted vs Actual with Best Fit Line')
plt.legend()
plt.show()
