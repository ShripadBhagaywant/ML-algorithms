#Liner Regression.
import matplotlib.pyplot as plt
from scipy import stats
x = [10, 30, 80, 97, 89, 88]
y = [22, 76, 68, 97, 45, 44]
# Perform linear regression
slope, intercept, r, p, std_err = stats.linregress(x, y)
# Create prediction function
def myfunction(x):
    return slope * x + intercept
# Generate model predictions
model = [myfunction(xi) for xi in x]
# Create plot
plt.scatter(x, y, label='Data points')
plt.plot(x, model, color='red', label='Regression line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.show()