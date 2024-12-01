#SVM
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# Input data
x = np.array([[1,2],[5,8],[8,8],[1,0.6]])
y = [1,0,1,0]

# Train SVM classifier
clf = svm.SVC(kernel='linear').fit(x, y)

# Predict for a new point
print("Prediction:", clf.predict([[0.56, 0.76]]))

# Get coefficients
w = clf.coef_[0]
print("Coefficients:", w)

# Calculate decision boundary line
a = -w[0] / w[1]
xx = np.linspace(0, 12)
yy = a * xx - clf.intercept_[0] / w[1]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(xx, yy, 'k--', label="Decision Boundary")
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('SVM Classification')
plt.legend()
plt.show()