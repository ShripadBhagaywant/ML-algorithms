#Native Bayes.
import numpy as np
from sklearn.naive_bayes import GaussianNB

# Correct input data shapes
x_train = np.array([[1,2],[3,4],[5,6],[7,8]])
y_train = np.array([1,1,2,2])  # Flattened 1D array for training labels
x_test = np.array([[9,10],[11,12],[13,14],[15,16]])

# Create and train classifier
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predict and calculate score
y_pred = classifier.predict(x_test)
print("Predictions:", y_pred)
print("Accuracy : " ,classifier.score(x_test,y_pred))
