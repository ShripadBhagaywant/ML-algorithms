#Dicision Tree.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("C:\\Users\\Shreepad\\Downloads\\weather\\seattle-weather.csv")

# Create a copy of the dataframe to apply label encoding
df_encoded = df.copy()

# Apply Label Encoding to categorical columns
label_encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df_encoded[col] = label_encoder.fit_transform(df[col])

# Prepare features and target
# Correct the column names and remove extra space
X = df_encoded[['temp_min', 'wind', 'weather']]
Y = df_encoded['weather']

# Create and train the Decision Tree Classifier
clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(X, Y)

# Visualize the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=X.columns, class_names=label_encoder.classes_, filled=True)
plt.show()