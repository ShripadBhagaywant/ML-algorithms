# Multiple Regression.
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
df=pd.read_csv("C:\\Users\\Shreepad\\Downloads\\House\\housing_price_dataset.csv")
print(df)
x=df[['Price']]
y=df[['SquareFeet']]
req = LinearRegression().fit(x,y)
y_pred = req.predict(x)
print(y_pred)
score = r2_score(y,y_pred)
print(score)
plt.plot(x,y)
plt.show()