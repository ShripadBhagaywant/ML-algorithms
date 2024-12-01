
import  pandas as pd
from matplotlib import pyplot as plt
iris = pd.read_csv("C:\\Users\\Shreepad\\Downloads\\iris\\iris.csv")
print(iris.head(20))
plt.plot(iris["sepal_length"],"r--")
plt.show()
iris.plot(kind="Scatter",X="sepal_length",Y="petal_length")
plt.show()