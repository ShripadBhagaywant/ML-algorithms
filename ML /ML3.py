#Conversion of Categorial data to numerical.
import  pandas as pd
play = pd.read_csv("C:\\Users\\Shreepad\\Downloads\\PlayTennis\\Play-Tennis.csv")
print(play)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
lable =le.fit_transform(play['Wind'])
print(lable)
play.drop("Wind",axis=1,inplace=True)
play['Wind']=lable
print(play)