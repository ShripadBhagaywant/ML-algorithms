#Null value remove.
import  numpy as np
import  pandas as pd
dfd = {'First Score':[100,90,np.nan,95],
       'Second Score':[100,90,85,np.nan],
       'Third Score':[np.nan,90,80,95]}
df=pd.DataFrame(dfd)
#print(df)
x = df.isnull()
#print(x)
y=df.notnull()
#print(y)
z=df.fillna(0)
#print(z)
b=df.replace(to_replace=np.nan,value=-99)
#print(b)
c=df.dropna()
#print(c)
#dx=df.dropna(axis=1)
#print(dx)