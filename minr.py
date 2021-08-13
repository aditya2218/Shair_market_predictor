import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn import linear_model
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

path="msft.csv"
name=["Date "  ,"Open  " ,"High "   ,"Low "," Close  "  ,"Volume"  ,"Adj." ,"Close*"]
data=pd.read_csv(path,header=None,skiprows=1,names=name)
print(data.head())
x=np.array(data.iloc[:,2:3])
y=np.array(data.iloc[:,3:4])
'''
print(x.ndim)
print(y)
'''
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2)

model=linear_model.LinearRegression()
model.fit(x_train,y_train)

#print(x_test.ndim)
y_predicted=(model.predict(x_test))
print(len(y_predicted))
print(len(y_test))




a=float(input("Enter Open    :-"))
b=model.predict(np.array([[a]]))
print(b)




plt.axis([24,30,24,30])
plt.xlabel("Open  ")
plt.ylabel(" Close  ")
plt.scatter(x,y)
plt.plot(x_test,y_predicted,color="g")
plt.scatter(a,b,color="r",marker="^")
plt.show()

wwe=[int(i) for i in y_test]
kk=[int(i) for i in y_predicted]
print("Accuracy   :- {}".format(accuracy_score(wwe,kk)))
print(classification_report(wwe,kk))














