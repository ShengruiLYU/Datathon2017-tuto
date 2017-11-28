from sklearn import linear_model
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
from sklearn.model_selection import validation_curve
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb


## to read the data sheet from excel
sheet=pd.read_excel('StockPriceExercise.xlsx',sheet_name='Sheet1')

## iris, a toy data set created by sk-learn, which is a graph of iris
#iris=load_iris()

# print(type(iris))

# print(sheet['Date'].values.tolist())
# print(sheet['Date'].values)
# plt.scatter(sheet['Date'].values.tolist(),sheet['NINTENDO'].values.tolist())
# plt.show()
# print(sheet.loc[1:,['Date','KONAMI','EA']])

# print(sheet.loc[1:,['NINTENDO']])
#model=linear_model.LogisticRegression()
model=xgb.XGBRegressor()

# data=sheet.loc[:,['Date','KONAMI','EA']].values

data=sheet.loc[:,['KONAMI','EA']].values

labels=sheet.loc[:,['NINTENDO']].values.astype(int)


#model.fit(X=data,y=labels.ravel())

## validation curve
## ts: training score, vs: validation score
## you can choose the parameter you want to tune by specifying its name, and set the range for it. 
##logspace(-3,0): (0.001,1)
ts,vs = validation_curve(model,data,labels,param_name="learning_rate",param_range=np.logspace(-3,0),cv=5)

## cross_validation
## return the score for each fold. For example, cv=10, the first entry is the score for using 1st data cluster as testing set, the remaining 9 as training set.  
#predicted =cross_val_predict(model,data,labels,cv=5)


##calculate the mean and sd for 5(number of cross validation) curves. 
tsmean = np.mean(ts,axis=1)
tsstd = np.std(ts,axis=1)
vsmean = np.mean(vs,axis = 1)
vsstd = np.std(vs,axis = 1)




fig, ax = plt.subplots()

## scatter: scatter graph (discrete points)
# ax.scatter(sheet.loc[:,['Date']],predicted,label='predicted')

## x-axis first, then y axis. Specify the label, remenber to add plt.legend() to show the label in the graph
ax.plot(np.logspace(-3,0),tsmean,label = 'training score')
ax.plot(np.logspace(-3,0),vsmean,label = 'validation score')
#ax.plot(sheet.loc[:,['Date']],labels,label='actual values')


## locate the maximun y value and the corresponding x value
max_y  = max(vsmean.tolist())
max_x = np.logspace(-3,0)[vsmean.tolist().index(max_y)]
print (max_x, max_y)


ax.set_xlabel('learning_rate')
ax.set_ylabel('score')

## to show the label
plt.legend()

## to show the graph
plt.show()

#for i in range(0,15):
	# print(i,model.predict([sheet.loc[i,['KONAMI','EA']]]))


# for i in range(7001,7015):
# 	print(i,model.predict([sheet.loc[i,['KONAMI','EA']]]))

# xgb.plot_importance(model).figure.savefig("importance.png")

