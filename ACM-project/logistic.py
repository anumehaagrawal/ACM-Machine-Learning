from sklearn import datasets,preprocessing
import numpy as py
import matplotlib.pyplot as plot
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split  

iris=datasets.load_iris()
X=iris.data[:,:]
y=iris.target
maxi=list()


X_train,X_test , y_train ,y_test = train_test_split(X ,y ,test_size=0.25, random_state=20)
a=[0.01,0.05,0.1,0.5,1,5,10,100]
X_trainscale=preprocessing.scale(X_train)
X_testscale=preprocessing.scale(X_test)
for i in a:
	model=LogisticRegression(C= i,solver='lbfgs' ,multi_class='multinomial', random_state=6)
	model.fit(X_trainscale,y_train)


	print model.predict(X_testscale)
	maxi.append(model.score(X_testscale,y_test))
	print("Accuracy for c = %f is %f" % (i,model.score(X_testscale,y_test)))
print max(maxi)