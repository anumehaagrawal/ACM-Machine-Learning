from sklearn import datasets, preprocessing
import numpy as py
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split  

iris=datasets.load_iris()
X=iris.data[:,:]
y=iris.target
maxi=list()


X_train,X_test , y_train ,y_test = train_test_split(X ,y ,test_size=0.25, random_state=20)
a=[(100,),
	(2,3),(10,5),(30,100),(10,6),(9,40),(100,30)]
X_trainscale=preprocessing.scale(X_train)
X_testscale=preprocessing.scale(X_test)
for i in a:
	model=MLPClassifier(hidden_layer_sizes= i,solver='lbfgs' , activation ='tanh', random_state=6)
	model.fit(X_trainscale,y_train)

	print model.predict(X_testscale)
	maxi.append(model.score(X_testscale,y_test))
	print("Accuracy for hidden layer %s is %f" % (i,model.score(X_testscale,y_test)))
print max(maxi)




