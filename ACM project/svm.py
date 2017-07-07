from sklearn import datasets,svm , preprocessing
import numpy as np
from sklearn.model_selection import train_test_split  

# loading data set iris
iris=datasets.load_iris()
X=iris.data[:,:]
y=iris.target
X_train,X_test , y_train ,y_test = train_test_split(X ,y ,test_size=0.25, random_state=20)
maxi=list()
C=[0.01 , 0.05 , 0.1 , 0.5 , 1 , 2 , 5 , 10 , 100]
X_trainscale=preprocessing.scale(X_train)
X_testscale=preprocessing.scale(X_test)


for i in C :
	# classifier and kernel rbf
	model=svm.SVC(kernel='rbf',C=i,gamma='auto')
		# fitting training data
	model.fit(X_trainscale,y_train)
		
		# predicting accuracy of dat
	accuracy = model.score(X_testscale,y_test)
	maxi.append(accuracy)
	print("Output for test data is")
	
	print("for the given c %f  we have accuracy as %f" % (i,accuracy))
	# print predictedted output
	print model.predict(X_testscale)
print("maximum accuracy in svm method is %f" % (max(maxi)))




