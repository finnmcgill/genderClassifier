from sklearn import tree, svm, linear_model

# [height, weight, shoe size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

testData = [[190, 70, 43]]

# Decision Tree (example from video)
def decTreePredFunc(X,Y,arr):
	decTree = tree.DecisionTreeClassifier()

	decTree = decTree.fit(X,Y) # fit trains decision tree on data set

	decTreePrediction = decTree.predict([arr])

	return decTreePrediction

# Support Vector Machine
def SVMPredFunc(X,Y,arr):
	SVM = svm.SVC(gamma='auto')

	SVM = SVM.fit(X,Y)

	SVMPred = SVM.predict([arr])

	return SVMPred

# Stochastic Gradient Descent
def SGDPredFunc(X,Y,arr):
	SGD = linear_model.SGDClassifier()

	SGD = SGD.fit(X,Y)

	SGDPred = SGD.predict([arr])

	return SGDPred

# Passive Aggressive Algorithm
def PAAPredFunc(X,Y,arr):
	PAA = linear_model.PassiveAggressiveClassifier()

	PAA = PAA.fit(X,Y)

	PAAPred = PAA.predict([arr])

	return PAAPred


# Trying different models on test data:
decTreeScore = 0
for i in range(0, 11):
	if decTreePredFunc(X,Y,X[i]) == Y[i]:
		decTreeScore += 1

SVMScore = 0
for i in range(0, 11):
	if SVMPredFunc(X,Y,X[i]) == Y[i]:
		SVMScore += 1

SGDScore = 0
for i in range(0, 11):
	if SGDPredFunc(X,Y,X[i]) == Y[i]:
		SGDScore += 1

PAAScore = 0
for i in range(0, 11):
	if PAAPredFunc(X,Y,X[i]) == Y[i]:
		PAAScore += 1

#print('Decision Tree ' + str(decTreeScore))
#print('Support Vector Machine ' + str(SVMScore))
#print('Stochastic Gradient Descent ' + str(SGDScore))
#print('Passive Aggressive Algorithm ' + str(PAAScore))

# Since out of the 3 latter algorithms SVM generally scores the highest it is showcased
print('SVM Predictions from data:')
for i in range(0, 11):
	print('Prediction: ' + str(SVMPredFunc(X, Y, X[i])[0]) + ', Actual: ' + str(Y[i]))