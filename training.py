
#importing required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
import pickle
import colorama
colorama.init()



##############################################**Linear regression**####################################################

def LinearRegressionFunc():
	from sklearn.linear_model import LinearRegression
	Lreg = LinearRegression()
	Lreg.fit(X_train,y_train)

	global LinearRegressionScore
	LinearRegressionScore = Lreg.score(X_test,y_test)
	print(colorama.Fore.LIGHTMAGENTA_EX,"\nAccuracy obtained by Linear Regression model:", colorama.Fore.LIGHTGREEN_EX,LinearRegressionScore*100,"%")

	# Saving model to current directory
	# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
	pickle.dump(Lreg, open('LinearReg.pkl','wb'))



##############################################**K Neighbors Classifier**####################################################

def KNeighborsClassifierFunc():
	from sklearn.neighbors import KNeighborsClassifier
	KNN =  KNeighborsClassifier(n_neighbors=7, metric='manhattan', weights='distance')
	KNN.fit(X_train,y_train)

	global KNeighborsClassifierScore
	KNeighborsClassifierScore = KNN.score(X_test,y_test)
	print(colorama.Fore.LIGHTMAGENTA_EX,"\nAccuracy obtained by K Neighbors Classifier model:",colorama.Fore.LIGHTGREEN_EX,KNeighborsClassifierScore*100,"%")

	pickle.dump(KNN, open('KNNclassifier.pkl','wb'))



##############################################**Decision Tree Classifier**####################################################

def DecisionTreeClassifierFunc():
	from sklearn.tree import DecisionTreeClassifier
	tree = DecisionTreeClassifier()
	tree.fit(X_train,y_train)

	global DecisionTreeClassifierScore
	DecisionTreeClassifierScore = tree.score(X_test,y_test)
	print(colorama.Fore.LIGHTMAGENTA_EX,"\nAccuracy obtained by Decision Tree Classifier model:",colorama.Fore.LIGHTGREEN_EX,DecisionTreeClassifierScore*100,"%")

	pickle.dump(tree, open('DTclassifier.pkl','wb'))



##############################################**Random Forest Classifier**####################################################

def RandomForestClassifierFunc():
	from sklearn.ensemble import RandomForestClassifier
	forest = RandomForestClassifier()
	forest.fit(X_train,y_train)

	global RandomForestClassifierScore
	RandomForestClassifierScore = forest.score(X_test,y_test)
	print(colorama.Fore.LIGHTMAGENTA_EX,"\nAccuracy obtained by Random Forest Classifier model:",colorama.Fore.LIGHTGREEN_EX,RandomForestClassifierScore*100,"%")

	pickle.dump(forest, open('RFclassifier.pkl','wb'))


		
##############################################** main function **############################################################


if __name__ == '__main__':

	#####################################################setting-up data################################################# 

	print(colorama.Fore.LIGHTYELLOW_EX,'READING & SETTING-UP OUR DATA SET')
	data_set = pd.read_csv('cloth_size_test.csv')
	print(colorama.Fore.LIGHTBLUE_EX,"\nShape of csv file : ",colorama.Fore.LIGHTCYAN_EX,data_set.shape)
	print(colorama.Fore.LIGHTBLUE_EX,"\nFirst 5 rows of the data : \n",colorama.Fore.LIGHTCYAN_EX,data_set.head())
	data_set = data_set.dropna()   #Removing all such rows from our data_set

	
	fig, ax = plt.subplots(figsize=(8,6))   
	sns.countplot(x=data_set["size"]);
	plt.show()

	fig, ax = plt.subplots(figsize=(8,6))
	sns.distplot(data_set["height"], color="r");
	plt.show()

	fig, ax = plt.subplots(figsize=(8,6))
	sns.distplot(data_set["weight"], color="b");
	plt.show()

	fig, ax = plt.subplots(figsize=(8,6))
	sns.distplot(data_set["age"], color="g");
	plt.show()

	################################Plot to check Impact of features on size#############################################

	fig, axes = plt.subplots(1,3,figsize=(20,5))
	fig.suptitle('Predictor', fontdict = { 'fontsize': 30})

	size_order = ['XXS','S','M','L','XL','XXL','XXXL']

	# weight
	sns.boxplot(x = 'size',y = 'weight', data = data_set, ax = axes[0], order=size_order)
	axes[0].set_title('weight')

	# age
	sns.boxplot(x = 'size',y = 'age', data = data_set, ax = axes[1], order=size_order)
	axes[1].set_title('age')

	# height
	sns.boxplot(x = 'size',y = 'height', data = data_set, ax = axes[2], order=size_order)
	axes[2].set_title('height')
	plt.show()

	print(colorama.Fore.LIGHTCYAN_EX,"\nWeight has the most impact on the size of clothes, gradually with increase in weight, size of the clothes increases")


	###################################################**Training**########################################################
	data_set['size'] = data_set['size'].map({'XXS': 1, 'S': 2, "M" : 3, "L" : 4, "XL" : 5, "XXL" : 6, "XXXL" : 7})
	X = data_set.drop("size", axis=1)
	y = data_set["size"]
	
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

	from sklearn.metrics import accuracy_score,classification_report

	
	print(colorama.Fore.LIGHTYELLOW_EX,'\nTRAINING')
	LinearRegressionFunc()
	KNeighborsClassifierFunc()
	DecisionTreeClassifierFunc()
	RandomForestClassifierFunc()

	x = ["Linear Regression", "K Neighbors Classifier", "Decision Tree Classifier", "Random Forest Classifier" ]
	y = [LinearRegressionScore*100, KNeighborsClassifierScore*100,  DecisionTreeClassifierScore*100, RandomForestClassifierScore*100]
	fig, ax = plt.subplots(figsize=(8,6))
	sns.barplot(x=x,y=y, palette="flare");
	plt.ylabel("Model Accuracy %")
	plt.xticks(rotation=10)
	plt.title("Model Comparison based om Accuracy");
	plt.show()


	##############################################**Removing the outliers**################################################
	print(colorama.Fore.LIGHTCYAN_EX,"\nRemoving the outliers (by Z-score method) in order to increase model accuracy")

	data_set_WO = []
	sizes = []
	for size_type in data_set['size'].unique():
		sizes.append(size_type)
		ndf = data_set[['age','height','weight']][data_set['size'] == size_type]
		zscore = ((ndf - ndf.mean())/ndf.std())
		data_set_WO.append(zscore)

	for i in range(len(data_set_WO)):
		data_set_WO[i]['age'] = data_set_WO[i]['age'][(data_set_WO[i]['age']>-3) & (data_set_WO[i]['age']<3)]
		data_set_WO[i]['height'] = data_set_WO[i]['height'][(data_set_WO[i]['height']>-3) & (data_set_WO[i]['height']<3)]
		data_set_WO[i]['weight'] = data_set_WO[i]['weight'][(data_set_WO[i]['weight']>-3) & (data_set_WO[i]['weight']<3)]

	for i in range(len(sizes)):
		data_set_WO[i]['size'] = sizes[i]

	new_data_set = pd.concat(data_set_WO)
	new_data_set['age'][new_data_set['age']<-3]
	new_data_set['height'][new_data_set['height']<-3]
	new_data_set['weight'][new_data_set['weight']<-3]
	new_data_set = new_data_set.dropna()

	fig, axes = plt.subplots(1,3,figsize=(20,5))
	fig.suptitle('Predictor')

	# weight
	sns.boxplot(x = 'size',y = 'weight', data = new_data_set, ax = axes[0])
	axes[0].set_title('weight')

	# age
	sns.boxplot(x = 'size',y = 'age', data = new_data_set, ax = axes[1])
	axes[1].set_title('age')

	# height
	sns.boxplot(x = 'size',y = 'height', data = new_data_set, ax = axes[2])
	axes[2].set_title('height')
	plt.show()

	###################################################**Training**########################################################
	X = new_data_set.drop("size", axis=1)
	y = new_data_set["size"]

	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
	print(colorama.Fore.LIGHTYELLOW_EX,'\nTRAINING')
	KNeighborsClassifierFunc()
	DecisionTreeClassifierFunc()
	RandomForestClassifierFunc()

	x = ["Linear Regression", "K Neighbors Classifier", "Decision Tree Classifier", "Random Forest Classifier" ]
	y = [LinearRegressionScore*100, KNeighborsClassifierScore*100,  DecisionTreeClassifierScore*100, RandomForestClassifierScore*100]
	fig, ax = plt.subplots(figsize=(8,6))
	sns.barplot(x=x,y=y, palette="flare");
	plt.ylabel("Model Accuracy %")
	plt.xticks(rotation=10)
	plt.title("Model Comparison based om Accuracy");
	plt.show()	

	print(colorama.Fore.LIGHTYELLOW_EX,'\nPREDICTION',colorama.Fore.LIGHTBLUE_EX)
	choice = int(input('Choose the model you want to use for predicting size of the cloth : \n 1 : Linear Regression \n 2 : K Neighbors Classifier \n 3 : Decision Tree Classifier \n 4 : Random Forest Classifier \n '))
	if(choice == 1):
		model = pickle.load(open('LinearReg.pkl', 'rb'))
		pickle.dump(model, open('model.pkl','wb'))
	elif(choice == 2):
		model = pickle.load(open('KNNclassifier.pkl', 'rb'))
		pickle.dump(model, open('model.pkl','wb'))
	elif(choice == 3):
		model = pickle.load(open('DTclassifier.pkl', 'rb'))
		pickle.dump(model, open('model.pkl','wb'))
	elif(choice == 4):
		model = pickle.load(open('RFclassifier.pkl', 'rb'))
		pickle.dump(model, open('model.pkl','wb'))
	else:
		print(colorama.Fore.LIGHTMAGENTA_EX,"invalid choice, Linear Regression model selected by default ")
		model = pickle.load(open('LinearReg.pkl', 'rb'))
		pickle.dump(model, open('model.pkl','wb'))
































