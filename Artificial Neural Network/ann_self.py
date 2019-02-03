# -*- coding: utf-8 -*-
################# Part 1  #################################

#1. Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#2. Import dataset and see basic information about it
df=pd.read_csv('churn_modelling.csv')
print (df.head(10))
print(df.info())
print(df.describe())
X= df.iloc[:,3:13].values
y= df.iloc[:,13].values
print(X)

#3. Exploratory data Analysis

# Checking correraltion among variables in Dataframes
corr=df.corr()

fig, ax = plt.subplots(figsize=(10, 10))
#Generate Color Map, red & blue
colormap = sns.diverging_palette(220, 10, as_cmap=True)
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
#Apply xticks
plt.xticks(range(len(corr.columns)), corr.columns);
#Apply yticks
plt.yticks(range(len(corr.columns)), corr.columns)
#show plot
plt.show()

# checking distribution of Gender with Exited column
g=sns.lmplot(y='Gender', x='Tenure', col="Exited", data=df)
plt.show()

g=sns.countplot(x='HasCrCard',data=df,hue='Exited')
plt.show()

#4. Handling and Encoding Categorical data
# Geography Column
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Gender Column
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Create dummy variables for geography columna and if there is no relation for value in Geography Column
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
#Removing 1 Column for Dummy Variable
X=X[:,1:]
print(X)

#5. Spliting the train set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.3,random_state=103)


#6. Feature scaling to ease calculation and avoiding biasness between the variable
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


####################################################################

############# Part 2 ################################



###################################################################

import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout


# seq   ##initialise model
# dense ## create layers

#6. Initialise the ANN

classifier = Sequential()

#7 Adding input layer and first hidden layer

classifier.add(Dense(activation="relu", # rectifier function
                     units=6, 
                     input_dim=11, #Input Fetures
                     kernel_initializer="uniform"))
classifier.add(Dropout(rate=0.1))   ##-- p = 10% , remove 1 in 10 neuron in first layer


#8 Adding second hidden layer

classifier.add(Dense(activation="relu", 
                     units=6,  
                     kernel_initializer="uniform"))

classifier.add(Dropout(rate=0.1))   ##-- p = 10% , remove 1 in 10 neuron in first layer

#9 Adding Output layer 

# soft max if more than 2 category in output -- units =3 and activation to "softmax"
classifier.add(Dense(activation="sigmoid",  # sigmoid function 
                     units=1,  
                     kernel_initializer="uniform"))

#10 Compiling the ANN

# for more than 2 output category then loss=Categorical_crossentropy

classifier.compile(optimizer="adam",  # adam schocastic gradient method
                   loss="binary_crossentropy",
                   metrics=['accuracy']
                   )

#11 Fitting data into ANN

classifier.fit(X_train,
               y_train,
               batch_size=10,
               epochs=100)


#12 Prediction of labels

 y_pred=classifier.predict(X_test)  # gives probability
 
 y_pred=(y_pred > 0.5)  # cnverting to boolean

X_test[0]
#13 Making confusion matrix
 
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
 

####################################################################

############# Part 3 ################################



###################################################################

####### Home work #######

"""
Use our ANN model to predict if the customer with the following informations will leave the bank: 

Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000
So should we say goodbye to that customer ?

""""
#[]-- give result in column
#[[]] --- give result in rows

# 1 Feature should be in np.array 
# 2 Check for any encoding 
# 3 Check for any scalaing
# $ Then boom, get a prediction


new_prediction=classifier.predict(sc.transform(
        np.array([
                [
                0,0,600,1,40,
                3,60000,2,1,1,50000
                ]
                ])
              )
        )


new_prediction = (new_prediction >0.5)



####################################################################

############# Part 4 ################################

###################################################################

""" Evaluating Improving Tuning """

# Evaluating the ANN

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

def build_classifier():
    classifier=Sequential()
    classifier.add(Dense(activation="relu", # rectifier function
                     units=6, 
                     input_dim=11, #Input Fetures
                     kernel_initializer="uniform"))

    classifier.add(Dense(activation="relu", 
                     units=6,  
                     kernel_initializer="uniform"))

    classifier.add(Dense(activation="sigmoid",  # sigmoid function 
                     units=1,  
                     kernel_initializer="uniform"))

    classifier.compile(optimizer="adam",  # adam schocastic gradient method
                   loss="binary_crossentropy",
                   metrics=['accuracy']
                   )
    return classifier
classifier= KerasClassifier(build_fn=build_classifier, 
                            batch_size=10,
                            epochs=100)
accuracies = cross_val_score(estimator=classifier,
                             X=X_train,
                             y=y_train,
                             cv=10)
mn_acc=accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout regularisation to reduce over fitting of data

# Tuning the ANN

from sklearn.model_selection import GridSearchCV

def build_classifier(optimiser,loss):
    classifier=Sequential()
    classifier.add(Dense(activation="relu", # rectifier function
                     units=6, 
                     input_dim=11, #Input Fetures
                     kernel_initializer="uniform"))

    classifier.add(Dense(activation="relu", 
                     units=6,  
                     kernel_initializer="uniform"))

    classifier.add(Dense(activation="sigmoid",  # sigmoid function 
                     units=1,  
                     kernel_initializer="uniform"))

    classifier.compile(optimizer=optimiser,  # adam schocastic gradient method
                   loss=loss,
                   metrics=['accuracy']
                   )
    return classifier
classifier= KerasClassifier(build_fn=build_classifier)

parameters ={'batch_size': [20,25,28],
                'epochs':[80,380],
                'optimiser':['SGD','adam','RMSprop'],
                'loss':['mean_squared_error','poisson','binary_crossentropy']
        }
grid_serach = GridSearchCV(estimator=classifier,param_grid=parameters,scoring='accuracy',cv = 10 )
grid_serach.fit(X_train,y_train)
best_param= grid_serach.best_params_
best_accuracy = grid_serach.best_score_




 
