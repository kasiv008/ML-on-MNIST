# -*- coding: utf-8 -*-
"""
Created on Sun march  9 19:08:01 2021
#Python 3.8.5
@author: KASI VISWANATH
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import seaborn as sns
from tensorflow import keras
from math import log,pi 
from scipy.linalg import eigh
from sklearn.decomposition import PCA 
from sklearn import svm


#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#loading the dataset
train_data=pd.read_csv('train.csv',header=None)
train_d=np.array(train_data.iloc[:,1:])     # Training data
train_l=np.array(train_data.iloc[:,0])      # Training data labels

# print(np.transpose(train_data.iloc[:,1:]).describe())

# Splitting data into training and testing subsets below
train_d, test_d, train_l, test_l =train_test_split(train_d, train_l, test_size=0.2, random_state=0) 
sns.countplot(train_l)

#scaling the data to (0,1)
scaler = StandardScaler()
train_d = scaler.fit_transform(train_d)
test_d = scaler.transform(test_d)

test_data=pd.read_csv('test.csv',header=None) 
# visualizing performance on actual test data
test_d1 = np.array(test_data.iloc[:,:])
test_d1 = scaler.fit_transform(test_d1)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#ARTIFICIAL NEURAL NETWORK
def ANN():
#initiating the model
    model=keras.Sequential([
        keras.layers.Dense(512,activation='relu',input_dim=train_d.shape[1]),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128,activation='relu'),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(64,activation='relu',name="intermediate_layer"),
        keras.layers.Dense(10,activation='softmax')])
    
    #compiling the model
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    
    #Model Summary
    model.summary()
    
    #checkpoint
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath='ANN_model',
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    
    #Training the model
    model.fit(train_d,train_l,epochs=100,batch_size=10,validation_data=(test_d,test_l),callbacks=[model_checkpoint_callback])
    

    #Loading the trained model and testing.
    test_model=keras.models.load_model('ANN_model')
    #evaluating the validation data
    test_model.evaluate(test_d,test_l)
    #Generating the confusion matrix
    predicted_digits=test_model.predict_classes(test_d)
    cm=confusion_matrix(test_l,predicted_digits)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.set_xlabel('Predicted outputs', color='black')
    ax.set_ylabel('Actual outputs', color='black')
    ax.xaxis.set(ticks=range(10))
    ax.yaxis.set(ticks=range(10))
    ax.set_ylim(9.5, -0.5)
    for i in range(10):
        for j in range(10):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
    plt.show()
    print('Classification report of ANN for test data')
    print(classification_report(test_l, predicted_digits))
    predictions = test_model.predict_classes(test_d1)
    for i in range(10):
        img = np.reshape(np.array(test_d1)[i],(28,28))
        plt.imshow(img)
        print(predictions[i])
        plt.pause(0.1)

    #Extracting the features from an intermediate layer
    feature_extractor = keras.Model(
        inputs=test_model.inputs,
        outputs=test_model.get_layer(name="intermediate_layer").output,
    )
    print('Extracted Features' )
    for i in range(10):
      sample=test_d[i]
      sample=np.reshape(sample,(1,784))
      features=feature_extractor(sample)
      features=np.reshape(features,(8,8))
      plt.imshow(features)
      print(predictions[i])
      plt.pause(0.1)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#MAXIMUM LIKELIHOOD ESTIMATION (MLE)

def MLE():

    covmat_list = []
    mean_list = []
    invcov_list = []
    eigval_list = []
    logdet_list = []
    predicted_digits=[]
    N = 500000 #train_d.shape[0]/10 
    k = train_d.shape[1] #Number of pixels 28*28
    
    for i in range(10):
        class_data = train_d[train_l==i]#filtering data points of a particular label
        class_mean = np.mean(class_data,0)#mean
        mean_list.append(class_mean)
        class_cov = np.matmul((class_data-class_mean).T, (class_data-class_mean))/N#computibg covarience matrix
        covmat_list.append(class_cov)
        invcov_list.append(np.linalg.pinv(class_cov))
        eigvals,_ = np.linalg.eig(class_cov)#computing eigen values
        eigval_list.append(eigvals)
        log_det = 0
        for k in range(eigvals.shape[0]):
                if eigvals[k].real > 0 :
                    log_det += log(eigvals[k].real)
        logdet_list.append(log_det)            
        
        
    covmat_list = np.array(covmat_list)   
    mean_list = np.array(mean_list)
    invcov_list = np.array(invcov_list)
    logdet_list = np.array(logdet_list)
        
    for j in range(len(test_l)):
        predictions = []
        for i in range(10):
            prob = -0.5*(((test_d[j]-mean_list[i]).T)@(invcov_list[i])@(test_d[j]-mean_list[i])) - 0.5*(k*log(2*pi)+logdet_list[i])#computing the probability
            predictions.append(prob)
        predicted_class = np.argmax(predictions)
        predicted_digits.append(predicted_class)
    
    cm = confusion_matrix(test_l, predicted_digits)#confusion matrix
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm,cmap='viridis')
    ax.grid(False)
    ax.set_xlabel('Predicted outputs', color='black')
    ax.set_ylabel('Actual outputs', color='black')
    ax.xaxis.set(ticks=range(10))
    ax.yaxis.set(ticks=range(10))
    ax.set_ylim(9.5, -0.5)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
    plt.show()
    print('Classification report of MLE for test day=ta')
    print(classification_report(test_l, predicted_digits))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#K NEAREST NEIGHBOR (KNN)
# Printing accuracies with K = 1, 3, 7
#here i am directly predicting the test labels

def KNN(k,train_d,test_d,train_l,test_l):
    nsamples = test_l.shape[0]
    ndata = train_l.shape[0]
    predicted_digits = np.zeros(nsamples,dtype=np.float64)
    for i in range(nsamples):
        distances = np.zeros(ndata)
        for j in range(ndata):
            distances[j] = np.sqrt(np.sum(np.square(test_d[i,:]-train_d[j,:])))#calculating the distance between the examples 
        indices = np.argsort(distances)[:k]# sorting the distance
        votes = np.array(train_l)[indices].astype(int)
        predicted_digits[i] = np.argmax(np.bincount(votes))

    accuracy = np.sum((predicted_digits==test_l)*1)/nsamples*100
    print("Putting K =",k,"Accuracy is",accuracy)
    cm = confusion_matrix(test_l, predicted_digits)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm,cmap='viridis')
    ax.grid(False)
    ax.set_xlabel('Predicted outputs', color='black')
    ax.set_ylabel('Actual outputs', color='black')
    ax.xaxis.set(ticks=range(10))
    ax.yaxis.set(ticks=range(10))
    ax.set_ylim(9.5, -0.5)
    for i in range(10):
      for j in range(10):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
    plt.show()
    print('Classification report of KNN for est data')
    print(classification_report(test_l, predicted_digits))
    
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#PRINCIPAL COMPONENT ANALYSIS(2-D)
def PCAV():
    #Computing Covarience Matrix
    cov_mat=np.matmul(train_d.T ,train_d)
    print(cov_mat.shape)
    
    # the parameter 'eigvals' is defined (low value to heigh value) 
    # eigh function will return the eigen values in asending order
    # this code generates only the top 2 (782 and 783)(index) eigenvalues.
    values, vectors = eigh(cov_mat, eigvals=(782,783))
    vectors = vectors.T
    
    new_data = np.matmul(vectors, train_d.T)
    
    # appending label to the 2d projected data(vertical stack)
    new_coordinates = np.vstack((new_data, train_l)).T
    
    # creating a new data frame for ploting the labeled points.
    dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))
    print(dataframe.head())
    # ploting the 2d data points with seaborn
    print('Plotting the two principals')
    sns.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
    plt.show()   

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#PRINCIPAL COMPONENT ANALYSIS + SUPPORT VECTOR MACHINE (PCA+SVM)   
def PCASVM():
    pca = PCA(n_components=50,whiten=True) #initiating PCA with 50 components 
    train_dat = pca.fit_transform(train_d)
    test_dat=pca.transform(test_d)# transforming test data into pca
    svc = svm.SVC(kernel='poly',C=10,gamma=1)#loading SVM model
    svc.fit(train_dat, train_l)
    predicted_digits=svc.predict(test_dat)
    print('Classification report of PCA+SVM for test data')
    print(classification_report(test_l,predicted_digits)) 

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------                                       

#Run different methods
print("RUNNING CODE FOR ANN")
ANN()
print("RUNNING CODE FOR MLE")
MLE()
print("RUNNING CODE FOR PCA")
PCAV()
print("RUNNING CODE FOR PCA+SVM")
PCASVM()
print('RUNNING CODE FOR KNN(this may take a while(~56 mins) for training,running for k=1,\
      please change the number of datapoints for faster implementation )')
                                                       #*Datapoints*=*accuracy*(*time*s)
KNN(1,train_d[:30000,:],test_d,train_l[:30000],test_l) #100=63.2(10s),1000=83.3(112s),5000=89.6(566)
#KNN(3,train_d[:30000,:],test_d,train_l[:30000],test_l)#100=56.3(13s),1000=82.4(111s),5000=90.1
#KNN(7,train_d[:30000,:],test_d,train_l[:30000],test_l)#10000=91.75%