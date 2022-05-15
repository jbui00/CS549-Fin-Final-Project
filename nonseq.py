import matplotlib.pyplot as plt 
import csv
import numpy as np
import multiprocessing as mp
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier  
from imblearn.under_sampling import RandomUnderSampler

%matplotlib inline

def setup():
    
    def setup():
    #load data
     train_data = open('targetfirm_prediction_dataset_small.csv', 'r', newline = '')
    reader  = csv.reader(train_data, delimiter = ',' , quoting = csv.QUOTE_NONE) 
    #close loaded data
    readerlist = list(reader)
    train_data.close()
    data = np.array(readerlist)
    data[data == ''] = 0
    #set up data
    X_data = data[1:data.shape[0],4:18] 
    print("X_data Shape: {}".format(X_data.shape))
    Y_data = data[1:data.shape[0],3] 
    print("Y_data Shape: {}".format(Y_data.shape))
    #train data
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size = 0.2, shuffle = True)
    
    print("X_train shape: {}".format(X_train.shape))
    print("X_test shape: {}".format(X_test.shape))
    print("Y_train shape: {}".format(Y_train.shape))
    print("Y_test shape: {}".format(Y_test.shape))

    
    #set test data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    
    rus = RandomUnderSampler(sampling_strategy = 0.6, random_state = 42)
    X_train_resample, Y_train_resample = rus.fit_resample(X_train, Y_train)
    print("X_train resampled shape: {}".format(X_train_resample.shape))
    print("Y_train resampled shape: {}".format(Y_train_resample.shape))

    return X_train_resample, Y_train_resample, X_test, Y_test

def logisticregression(X_train,Y_train,X_test,Y_test):
    l_reg = LogisticRegression() 
    l_reg.fit(X_train,Y_train)
    y_pred = l_reg.predict(X_test)

    confusion_matrix = metrics.confusion_matrix(Y_test, y_pred) 
    accuracy = l_reg.score(X_test, Y_test) 
    print("confusion matrix : \n{}".format(confusion_matrix)) 
    print("accuracy : {0:.2%}".format(accuracy))
    print("F1 Score: ", f1_score(Y_test, y_pred, average=None))

    
def logisticregressiontest(X_train,Y_train,X_test,Y_test):
    l_reg = LogisticRegression() 
    l_reg.fit(X_train,Y_train)
    y_pred = l_reg.predict(X_test)

    confusion_matrix = metrics.confusion_matrix(Y_test, y_pred) 
    accuracy = l_reg.score(X_test, Y_test) 
    print("confusion matrix : \n{}".format(confusion_matrix)) 
    print("accuracy : {0:.2%}".format(accuracy))
    print("F1 Score: ", f1_score(Y_test, y_pred, average=None))
    #return confusion_matrix for testing purposes
    return confusion_matrix
  
if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = setup()
    logm = mp.Process(target = logisticregression(X_train, Y_train, X_test, Y_test))
    #plt.plot(X_train, Y_train, color = "blue")
    #plt.plot(X_test, Y_test, color = "green")
    #plt.xlim(-2,10)
    #plt.ylim(0,1)
    #start = time.time()
    #logm.start()
    #logm.join()
    #end = time.time()

confusion_matrix = logisticregressiontest(X_train, Y_train, X_test, Y_test)
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
