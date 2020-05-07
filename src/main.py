import numpy as np
import pickle

from sklearn import metrics,svm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.datasets import load_svmlight_file
 
from utilities import writevec
from tdparse import targettw

class TDParse():
    def __init__ (self,tr_file=None,model_file=None,scaler_file=None):
        if model_file == None and scaler_file == None and tr_file != None:
            x_train,y_train = self.read_tdparse(tr_file)
            self.train(x_train,y_train)
        elif model_file != None and scaler_file != None and tr_file == None:
            self.load(model_file,scaler_file)
            
    def train(self,x_train,y_train):
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x_train)
        
        c_vals = [1e-05, 3e-05, 5e-05, 7e-05, 9e-05, 0.0001, 0.00030000000000000003, 0.0005, 0.0006999999999999999, 0.0009000000000000001, 0.001, 0.003, 0.005, 0.006999999999999999, 0.009000000000000001, 0.01, 0.03, 0.05, 0.06999999999999999, 0.09000000000000001, 0.1, 0.3, 0.5, 0.7, 0.9000000000000001, 1.0, 3.0, 5.0, 7.0, 9.000000000000002]
        #c_vals=[1e-05,1]
        cv=ShuffleSplit(n_splits=5,test_size=0.2).split(x_scaled,y_train)
        self.clf = svm.LinearSVC(max_iter=10000,dual=False)
        print('Grid searching')
        self.grid = GridSearchCV(self.clf,param_grid=[{'C':c_vals}],cv=cv,scoring=['accuracy','f1_macro'],verbose=5,refit='f1_macro')
        self.grid.fit(x_scaled,y_train)
        self.clf = self.grid.best_estimator_

    def predict(self,x):
        x_scaled = self.scaler.transform(x)
        return self.clf.predict(x_scaled)

    def evaluate(self,x_test,y_test):
        y_pred = self.predict(x_test)
        acc=metrics.accuracy_score(y_test, y_pred)
        f1 = metrics.f1_score(y_test,y_pred,average='macro')
        return acc,f1
        
    def read_tdparse(self,file):
        data = load_svmlight_file(file)
        return data[0].toarray(), data[1]

    def load(self,model_file,scaler_file):
        with open(model_file,'rb') as f:
            self.clf = pickle.load(f)
        with open(scaler_file,'rb') as f:
            self.scaler = pickle.load(f)

    def save(self,model_file,scaler_file):
        with open(model_file,'wb') as f:
            pickle.dump(self.clf,f)
        with open(scaler_file,'wb') as f:
            pickle.dump(self.scaler,f)

    r'''def write_tr_files(self):
        features=targettw()
        x_train,y_train=features.lidongfeat(r'C:\Users\milok\tdparse\data\lidong\training', r'C:\Users\milok\tdparse\data\lidong\parses\lidong.train.conll') #data, conll
        writevec(r'C:\Users\milok\tdparse\data\lidong\output\training',x_train,y_train)
        x_test,y_test=features.lidongfeat(r'C:\Users\milok\tdparse\data\lidong\testing', r'C:\Users\milok\tdparse\data\lidong\parses\lidong.test.conll')
        writevec(r'C:\Users\milok\tdparse\data\lidong\output\testing',x_test,y_test)'''
def write_tr_files():
    features=targettw()
    x_train,y_train=features.sentihoodfeats(r'C:\Users\milok\tdparse\data\sentihood\training', r'C:\Users\milok\tdparse\data\sentihood\parses\sentihood.train.conll') #data, conll
    writevec(r'C:\Users\milok\tdparse\data\sentihood\output\training',x_train,y_train)
    x_test,y_test=features.sentihoodfeats(r'C:\Users\milok\tdparse\data\sentihood\testing', r'C:\Users\milok\tdparse\data\sentihood\parses\sentihood.test.conll')
    writevec(r'C:\Users\milok\tdparse\data\sentihood\output\testing',x_test,y_test)        
            
#write_tr_files()
model = TDParse(tr_file=r'C:\Users\milok\tdparse\data\sentihood\output\training')
#model = TDParse(model_file='sentihood_tdparse_clf.pkl',scaler_file='sentihood_tdparse_scaler.pkl')
acc,f1 = model.evaluate(*model.read_tdparse(r'C:\Users\milok\tdparse\data\sentihood\output\testing'))
print(acc,f1)

#Paper result 68.4
#my results: 68.8 acc, 66.6 f1
