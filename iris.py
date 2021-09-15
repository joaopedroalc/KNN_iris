# UNIVERSIDADE FEDERAL DO MARANHÃO
# DISCENTE: JOÃO PEDRO DE ALCÂNTARA LIMA
# DISCIPLINA: INTELIGÊNCIA ARTIFICIAL (EECP0008)
# MATRÍCULA : 2019004788

from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


iris = datasets.load_iris() 
#PRINT DO DATASET 
data1 = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                columns= iris['feature_names'] + ['target'])
print(data1)

#DIVISÃO DOS DADOS EM TREINAMENTO E TESTE
X, y = iris.data[:, :], iris.target
Xtrain, Xtest, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 0, train_size = 0.7)

scaler = preprocessing.StandardScaler().fit(Xtrain)
Xtrain = scaler.transform(Xtrain)
Xtest = scaler.transform(Xtest)

knn = neighbors.KNeighborsClassifier(n_neighbors=3)
knn.fit(Xtrain, y_train)
y_pred = knn.predict(Xtest)

print("As medidas de desempenho:")
print(classification_report(y_test, y_pred,target_names=iris.target_names))
print("-----------------------------------------------------")
print("A acurácia:")
print(accuracy_score(y_test, y_pred))
print("-----------------------------------------------------")
print("A matriz confusão:")
print(confusion_matrix(y_test,y_pred))