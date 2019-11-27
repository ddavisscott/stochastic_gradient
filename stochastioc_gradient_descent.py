import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Sigmoid Function
def sig(w,x):
    d=np.dot(x,w)
    t_pred = 1/(1+np.exp(-d))
    return t_pred

# Log Likelihood
def loglikelihood(w,x,t):
    l=0
    for i in range(len(t)):
        l += t[i]*np.log(sig(w,x[i,:]))+(1-t[i])*np.log(1-sig(w,x[i,:]))
    return l

# Derivative of the Loss Function
def dLdW(w,x,t):
    t_pred = sig(w,x)
    return (t_pred-t)*x

# Read datasets and make normalized version
dataset = pd.read_csv('diabetes_2.csv')
scaler = MinMaxScaler()
scaler.fit(dataset)
scaler.data_max_
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
normalized_dataset = pd.DataFrame(scaler.transform(dataset), columns=column_names)

# Split Unnormalized Data
X = dataset.drop('Outcome', axis=1)
t = dataset['Outcome']
train_X, test_X, train_t, test_t = train_test_split(X, t, test_size=0.2, stratify=t)
del dataset, X, t
column_names = ['Bias', 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
train_X = pd.DataFrame(np.concatenate((np.ones(train_X.shape[0]).reshape(-1,1), train_X.values), axis=1),columns = column_names)
train_t = pd.DataFrame(train_t.values,columns = ['Outcome'])
test_X = pd.DataFrame(np.concatenate((np.ones(test_X.shape[0]).reshape(-1,1), test_X.values), axis=1),columns = column_names)
test_t = pd.DataFrame(test_t.values,columns = ['Outcome'])

train_X = train_X.values
train_t = train_t.values
test_X = test_X.values
test_t = test_t.values

l = train_X.shape[0]
X = np.concatenate((train_X,test_X), axis=0)
scaler.fit(X)
scaler.data_max_
X = scaler.transform(X)
norm_train_X = X[0:l,:]
norm_test_X = X[l:,:]
norm_train_t = train_t
norm_test_t = test_t

# Initialize weights and Likelihood values
w = np.random.random(train_X.shape[1])
likelihood = np.array([])
ind = np.array([])

# Stochastic Gradient Descent
threshold = 1e-4
nsteps = train_t.shape[0]
alpha = 1e-4
l_old = -np.inf
l_new = -np.random.random(1)
start = time.time()
epochs = 0
while (abs(l_old - l_new) > threshold) or (np.isnan(l_old) or np.isnan(l_new)):
    for i in range(nsteps):
        w = w - alpha * dLdW(w.reshape(-1), train_X[i,:], train_t[i])
    epochs += 1
    ind = np.append(ind,epochs)
    l_old = l_new
    l_new = loglikelihood(w.reshape(-1),train_X,train_t)/nsteps
    likelihood = np.append(likelihood,l_new)
end = time.time()
runtime = end-start
pred_train_t = (sig(w.reshape(-1),train_X) > 0.5).astype(int)
pred_test_t = (sig(w.reshape(-1),test_X) > 0.5).astype(int)
plt.plot(ind,likelihood,'-')
print('Epochs:', epochs)
print('Runtime:', runtime)
print('Log Likelihood of Training Data:', loglikelihood(w.reshape(-1), train_X, train_t)/nsteps)
print('Log Likelihood of Test Set:', loglikelihood(w.reshape(-1), test_X, test_t)/nsteps)
print('Error on Training Data:', mean_squared_error(train_t, pred_train_t))
print('Error on Test Data:', mean_squared_error(test_t, pred_test_t))
w = pd.DataFrame(w.reshape(1,-1),columns = column_names)
w.head()
del likelihood, ind, threshold, nsteps, alpha, epochs, l_old, l_new, i, pred_train_t, pred_test_t

# Initialize weights and Likelihood values
w = np.random.random(norm_train_X.shape[1])
likelihood = np.array([])
ind = np.array([])

# Stochastic Gradient Descent
threshold = 1e-4
nsteps = norm_train_t.shape[0]
alpha = 1e-4
epochs = 0
l_old = -np.inf
l_new = -np.random.random(1)
start = time.time()
# for e in range(200):
while (abs(l_old - l_new) > threshold) or (np.isnan(l_old) or np.isnan(l_new)):
    for i in range(nsteps):
        w = w - alpha * dLdW(w.reshape(-1), norm_train_X[i,:], norm_train_t[i])
    epochs += 1
    ind = np.append(ind,epochs)
    l_old = l_new
    l_new = loglikelihood(w.reshape(-1),norm_train_X,norm_train_t)/nsteps
    likelihood = np.append(likelihood,l_new)
end = time.time()
print(l_old, l_new)
print('Runtime:', end-start)
pred_train_t = (sig(w.reshape(-1),norm_train_X) > 0.5).astype(int)
pred_test_t = (sig(w.reshape(-1),norm_test_X) > 0.5).astype(int)
plt.plot(ind,likelihood,'-')
print('Epochs:', epochs)
print('Log Likelihood of Training Data:', loglikelihood(w.reshape(-1), norm_train_X, norm_train_t)/nsteps)
print('Log Likelihood of Test Set:', loglikelihood(w.reshape(-1), norm_test_X, norm_test_t)/nsteps)
print('Error on Training Data:', mean_squared_error(norm_train_t, pred_train_t))
print('Error on Test Data:', mean_squared_error(norm_test_t, pred_test_t))
w = pd.DataFrame(w.reshape(1,-1),columns = column_names)
w.head()

del likelihood, ind, threshold, nsteps, alpha, epochs, l_old, l_new, i, pred_train_t, pred_test_t