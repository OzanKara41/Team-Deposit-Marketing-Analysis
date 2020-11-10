#import statements for all necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
#make your plot outputs appear and be stored within the notebook.
import sklearn
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

bank_data = pd.read_csv("term-deposit-marketing-2020.csv")
bank_data.info()
bank_data.head()

#drop any duplicates in the data
bank_data = bank_data.drop_duplicates()
bank_data.shape

#distribution of subscribed plot
sns_plot =sns.countplot(x='y',data=bank_data)
plt.savefig('subdistribution_graph.png')

#Count distribution of subscribed
bank_data['y'].value_counts()
#percentage distribution of subscribed
bank_data['y'].value_counts()/bank_data['y'].count()

#Bar plots of categorical features
for feature in bank_data.dtypes[bank_data.dtypes == 'object'].index:
    sns.countplot(x=feature, data=bank_data, order = bank_data[feature].value_counts().index)
    plt.show()

bank_data['month'].value_counts()/bank_data['month'].count()
bank_data['housing'].value_counts()/bank_data['loan'].count()
bank_data['default'].value_counts()/bank_data['default'].count()
bank_data['contact'].value_counts()/bank_data['contact'].count()
bank_data['loan'].value_counts()/bank_data['loan'].count()
sns_plot = sns.countplot(y='job', data=bank_data)
plt.savefig('jobdistribution_graph.png')

#Histogram grid
bank_data.hist(figsize=(15,15))
plt.show()
bank_data.describe()


plot = sns.boxplot(y=bank_data["age"])
plot = sns.boxplot(y=bank_data["balance"])
plot = sns.boxplot(y=bank_data["campaign"])

#count missing values
bank_data.isnull().sum()

bank_data = bank_data.dropna()
bank_data.info()

#percentage distribution of subscribed
bank_data['y'].value_counts()/bank_data['y'].count()

#dropping unnown values in job aatribute
bank_data = bank_data[bank_data.job != 'unknown']

#dropping unnown values in education attribute
bank_data = bank_data[bank_data.education != 'unknown']
bank_data.info()

#percentage distribution of subscribed
bank_data['y'].value_counts()/bank_data['y'].count()
bank_data['y'].value_counts()

# Separate majority and minority classes
unsubscribed = bank_data[bank_data.y =='no']
subscribed= bank_data[bank_data.y =='yes']
subscribed_upsampled = resample(subscribed, replace=True, n_samples=11213, random_state=0)
upsampled_bankdata = pd.concat([unsubscribed, subscribed_upsampled])
upsampled_bankdata.head()

sns_plot =sns.countplot(x='y',data=upsampled_bankdata)
plt.savefig('upsampled_graph.png')

upsampled_bankdata['y'].value_counts()
data_copy = upsampled_bankdata.copy()


# encoding the categorical variables
encoder = preprocessing.LabelEncoder()
upsampled_bankdata['job'] = encoder.fit_transform(upsampled_bankdata['job'])
upsampled_bankdata['marital'] = encoder.fit_transform(upsampled_bankdata['marital'])
upsampled_bankdata['education'] = encoder.fit_transform(upsampled_bankdata['education'])
upsampled_bankdata['default'] = encoder.fit_transform(upsampled_bankdata['default'])
upsampled_bankdata['housing'] = encoder.fit_transform(upsampled_bankdata['housing'])
upsampled_bankdata['loan'] = encoder.fit_transform(upsampled_bankdata['loan'])
upsampled_bankdata['y'] = encoder.fit_transform(upsampled_bankdata['y'])
upsampled_bankdata['contact'] = encoder.fit_transform(upsampled_bankdata['contact'])
upsampled_bankdata['month'] = encoder.fit_transform(upsampled_bankdata['month'])

upsampled_bankdata.head()

X= upsampled_bankdata.iloc[:,0:14]
X[:]
yk = upsampled_bankdata.iloc[:,13]
yk[:]

X_train, X_test, yk_train, yk_test = train_test_split(X, yk, test_size=0.3, random_state=42)
X_train.shape, yk_train.shape
X_test.shape, yk_test.shape

svc_model = SVC()
svc_model.fit(X_train,yk_train)
predictions = svc_model.predict(X_test)

print(confusion_matrix(yk_test,predictions))
accuracy_score(yk_test, predictions)
accuracy_score(yk_train, svc_model.predict(X_train))
print(accuracy_score(yk_test, predictions))
print(accuracy_score(yk_train, svc_model.predict(X_train)))
print(classification_report(yk_test,predictions))

print(cross_val_score(svc_model, X, yk, cv=5))
accuracies = cross_val_score(svc_model, X, yk, cv=5)
print("Accuracy (mean): %",accuracies.mean()*100)
print("std: %",accuracies.std()*100)
