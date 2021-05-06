from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import radviz
from pandas.plotting import lag_plot
import warnings

data = pd.read_csv("../input/zoo-animal-classification/zoo.csv")
data.head(6)

data.plot.bar(title="Zoo Animal Classification",figsize=(20,10))

data.plot(x="eggs",y="milk")

color = {"boxes": "DarkGreen","whiskers": "DarkOrange", "medians": "DarkBlue","caps": "Gray",}
data.plot.box(color=color, sym="r+",figsize=(20,10))

label_encoder = LabelEncoder()
data.iloc[:,0] = label_encoder.fit_transform(data.iloc[:,0]).astype('float64')

plt.figure(figsize=(12,10))
corr = data.corr()
sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
plt.show()

columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = data.columns[columns]
data = data[selected_columns]
data.head(5)

def preprocess(data):
    X = data.iloc[:, 1:17]  # all rows, all the features and no labels
    y = data.iloc[:, 17]  # all rows, label only
    return X, y

# Shuffle and split the dataset

data = data.sample(frac=1).reset_index(drop=True)
print("Data",data)
data_total_len = data[data.columns[0]].size
print("Length",data_total_len)
data_train_frac = 0.8
split_index = math.floor(data_total_len*data_train_frac)

train_data = data.iloc[:split_index]
eval_data = data.iloc[split_index:]

print(train_data,"\nE ", eval_data)

train_X, train_Y = preprocess(train_data)
test_X,test_Y = preprocess(eval_data)

print(train_X, train_Y,test_X,test_Y)

print(train_X.shape, train_Y.shape,test_X.shape,test_Y.shape)

clf = LogisticRegression()
clf.fit(train_X, train_Y)
clf.score(test_X,test_Y)
clf.predict(test_X[1:25])

kn = KNeighborsClassifier()
kn.fit(train_X, train_Y)
kn.score(test_X,test_Y)
kn.predict(test_X[1:25])

nb = GaussianNB()
nb.fit(train_X, train_Y)
nb.score(test_X,test_Y)
nb.predict(test_X[1:25])

# Show what the correct answer is
test_Y[1:25]

error = []

for i in range(1, 100):
    pred_i = clf.predict(test_X)
    error.append(np.mean(pred_i != test_Y))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 100), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.ylabel('Mean Error')
warnings.filterwarnings('ignore', category=FutureWarning, append=True)

error = []

for i in range(1, 80):
    pred_i = kn.predict(test_X)
    error.append(np.mean(pred_i != test_Y))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 80), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.ylabel('Mean Error')
warnings.filterwarnings('ignore', category=FutureWarning, append=True)

error = []

for i in range(1, 80):
    pred_i = nb.predict(test_X)
    error.append(np.mean(pred_i != test_Y))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 80), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.ylabel('Mean Error')
warnings.filterwarnings('ignore', category=FutureWarning, append=True)

hair = int(input(" Value of hair: "))
feathers = int(input(" Value of feathers: "))
eggs = int(input(" Value of eggs :"))
milk  = int(input(" Value of milk:"))
airborne  = int(input(" Value of airborne: "))
aquatic  = int(input(" Value of aquactic: "))
predator  = int(input(" Value of predator: "))
toothed  = int(input(" Value of toothed: "))
backbone  = int(input(" Value of backbone: "))
breathes  = int(input(" Value of breathes: "))
venomous  = int(input(" Value of venomous: "))
fins  = int(input(" Value of fins:"))
legs  = int(input(" Value of legs:"))
tail  = int(input(" Value of tail:"))
domestic = int(input(" Value of domestic :"))
catsize = int(input(" Value of catsize :"))
data=[[hair,feathers,eggs,milk,airborne,aquatic,predator,toothed,backbone,breathes,venomous,fins,legs,tail,domestic,catsize]]
df = pd.DataFrame(data, columns = ['hair','feathers','eggs','milk','airborne','aquatic','predator','toothed','backbone','breathes','venomous','fins','legs','tail','domestic','catsize']) 
df

clf.predict(df)

kn.predict(df)

nb.predict(df)

class_type = pd.read_csv("../input/zoo-animal-classification/class.csv")
model_prediction = [clf,kn,nb]
for i in model_prediction:
    for j in i.predict(df):
        print(class_type.loc[class_type['Class_Number'] == j, 'Class_Type'])

        
