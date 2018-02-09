
# coding: utf-8

# # Titanic Survival Prediction
# 
# ### Contents:
# 
# 1. Import necessary libraries
# 2. Read in and Explore the data
# 3. Data Analysis
# 4. Data Visualization
# 5. Cleaning the data
# 6. Choosing the Best Model
# 7. Creating the submission file
# 
# feedback is welcome!

# ## 1) Import Necessary Libraries
# Importing several python libraries like numpy, pandas, matplotlib and seaborn
# 

# In[248]:

#data analysis libraries 
import numpy as np
import pandas as pd

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# ## 2) Read in and Explore the data
# Reading our training and testing data and taking a look at the training data using `describe()` function

# In[249]:

#import train and test CSV files
train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")

#take a look at the training data
train.describe(include="all")


# ## 3) Data Analysis
# Looking into the usefulness of the features provided in the dataset

# In[250]:

#list of the features within the dataset
print(train.columns)


# * **Numerical Features:** Age(Continuous), Fare(Continuous), SibSp(Discrete), Parch(Discrete)
# * **Categorical Features:** Survived, Sex, Embarked, Pclass
# * **Alphanumeric Features:** Ticket, Cabin
# 
# #### Data Types of each feature
# * Survived: int
# * Pclass: int
# * Name: string
# * Sex: string
# * Age: float
# * SibSp: int
# * Parch: int
# * Ticket: string
# * Fare: float
# * Cabin: string
# * Embarked: string

# In[251]:

#looking into sample of the dataset to get an idea of the variables
train.sample(5)


# In[252]:

#summary of the training dataset
train.describe(include = "all")


# #### Some observations:
# ##### Missing features in %
# 
# * Age: 19.8%
#  **conclusion:** important for guessing the survival rate so we try to fill gaps
#  
# * Cabin:77.1%
#  **conclusion:** most of data are missing so we will drop this feature
#  
# * Embarked:0.22%
#  **conclusion:** this feature will also be taken into account

# In[253]:

#checking for any other unusable values
print(pd.isnull(train).sum())


# ### Some Predictions
# * Sex: Females are more likey to survive.
# * SibSp/Parch: People travelling alone are more likely to survive
# * Age: Young children are more likely to survive
# * Pclass: People of higher socioeconomic class are more likely to survive

# ## 4) Data Visualization
# Visualizing our data so we can see whether our predictions were accurate! or not

# ### Sex Feature

# In[254]:

#drawing a bar plot of survival by sex
sns.barplot(x="Sex", y="Survived", data=train)

#printing percentages of females vs. males that survive
print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)


# As predicted earlier females have a much higher chance of survival than males. Thus we conclude that sex feature is essential in our predictions.

# ### Pclass Feature

# In[255]:

#drawing a bar plot of survival by Pclass
sns.barplot(x="Pclass", y="Survived", data=train)

#printing percentage of people by Pclass that survived
print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)


# In general, its clear that people with more siblings or spouses aboard were less likely to survive.However, people with no siblings or spouses were less likey to survive than those with one or two. 

# ### SibSp Feature

# In[256]:

#drawing a bar plot for SibSp vs. survival
sns.barplot(x="SibSp", y="Survived", data=train)

print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)


# ### Parch Feature

# In[257]:

#drawing a bar plot for Parch vs. survival
sns.barplot(x="Parch", y="Survived", data=train)
plt.show()


# People with less than 4 parents or children aboard are more likey to survive than those with with four or more. Again, people travelling alone are less likey to survive than those with 1-3 parents or children

# ### Age Feature

# In[258]:

#sorting the ages into logical categories
train["Age"] = train["Age"].fillna(-0.5)
test["Age"] = test["Age"].fillna(-0.5)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)
test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)

#drawing a bar plot of Age vs. survival
sns.barplot(x="AgeGroup", y="Survived", data=train)
plt.show()


# Babies are more likely to survive than any other age groups

# ### Cabin Feature

# Those who have recorded cabin numbers are of higher socioeconomic class and thus more likely to survive

# In[259]:

train["CabinBool"] = (train["Cabin"].notnull().astype('int'))
test["CabinBool"] = (test["Cabin"].notnull().astype('int'))

#calculating percentages of CabinBool vs. survived
print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)
#drawing a bar plot of CabinBool vs. survival
sns.barplot(x="CabinBool", y="Survived", data=train)
plt.show()


# People with recorded cabin numbers are, in fact, more likely to survive

# ## 5) Cleaning the data

# ### Looking into test data

# In[260]:

test.describe(include="all")


# ### Cabin Feature

# In[261]:

#dropping the Cabin feature since not a lot more useful information can be extracted from it.
train = train.drop(['Cabin'], axis = 1)
test = test.drop(['Cabin'], axis = 1)


# In[262]:

#droping the Ticket feature since it's unlikely to yield any useful information
train = train.drop(['Ticket'], axis = 1)
test = test.drop(['Ticket'], axis = 1)


# ### Embarked Feature

# In[263]:

#filling in the missing values in the Embarked feature
print("Number of people embarking in Southampton (S):")
southampton = train[train["Embarked"] == "S"].shape[0]
print(southampton)

print("Number of people embarking in Cherbourg (C):")
cherbourg = train[train["Embarked"] == "C"].shape[0]
print(cherbourg)

print("Number of people embarking in Queenstown (Q):")
queenstown = train[train["Embarked"] == "Q"].shape[0]
print(queenstown)


# Since majoirity of people were from southhampton, we will fill in missing values with S

# In[264]:

#replacing the missing values in the Embarked feature with S
train = train.fillna({"Embarked": "S"})


# ### Age Feature

# In[265]:

#creating a combined group of both datasets
combine = [train, test]

#extracting a title for each Name in the train and test datasets
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train['Title'], train['Sex'])


# In[266]:

#replacing various titles with more common names
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()


# In[267]:

#mapping each of the title groups to a numerical value
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train.head()


# In[268]:

# filling missing age with mode age group for each title
mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult
miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student
mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult
master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby
royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult
rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult

age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}


for x in range(len(train["AgeGroup"])):
    if train["AgeGroup"][x] == "Unknown":
        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]
        
for x in range(len(test["AgeGroup"])):
    if test["AgeGroup"][x] == "Unknown":
        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]


# In[269]:

#mapping each Age value to a numerical value
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}
train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
test['AgeGroup'] = test['AgeGroup'].map(age_mapping)

train.head()

#dropping the Age feature for now, might change
train = train.drop(['Age'], axis = 1)
test = test.drop(['Age'], axis = 1)


# In[270]:

#dropping the name feature since it contains no more useful information.
train = train.drop(['Name'], axis = 1)
test = test.drop(['Name'], axis = 1)


# ### Sex Feature

# In[271]:

#mapping each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
train['Sex'] = train['Sex'].map(sex_mapping)
test['Sex'] = test['Sex'].map(sex_mapping)

train.head()


# ### Embarked Feature

# In[272]:

#mapping each Embarked value to a numerical value
embarked_mapping = {"S": 1, "C": 2, "Q": 3}
train['Embarked'] = train['Embarked'].map(embarked_mapping)
test['Embarked'] = test['Embarked'].map(embarked_mapping)

train.head()


# ### Fare Feature

# In[273]:

#filling in missing Fare value in test set based on mean fare for that Pclass 
for x in range(len(test["Fare"])):
    if pd.isnull(test["Fare"][x]):
        pclass = test["Pclass"][x] #Pclass = 3
        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)
        
#mapping Fare values into groups of numerical values
train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])
test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])

#dropping Fare values
train = train.drop(['Fare'], axis = 1)
test = test.drop(['Fare'], axis = 1)


# In[274]:

#check train data
train.head()


# In[275]:

#check test data
test.head()


# ## 6) Choosing the best Model

# ### Splitting the Training data
# We will use part of our training data (22%) to test the accuracy of our different models

# In[276]:

from sklearn.model_selection import train_test_split

predictors = train.drop(['Survived', 'PassengerId'], axis=1)
target = train["Survived"]
x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)


# ### Testing several ML algorithms
# 
# * Gaussian Naive Bayes
# * Logistic Regression
# * Support Vector Machines
# * Perceptron
# * Decision Tree Classifier
# * Random Forest Classifier
# * KNN or k-Nearest Neighbors
# * Stochastic Gradient Descent
# * Gradient Boosting Classifier
# 
# For each model, we fit it with 80% of training data and predict for 20% of the training data and check the accuracy
# 

# In[278]:

# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)


# In[279]:

# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)


# In[280]:

# Support Vector Machines
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)


# In[281]:

# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)


# In[282]:

# Perceptron
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)


# In[283]:

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)


# In[284]:

# Random Forest
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)


# In[285]:

# KNN or k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_knn)


# In[286]:

# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)


# In[287]:

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)


# In[288]:

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, 
              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbk]})
models.sort_values(by='Score', ascending=False)

