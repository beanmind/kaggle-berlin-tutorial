__author__ = 'sabine'

import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/train.csv')

#cleaning training data

df = df.drop(['Name', 'Ticket', 'Cabin'], axis=1)
print df.info()

# filling in the age with the median of the ages per class and gender
med_age = np.zeros((2,3))
df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

for i in range(0, 2):
    for j in range(0, 3):
        med_age[i,j] = df[(df['Gender'] == i)&(df['Pclass'] == j+1)]['Age'].dropna().median()
print med_age

for i in range(0, 2):
    for j in range(0, 3):
        df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1), 'Age'] = med_age[i,j]
#simpler possibility
#age_mean = df['Age'].mean()
#df['Age'] = df['Age'].fillna(age_mean)


mode_embarked = mode(df['Embarked'])[0][0]

# replacing Nan by mode = most frequently occurring value
df['Embarked'] = df['Embarked'].fillna(mode_embarked)

df['Port'] = df['Embarked'].map({'C': 1, 'S': 2, 'Q': 3}).astype(int)

# other possibility:
# df['Gender'] = df['Sex'].apply(lambda x:1 if x=='male' else 0).astype(int)

df = df.drop(['Sex', 'Embarked'], axis=1)
cols = df.columns.tolist()
cols = [cols[1]] + cols[0:1] + cols[2:]
df = df[cols]

train_data = df.values

# training model

model = RandomForestClassifier(n_estimators = 100)
model = model.fit(train_data[0:,2:],train_data[0:,0])


#cleaning test data
print "Loading test data."
df_test = pd.read_csv('data/test.csv')
print df_test.info()

print "Cleaning test data"
df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis=1)
df_test['Gender'] = df_test['Sex'].map({'female': 0, 'male': 1}).astype(int)

# age fill in
for i in range(0, 2):
    for j in range(0, 3):
        df_test.loc[(df_test.Age.isnull()) & (df_test.Gender == i) & (df_test.Pclass == j+1), 'Age'] = med_age[i,j]

# fare fill in
med_fare = np.zeros((3, 1))
for i in range(0, 3):
    med_fare[i] = df[(df['Pclass'] == i+1)]['Fare'].dropna().median()
print med_fare

for i in range(0, 3):
    df_test.loc[(df_test.Fare.isnull()) & (df_test.Pclass == i+1), 'Fare'] = med_fare[i]

#other possibility
#df_test['Age'] = df_test['Age'].fillna(age_mean)
#fare_means = df.pivot_table('Fare', index='Pclass', aggfunc='mean')
#df_test['Fare'] = df_test[['Fare', 'Pclass']].apply(lambda x:fare_means[x['Pclass']] if pd.isnull(x['Fare'])else x['Fare'], axis=1)


df_test['Port'] = df_test['Embarked'].map({'C': 1, 'S': 2, 'Q': 3})
df_test = df_test.drop(['Sex', 'Embarked'], axis=1)

test_data = df_test.values
output = model.predict(test_data[:, 1:])

#submission

result = np.c_[test_data[:, 0].astype(int), output.astype(int)]
df_result = pd.DataFrame(result[:, 0:2], columns=['PassengerId', 'Survived'])

df_result.to_csv('results/titanic_1-2.csv', index=False)

