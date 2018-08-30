import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

seed=42

features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']

# fetch data
training_data = pd.read_csv("train.csv")
X = training_data[features]
y = training_data.Survived

# test_data = pd.read_csv("test.csv")
# X_test = test_data[features]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

# encode female and male as 0 and 1
le_sex = LabelEncoder()
X_train.loc[:, ['Sex']] = le_sex.fit_transform(X_train.Sex)
X_test.loc[:, ['Sex']] = le_sex.transform(X_test.Sex)

# convert nan in Embarked column to a character (Z), then numerically encode labels (imputer only works on numeric data)
le_embarked = LabelEncoder()
X_train.loc[X_train.Embarked.isna(), ['Embarked']] = 'Z'    # replace nan with Z
X_train.loc[:, ['Embarked']] = le_embarked.fit_transform(X_train.Embarked)  # replace Z with 3
X_test.loc[X_test.Embarked.isna(), ['Embarked']] = 'Z'    # replace nan with Z
X_test.loc[:, ['Embarked']] = le_embarked.transform(X_test.Embarked)    # replace Z with 3

# fill missing values in Embarked column with most frequently appearing port of embarkation
imputer_embarked = Imputer(missing_values=3, strategy='most_frequent')
X_train.loc[:, ['Embarked']] = imputer_embarked.fit_transform(X_train.Embarked.values.reshape(-1, 1))
X_test.loc[:, ['Embarked']] = imputer_embarked.transform(X_test.Embarked.values.reshape(-1, 1))

# fill missing Age values with mean age
imputer_age = Imputer(missing_values='NaN', strategy='mean')
X_train.loc[:, ['Age']] = imputer_age.fit_transform(X_train.Age.values.reshape(-1, 1))
X_test.loc[:, ['Age']] = imputer_age.transform(X_test.Age.values.reshape(-1, 1))

# fill missing Fare values with mean fare
imputer_fare = Imputer(missing_values='NaN', strategy='mean')
X_train.loc[:, ['Fare']] = imputer_fare.fit_transform(X_train.Fare.values.reshape(-1, 1))
X_test.loc[:, ['Fare']] = imputer_fare.transform(X_test.Fare.values.reshape(-1, 1))

# perform one hot encoding on categorical columns
X_train = pd.get_dummies(X_train, drop_first=True, columns=['Pclass', 'Sex', 'Embarked'])
X_test = pd.get_dummies(X_test, drop_first=True, columns=['Pclass', 'Sex', 'Embarked'])

# start training
model = RandomForestClassifier(n_estimators=200, min_samples_leaf=15, random_state=seed)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("F1 score: {}".format(f1_score(y_test, y_pred)))
print("Accuracy score: {}".format(accuracy_score(y_test, y_pred)))

# my_submission = pd.DataFrame({'PassengerId': X_test.PassengerId, 'Survived': y_pred})
# my_submission.to_csv('submission.csv', index=False)