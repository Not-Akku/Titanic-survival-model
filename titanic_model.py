import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('archive/train.csv')

# ////CLEAN THE DATASET////

# drop the unwanted columns...
df = df.drop(columns = ['PassengerId', 'Name', 'Ticket', 'Cabin'])
# delete missing rows...
df = df.dropna()
# Embarked is a string, change it to integer...
df['Embarked'] = df['Embarked'].replace({'S': 0,
                                         'C': 1,
                                         'Q': 2})

# change male and female columns to integers...
df['Sex'] = df['Sex'].replace({'male' : 0, 'female': 1})

# delete duplicates if there is any...
df = df.drop_duplicates()

# //// SCIKIT LEARN ////

# prepare features, x have all feature except survived column
X = df.drop(columns=['Survived'])
# y have survived column or series
y = df['Survived']

# splitting the data into train and test datas
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 321)


# initialize the model
#model = DecisionTreeClassifier(max_depth=3) this model is good however logistic regression performs better
model = LogisticRegression()
#print(df)

# train the model
model.fit(x_train, y_train)

# get score
score = model.score(x_test, y_test)
print("Model Accuracy:", score)