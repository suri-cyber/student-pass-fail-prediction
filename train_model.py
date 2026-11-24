import pandas as pd

print("Step 1: Pandas imported")

data = pd.read_csv('student_pass_fail_practice.csv')

print('convert the data into a DataFrame')

df = pd.DataFrame(data)

print("Step 2: Dataset loaded")

print(df.head())

#independent variables

X = df.drop('Result',axis = 1)

print("Step 3: Independent variables initialized",X.head())
#dependent variable or target variable 

y = df['Result']

print('Step 4: Dependent variable initialized',y.head())

# splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(X_train.head())

#scaling for the training and testing data

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

#creating a model using Logistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train_scaled,y_train)

print("Step 5: Model trained successfully")

# predicting the results for the test data

predictions = model.predict(X_test_scaled)

print("Step 6: Predictions made successfully")

print(predictions)


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

predictions = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", accuracy)
print("Model accuracy percentage:", accuracy * 100, "%")

print("Test accuracy using model.score():",
      model.score(X_test_scaled, y_test))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("Classification Report:")
print(classification_report(y_test, predictions))

#the model is saved as a pickle file so that it can be used in the flask app
import pickle 

with open('model.pkl', 'wb') as file:
  pickle.dump(model,file)

with open('scaler.pkl','wb') as file:
  pickle.dump(scaler,file)

print('model and scaler saved as pickle files successfully')

