# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries for data handling, modeling, and evaluation.

2.Load the Iris dataset using load_iris().

3.Convert data to DataFrame and add the target column.

4.Split data into features (X) and target (y).

5.Split X and y into training and testing sets using train_test_split().

6.Initialize the SGDClassifier with specified parameters.

7.Train the model on the training data.

8.Predict the target values for the test data.

9.Evaluate the model using accuracy score, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the prediction of iris species using SGD Classifier.
Developed by: Rogith J
RegisterNumber: 212224040280
 import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print("Rogith J ")
print("212224040280")
print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)

sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Rogith J")
print("212224040280")
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Rogith J")
print("212224040280")
print("Confusion Matrix:")
print(cm)

classification_report1 = classification_report(y_test,y_pred)
print("Rogith J")
print("212224040280")
print(classification_report1)
*/
```

## Output:

<img width="773" height="320" alt="Screenshot 2025-09-25 091129" src="https://github.com/user-attachments/assets/6d93cb24-6cc5-43fd-b823-ee82b9d3b8d5" />

<img width="370" height="57" alt="Screenshot 2025-09-25 091137" src="https://github.com/user-attachments/assets/ff848a29-b688-4f66-957e-655aa3632d8f" />





<img width="575" height="124" alt="Screenshot 2025-09-25 091145" src="https://github.com/user-attachments/assets/4a14b05c-05ce-4bbf-bb3c-84ed5315166a" />

<img width="640" height="265" alt="Screenshot 2025-09-25 091152" src="https://github.com/user-attachments/assets/73f45f01-cf44-41ea-8acc-9bc9f845f687" />







## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
