import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# Task 2: Load and Explore Dataset
 
import pandas as pd

data = pd.read_csv("student.csv")
print(data.head())
print(data.describe())

# Task 3: Preprocess Data
 
data = data.dropna()
X = data[['study_hours', 'attendance', 'previous_score']]
y = data['result']

# Task 4: Train ML Model
 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Predictions:", predictions)
print("Accuracy:", accuracy)
