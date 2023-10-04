import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
dataset = dataset.drop('Surname', axis=1)
# Data preprocessing
# Handle missing values, encode categorical variables, etc.

# Split the data into features (X) and target (y)
target_column = 'Exited'  # Adjust 'Exited' to match your target column name

# Create features (X) by dropping the target column
X = dataset.drop(target_column, axis=1)

# Create the target (y)
y = dataset[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Feature scaling (if necessary)

# Data preprocessing: Handling categorical encoding
X_train_encoded = pd.get_dummies(X_train, columns=['Geography', 'Gender'])

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)

# Similar preprocessing for the test data
X_test_encoded = pd.get_dummies(X_test, columns=['Geography', 'Gender'])
X_test_scaled = scaler.transform(X_test_encoded)

# Model selection and training
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = rfc.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("**************   Random Forest  \n")
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_rep)

#with open('trained_model.pkl', 'wb') as model_file:
#pickle.dump(rfc, model_file)
