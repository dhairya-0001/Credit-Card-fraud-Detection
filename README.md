# Credit-Card-fraud-Detection

### Importing Dependencies
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```
- `numpy` and `pandas`: Used for data manipulation and analysis.
- `train_test_split`: Used to split the dataset into training and testing sets.
- `LogisticRegression`: Logistic regression model from scikit-learn.
- `accuracy_score`: Used to evaluate the accuracy of the model.

### Loading and Exploring the Dataset
```python
# Load the dataset into a pandas DataFrame
credit_card_data = pd.read_csv('/content/creditcard.csv')
```
- Loads the credit card fraud dataset from a CSV file into a pandas DataFrame.

```python
# Display the first 5 rows of the dataset
credit_card_data.head()
```
- Displays the first five rows to give an overview of the data.

```python
# Display the last 5 rows of the dataset
credit_card_data.tail()
```
- Displays the last five rows of the dataset.

```python
# Display the shape of the dataset (number of rows and columns)
credit_card_data.shape
```
- Prints the shape of the DataFrame, showing the number of rows and columns.

```python
# Get more information about the dataset
credit_card_data.info()
```
- Provides a concise summary of the DataFrame, including data types and non-null counts.

```python
# Check the number of missing values in each column
credit_card_data.isnull().sum()
```
- Checks for any missing values in each column of the DataFrame.

```python
# Distribution of legitimate transactions and fraudulent transactions
credit_card_data['Class'].value_counts()
```
- Displays the count of legitimate (`Class=0`) and fraudulent (`Class=1`) transactions. This shows the dataset is highly unbalanced.

### Data Segmentation
```python
# Separate the data into legitimate and fraudulent transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)
```
- Splits the dataset into two separate DataFrames: one for legitimate transactions and one for fraudulent transactions.
- Prints the shapes of these DataFrames to show the number of rows in each.

### Statistical Measures
```python
# Statistical measures of the legitimate transactions
legit.Amount.describe()
```
- Displays the statistical summary (mean, standard deviation, min, max, etc.) of the `Amount` column for legitimate transactions.

```python
# Statistical measures of the fraudulent transactions
fraud.Amount.describe()
```
- Displays the statistical summary of the `Amount` column for fraudulent transactions.

```python
# Compare the values of both transaction types
credit_card_data.groupby('Class').mean()
```
- Groups the data by the `Class` column and calculates the mean of each feature for both legitimate and fraudulent transactions.

### Under-Sampling
```python
# Build a sample dataset with a similar distribution of legitimate and fraudulent transactions
# Number of fraudulent transactions is 492
legit_sample = legit.sample(n=492)
```
- Creates a sample of legitimate transactions to match the number of fraudulent transactions (492).

```python
# Concatenate the legitimate sample and the fraudulent transactions to form a new dataset
new_dataset = pd.concat([legit_sample, fraud], axis=0)

new_dataset.head()
```
- Concatenates the sampled legitimate transactions with the fraudulent transactions to create a balanced dataset.

```python
# Display the first few rows of the new dataset
new_dataset.head()
```
- Displays the first few rows of the new, balanced dataset.

```python
# Display the last few rows of the new dataset
new_dataset.tail()
```
- Displays the last few rows of the new dataset.

```python
# Count the number of transactions in each class in the new dataset
new_dataset['Class'].value_counts()
```
- Displays the count of legitimate and fraudulent transactions in the new dataset.

```python
# Mean values of each class in the new dataset
new_dataset.groupby('Class').mean()
```
- Groups the new dataset by `Class` and calculates the mean of each feature for both legitimate and fraudulent transactions.

### Splitting the Data into Features and Target
```python
# Separate the features and the target
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']

print(X)
print(Y)
```
- Splits the new dataset into features (`X`) and target (`Y`).
- `X` contains all columns except `Class`.
- `Y` contains the `Class` column.

### Splitting the Data into Training and Testing Sets
```python
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)
```
- Splits the data into training (80%) and testing (20%) sets.
- The `stratify=Y` parameter ensures that the class distribution in the training and test sets is similar to the original distribution.
- `random_state=2` ensures reproducibility of the split.
- Prints the shapes of the original, training, and test sets.

### Model Training: Logistic Regression
```python
# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model with the training data
model.fit(X_train, Y_train)
```
- Initializes a Logistic Regression model.
- Trains the model using the training data (`X_train` and `Y_train`).

### Model Evaluation
```python
# Predictions on the training data
X_train_prediction = model.predict(X_train)
# Calculate the accuracy on the training data
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy on Training data: ", training_data_accuracy)
```
- Makes predictions on the training data.
- Calculates and prints the accuracy score for the training data.

```python
# Predictions on the test data
X_test_prediction = model.predict(X_test)
# Calculate the accuracy on the test data
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy score on Test Data: ", test_data_accuracy)
```
- Makes predictions on the test data.
- Calculates and prints the accuracy score for the test data.

### Summary
- This code provides a step-by-step process for building a credit card fraud detection model using Logistic Regression.
- It starts with loading and exploring the dataset, followed by data preprocessing, including under-sampling to handle class imbalance.
- The data is then split into training and testing sets.
- A Logistic Regression model is trained and evaluated using accuracy as the performance metric.
