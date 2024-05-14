from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def preprocess_prod_data(df):
    """
    Written for productivity data; will drop null values and 
    split into training and testing sets. Uses actual_productivity
    as the target column.
    """
    
    X = df.copy()
    X = X.drop(columns='class', axis=1)
    y = df['class'].values.reshape(-1, 1)
    
    return train_test_split(X, y)
    


def r2_adj(x, y, model):
    """
    Calculates adjusted r-squared values given an X variable, 
    predicted y values, and the model used for the predictions.
    """
    r2 = model.score(x,y)
#     n = x.shape[0]
#     p = y.shape[1]
#     return 1 - (1 - r2) * (n - 1) / (n - p - 1)

def prod_model_generator(df):
    """
    Defines a series of steps that will preprocess data,
    split data, and train a model for predicting actual productivity
    using linear regression. It will return the trained model
    and print the mean squared error, r-squared, and adjusted
    r-squared scores.
    """
    # Create a list of steps for a pipeline that will one hot encode and scale data
    # Each step should be a tuple with a name and a function
#     steps = [("knn", KNeighborsClassifier(n_neighbors=5)), 
#              ("Linear Regression", LinearRegression()),
#              ("Random Forest Classifier", RandomForestClassifier)] 

    # Create a pipeline object
#     pipeline = Pipeline(steps)

    # Apply the preprocess_rent_data step
    X_train, X_test, y_train, y_test = preprocess_prod_data(df)
    
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    y_train_encoded

    # Fit the pipeline
#     pipeline.fit(X_train, y_train_encoded)

#     # Use the pipeline to make predictions
#     y_pred = pipeline.predict(X_test)
    
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train_encoded)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train_encoded)
    rf_model = RandomForestClassifier(n_estimators=128, random_state=1)
    rf_model.fit(X_train, y_train_encoded)
#     mse = mean_squared_error(y_test, y_pred)
#     r2_value = r2_score(y_test, y_pred)
#     r2_adj_value = r2_adj(X_test, y_test, pipeline)

    # Print out the MSE, r-squared, and adjusted r-squared values
    print('Train Accuracy: %.3f' % lr.score(X_train, y_train_encoded))
    print('Test Accuracy: %.3f' % lr.score(X_test, y_test_encoded))
    print('Train Accuracy: %.3f' % knn.score(X_train, y_train_encoded))
    print('Test Accuracy: %.3f' % knn.score(X_test, y_test_encoded))
    print('Train Accuracy: %.3f' % rf_model.score(X_train, y_train_encoded))
    print('Test Accuracy: %.3f' % rf_model.score(X_test, y_test_encoded))
    

    # Return the trained model
  

if __name__ == "__main__":
    print("This script should not be run directly! Import these functions for use in another file.")

    




