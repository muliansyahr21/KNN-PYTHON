import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings("ignore", category=DataConversionWarning)



# Importing the dataset
data = pd.read_csv('data.csv')
X = data.iloc[:, 0:4].values
y = data.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting K-NN to the Training set
classifier = KNeighborsClassifier(n_neighbors=5, metric='euclidean', p=2)
classifier.fit(X_train, y_train)

# Predicting test set results
y_pred = classifier.predict(X_test)
y_score = classifier.score(X_train, y_train)

# Creating confusion matrix
cm = confusion_matrix(y_test, y_pred)

print(y_pred)
print(y_score)
print(classifier)

df_confusion = pd.crosstab(y_test, y_pred)
print(df_confusion)



