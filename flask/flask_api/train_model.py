from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load Iris data
iris = load_iris()

# Split into train and test sets
X_train, X_test, y_train, y_test = \
    train_test_split(iris['data'], iris['target'], random_state=12)

# Train the model
clf = RandomForestClassifier(random_state=12)
clf.fit(X_train, y_train)

# Make prediction on the test set
y_predict = clf.predict(X_test)
print(y_predict)

# Save model
with open('model.pickle', 'wb') as f:
    pickle.dump(clf, f)