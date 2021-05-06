from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


class DTrees:
    predictions = cm = accuracy = dtree_model_fit = ''

    def __init__(self, depth=50, method='best'):
        self.dtree_model = DecisionTreeClassifier(max_depth=depth, splitter=method)
        self.name = 'Decision Tree'
        self.xlabel = 'Depth'

    def fit(self, X_train, y_train):
        self.dtree_model_fit = self.dtree_model.fit(X_train, y_train)

    def results(self, X_test, y_test):
        self.predictions = self.dtree_model_fit.predict(X_test)
        self.accuracy = self.dtree_model_fit.score(X_test, y_test)
        self.cm = confusion_matrix(y_test, self.predictions)
