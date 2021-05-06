from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


class KNN:
    predictions = cm = accuracy = knn_fit = ''

    def __init__(self, k=7, method='auto'):
        self.knn = KNeighborsClassifier(n_neighbors=k, algorithm=method)
        self.name = 'KNN'
        self.xlabel = 'K'

    def fit(self, X_train, y_train):
        self.knn_fit = self.knn.fit(X_train, y_train)

    def results(self, X_test, y_test):
        self.predictions = self.knn_fit.predict(X_test)
        self.accuracy = self.knn_fit.score(X_test, y_test)
        self.cm = confusion_matrix(y_test, self.predictions)
