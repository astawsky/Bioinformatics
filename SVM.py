from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix


class SVM:
    predictions = cm = accuracy = svm_fit = ''

    def __init__(self, gamma='scale', method='linear'):
        self.svm = SVC(gamma='scale', kernel=method)
        self.name = 'SVM'
        self.xlabel = 'gamma'

    def fit(self, X_train, y_train):
        self.svm_fit = self.svm.fit(X_train, y_train)

    def results(self, X_test, y_test):
        self.predictions = self.svm_fit.predict(X_test)
        self.accuracy = self.svm_fit.score(X_test, y_test)
        self.cm = confusion_matrix(y_test, self.predictions)
