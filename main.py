import matplotlib.pyplot as plt
import numpy as np

from Data import BCData
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from model_selection_procedure import get_best_classifier


def plot_confusion_matrix(cm, classes, normalize=False,  title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(25, 25))
    plt.rcParams.update({'font.size': 25})
    plt.imshow(cm, interpolation='nearest', cmap=cmap, aspect='auto')
    plt.title(title, fontsize=40)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=35)
    plt.xlabel('Predicted label', fontsize=35)
    plt.show()


def predict_test(classifier, X_test, y_test):
    classifier.results(X_test, y_test)
    res_count = {key: 0 for key in classifier.predictions}
    for gene in classifier.predictions:
        res_count[gene] += 1
    plt.figure(figsize=(20, 20))
    plt.bar(res_count.keys(), res_count.values())
    plt.title('Party Prediction', fontsize=40)
    plt.ylabel('Number of predict voters', fontsize=35)
    plt.xlabel('Predicted party', fontsize=35)
    plt.xticks(list((0, 1)), list(('Normal', 'Tumor')), size=20)
    plt.yticks(size=20)
    plt.show()
    return [key for key in res_count.keys() if res_count[key] == max(res_count.values())][0]


if __name__ == "__main__":
    data = BCData('data/BC-TCGA-Normal.txt', 'data/BC-TCGA-Tumor.txt')
    data.set_data(method='median', use_validation=True)
    knn = KNeighborsClassifier(n_neighbors=7)
    # KNN selected features - cv=5
    # ('ELMO2', 'CREB3L1', 'RPS11', 'PNMA1', 'MMP2', 'C10orf90', 'ZHX3', 'ERCC5', 'GPR98', 'RXFP3', 'APBB2',
    # 'PRO0478', 'PRSSL1', 'CADM4', 'HNRPD', 'CFHR5', 'SLC10A7', 'SUHW1', 'GP1BA', 'FLVCR1')

    dtree = DecisionTreeClassifier(max_depth=50, splitter='best')
    # DTree selected features - cv=5
    # ('ELMO2', 'RPS11', 'C10orf90', 'ERCC5', 'GPR98', 'RXFP3', 'PRO0478', 'KLHL13', 'PRSSL1', 'DECR1', 'SALL1',
    # 'HNRPD', 'OR2K2', 'SUHW1', 'CHD8', 'DDB1', 'MMP7', 'ZEB1', 'PROSC', 'PAQR4')

    data.feature_selection(knn, verbose=2)

    best_classifier = get_best_classifier(data.X_train, data.y_train, data.X_val, data.y_val, data.X_test, data.y_test)


