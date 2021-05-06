from os import makedirs

from SVM import *
from KNN import *
from DTrees import *
import matplotlib.pyplot as plt

output_dir = 'model selection results/'


def plot_classifier(classifier, methods, train, test, iter_range, val_test):
    accuracy = []
    X_train = train[0]
    y_train = train[1]
    X_test = test[0]
    y_test = test[1]
    # X_train = train[features].as_matrix()
    # y_train = train['Vote'].astype('category').cat.rename_categories(
    #     range(train['Vote'].nunique())).astype(int).as_matrix()
    # X_validation = test[features].as_matrix()
    # y_validation = test['Vote'].astype('category').cat.rename_categories(
    #     range(test['Vote'].nunique())).astype(int).as_matrix()
    best_params = {}
    for method in methods:
        for i in iter_range:
            classifier_obj = classifier(i, method)
            classifier_obj.fit(X_train, y_train)
            classifier_obj.results(X_test, y_test)
            accuracy.append(classifier_obj.accuracy)
        plt.figure(figsize=(20, 20))
        plt.rcParams.update({'font.size': 30})
        plt.plot(iter_range, accuracy)
        plt.title('{} Classifier - Method {}\n{}'.format(classifier_obj.name, method, val_test), fontsize=40)
        plt.xlabel('{}'.format(classifier_obj.xlabel), fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        max_accuracy = max(accuracy)
        optimal_x = list(iter_range)[accuracy.index(max_accuracy)]
        plt.text(optimal_x, max_accuracy, '({}, {:.3f})'.format(optimal_x, max_accuracy), size='xx-small', fontsize=20)
        plt.savefig('{0}{1}.png'.format(output_dir, '{} Classifier Method {}{}'.format(classifier_obj.name, method, val_test)))
        plt.show()
        accuracy = []
        best_params[method] = {'optimal_x': optimal_x, 'max_accuracy': max_accuracy, 'classifier': classifier_obj.name}
    return best_params


def get_best_classifier(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    This function iterate over two models, KNN and Decision Tree. Finds out which model have the heights score
    for several params:
        - KNN:
            * Iterate the number of neighbors, all the odds numbers between 1 and 50.
            * Algorithm used to compute the nearest neighbors:
                # ‘ball_tree’ will use BallTree for organizing points in a multi-dimensional space.
                # ‘kd_tree’ will use KDTree for organizing some number of points in a space with k dimensions.
                # ‘brute’ will use a brute-force search.
                # ‘auto’ will attempt to decide the most appropriate algorithm based on the values.
        - Decision Tree:
            * Iterate depths for the tree, from 1 to 100
            * Splitter The strategy used to choose the split at each node:
                # “best” to choose the best split.
                # “random” to choose the best random split.
    The code will generate plots of each method (algorithm and splitter), and will create a classifier with the best
    method and iteration (k or depth) that will be returned.

    """
    try:
        makedirs(output_dir)
    except:
        pass
    KNN_algos = ['ball_tree', 'kd_tree', 'brute', 'auto']
    dtree_splitter = ['best', 'random']
    svm_kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    KNN_range = range(1, 50, 2)
    dtree_range = range(1, 100)
    train = [X_train, y_train]
    val = [X_val, y_val]
    test = [X_test, y_test]
    best_knn_val = plot_classifier(KNN, KNN_algos, train, val, KNN_range, 'Validation')
    best_knn_test = plot_classifier(KNN, KNN_algos, train, test, KNN_range, 'Test')
    best_dtree_val = plot_classifier(DTrees, dtree_splitter, train, val, dtree_range, 'Validation')
    best_dtree_test = plot_classifier(DTrees, dtree_splitter, train, test, dtree_range, 'Test')
    best_svm_val = plot_classifier(SVM, svm_kernel, train, test, dtree_range, 'Test')
    best_svm_test = plot_classifier(SVM, svm_kernel, train, test, dtree_range, 'Test')
    optimal_classifier = None
    best_accuracy = 0
    for result in [best_knn_val, best_dtree_val, best_knn_test, best_dtree_test]:
        for method, res in result.items():
            if res['max_accuracy'] > best_accuracy:
                best_accuracy = res['max_accuracy']
                if res['classifier'] == 'KNN':
                    optimal_classifier = KNN(k=res['optimal_x'], method=method)
                elif res['classifier'] == 'Decision Tree':
                    optimal_classifier = DTrees(depth=res['optimal_x'], method=method)
    optimal_classifier.fit(X_train, y_train)
    return optimal_classifier

