import pandas as pd

from sklearn.model_selection import train_test_split

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from os import makedirs


class BCData:
    NORMAL = 0
    TUMOR = 1
    normal_data, tumor_data, matrix_normal_data, matrix_tumor_data, all_data = 5 * [None]
    X, y, X_train_, y_train_, X_val_, y_val_, X_test_, y_test_, gene_list = 9 * [None]
    X_train, y_train, X_val, y_val, X_test, y_test = 6 * [None]
    feature_names, selected_features = 2 * [[]]
    sfs = []
    removed_features = []

    def __init__(self, normal_path, tumor_path):
        self.normal_data_path = normal_path
        self.tumor_data_path = tumor_path
        self.normal_data = pd.read_csv(self.normal_data_path, delim_whitespace=True)
        self.tumor_data = pd.read_csv(self.tumor_data_path, delim_whitespace=True)

    def set_data(self, method='median', use_validation=True):
        self.gene_list = self.normal_data['Hybridization']
        self.feature_names = list(self.gene_list)

        self.matrix_normal_data = self.normal_data.drop(['Hybridization'], axis=1)
        self.matrix_normal_data = self.matrix_normal_data.transpose()
        self.matrix_normal_data.insert(self.matrix_normal_data.shape[1], 'Target', self.NORMAL)

        self.matrix_tumor_data = self.tumor_data.drop(['Hybridization'], axis=1)
        self.matrix_tumor_data = self.matrix_tumor_data.transpose()
        self.matrix_tumor_data.insert(self.matrix_tumor_data.shape[1], 'Target', self.TUMOR)

        self.all_data = self.matrix_normal_data.append(self.matrix_tumor_data)

        self.__data_preparation(method, use_validation)

    def __data_preparation(self, method='median', use_validation=True):
        if str(method).lower() is 'median':
            self.normal_data = self.normal_data.fillna(self.normal_data.median())
            self.tumor_data = self.tumor_data.fillna(self.tumor_data.median())
        elif str(method).lower() is 'mean':
            self.normal_data = self.normal_data.fillna(self.normal_data.median())
            self.tumor_data = self.tumor_data.fillna(self.tumor_data.median())
        elif str(method).lower() is 'dropna':
            self.normal_data = self.normal_data.fillna(self.normal_data.dropna())
            self.tumor_data = self.tumor_data.fillna(self.tumor_data.dropna())

        if str(method).lower() in ['median', 'mean', 'dropna']:
            # If remains NA data
            self.normal_data = self.normal_data.dropna()
            self.tumor_data = self.tumor_data.dropna()

        self.X = self.all_data.dropna().drop(['Target'], axis=1)
        self.y = self.all_data.dropna()['Target']

        self.__split_data(use_validation)

    def __split_data(self, use_validation=True):
        # Split the data
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(self.X, self.y, test_size=0.2,
                                                                                    random_state=1)
        self.y_train = self.y_train_.copy(True)
        self.y_test = self.y_test_.copy(True)
        if use_validation:
            self.X_train_, self.X_val_, self.y_train_, self.y_val_ = train_test_split(self.X_train_, self.y_train_,
                                                                                      test_size=0.2, random_state=1)
            self.y_val = self.y_val_.copy(True)

    def feature_selection(self, estimator, num_of_features=20, forward=True, floating=False,
                          cross_validation=5, n_threads=-1, verbose=0, include_features_names=True,
                          advance=False):
        # Features in advance mode:
        # ['KLHL29', 'FLVCR1', 'MMP11', 'SYP', 'ZNF608', 'SLC25A42', 'TIMM8A', 'AJAP1', 'FGFBP1', 'MLCK', 'PLXND1',
        #  'INTS5', 'BSG', 'TMEM89', 'PARP8', 'TEAD4', 'ARMC2', 'TNRC6B', 'DTX4', 'ZNF498']

        output_dir = 'SFS results/'
        try:
            makedirs(output_dir)
        except:
            pass
        X = self.X_train_.copy(True)
        y = self.y_train_.copy(True)
        if forward and not floating:
            technique = 'Sequential Forward Selection '
        elif not forward and not floating:
            technique = 'Sequential Backward Selection '
        elif forward and floating:
            technique = 'Sequential Forward Floating Selection '
        else:  # not forward and floating
            technique = 'Sequential Backward Floating Selection '
        technique = technique + __get_estimator__name__(estimator)

        removed_count = num_of_features
        iteration = 0
        feature_names = self.feature_names
        while True:
            sfs = SFS(estimator=estimator,
                      k_features=num_of_features,
                      forward=forward,
                      floating=floating,
                      verbose=verbose,
                      scoring='accuracy',
                      cv=cross_validation,
                      n_jobs=n_threads)
            iteration += 1
            if include_features_names:
                sfs = sfs.fit(X.as_matrix(), y.as_matrix(), feature_names)

                # Sets the selected features
                self.selected_features = sfs.k_feature_names_
            else:
                sfs = sfs.fit(X.as_matrix(), y.as_matrix())

            # Information about the SFS results
            print('\n' + technique + ' (k=' + str(num_of_features) + ') :')
            print(sfs.k_feature_idx_)
            print('CV Score:')
            print(sfs.k_score_)

            fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
            plt.ylim([0.8, 1])
            plt.title(technique + ' (w. StdDev)')
            plt.grid()
            fig1.savefig('{0}{1} {2}.png'.format(output_dir, technique, str(iteration)))
            plt.show()

            self.sfs.append(sfs)
            if verbose > 0:
                print('Saves SFS iterations at ' + technique + ' results.csv')
                pd.DataFrame.from_dict(sfs.get_metric_dict()).T.to_csv(
                    '{0}{1} results {2}.csv'.format(output_dir, technique, str(iteration)))
            if not advance:
                self.X_train = self.X_train_[list(sfs.k_feature_idx_)]
                if type(self.X_val_) == pd.core.frame.DataFrame:
                    self.X_val = self.X_val_[list(sfs.k_feature_idx_)]
                self.X_test = self.X_test_[list(sfs.k_feature_idx_)]
                self.feature_names = list(self.gene_list)

                return sfs
            else:
                if removed_count == 0:
                    self.X_train = self.X_train_[list(sfs.k_feature_idx_)]
                    if type(self.X_val_) == pd.core.frame.DataFrame:
                        self.X_val = self.X_val_[list(sfs.k_feature_idx_)]
                    self.X_test = self.X_test_[list(sfs.k_feature_idx_)]
                    self.feature_names = list(self.gene_list)

                    return sfs

                features_to_remove = []
                features_idx_to_remove = []
                for idx, res in enumerate(sfs.get_metric_dict().values()):
                    if res['avg_score'] < 1.0 or (idx == 0 and res['avg_score'] == 1.0):
                        self.removed_features.append(res['feature_names'][0])
                        features_to_remove.append(res['feature_names'][0])
                        features_idx_to_remove.append(res['feature_idx'][0])
                        feature_names.remove(res['feature_names'][0])
                for idx in features_idx_to_remove:
                    X = X.drop(idx, axis=1)
                    removed_count -= 1
                if len(features_to_remove) == 2:
                    self.plot(features_to_remove)
                else:
                    print("\n\n\t{}\n\n".format(features_to_remove))

    def plot(self, features):
        tmp = self.X_train_.copy(True)
        tmp.columns = self.feature_names
        tmp['Target'] = self.y_train
        normal_tissue = tmp.loc[tmp.Target == self.NORMAL]
        tumor_tissue = tmp.loc[tmp.Target == self.TUMOR]

        df = pd.DataFrame(normal_tissue, columns=features)
        ax = df.plot.scatter(x=features[0], y=features[1], color='DarkBlue', label='Normal Tissue')
        df2 = pd.DataFrame(tumor_tissue, columns=features)
        df2.plot.scatter(x=features[0], y=features[1], color='DarkGreen', label='Tumor tissue', ax=ax)
        plt.show()


def __get_estimator__name__(estimator):
    return estimator.__str__.__str__().split()[estimator.__str__.__str__().split().index('of') + 1]
