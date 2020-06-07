import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
import warnings


class MeanEncoding(BaseEstimator, TransformerMixin):
    """Class to calculate mean encoding also know as Target encoding in our his flavors
    simple mean-encoding mean-encoding-regularization and mean-encoding-regularization-smooth

    Parameters
    ----------
    strategy: The strategy to follow in mean encoding
    alpha: optional value if the strategy is 'mean_reg_smooth'
    seed: Optional value to replicate experiments
    """

    def __init__(self, target, strategy='mean_reg', alpha=0, seed=1):
        self.features_ = None
        self.strategy = strategy
        self.target = target
        self.seed = seed
        self.means_ = dict()
        self.global_mean_ = None
        self.alpha = alpha
        self.__check_data_type()

    def fit(self, X, y=None):
        # check data type
        self.__check_data_type(X=X)
        # omit target column in transformation
        self.features_ = X.drop(columns=self.target).columns.tolist()
        # fit mean-encoding
        if self.strategy == 'mean':
            for col in self.features_:
                self.means_[col] = X.groupby(col)[self.target].mean().to_dict()

        # fit mean-encoding-regularization
        elif self.strategy == 'mean_reg':
            if self.alpha > 0:
                raise ValueError("mean-encoding with regularization and without smoothing require alpha=0."
                                 " If you want use smooth consider strategy='mean_reg_smooth'")
            self.__mean_encoding_reg_fit(X)

        # fit mean-encoding-regularization-smooth
        elif self.strategy == 'mean_reg_smooth':
            if self.alpha == 0:
                self.alpha = 5
                warnings.warn("for mean-encoding with regularization and smooth alpha "
                              "must be greater than 0. Automatically set to 5")
            self.__mean_encoding_reg_fit(X)

        return self

    #
    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError('X must be pd.DataFrame Object')
        if self.strategy == 'mean':
            df_aux = self.__mean_encoding(X)
        elif self.strategy == 'mean_reg':
            df_aux = self.__mean_encoding_reg(X)
        elif self.strategy == 'mean_reg_smooth':
            df_aux = self.__mean_encoding_reg(X)
        return df_aux

    def get_feature_names(self):
        names = [col + 'mean_target' for col in self.features_]
        return names

    def __mean_encoding_reg_fit(self, X):
        # get global mean
        self.global_mean_ = X[self.target].mean()
        # create fold to iterate
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        count = 0
        # iterate over each kfold defined
        for train_index, test_index in skf.split(X, X[self.target]):
            print(train_index, test_index)
            train, test = X.iloc[train_index], X.iloc[test_index]
            count += 1
            # iterate over each columns passed
            for col in self.features_:
                num_samples = train.groupby(col)[self.target].count()
                pond = (train.groupby(col)[self.target].mean() * num_samples + self.alpha * self.global_mean_) / (
                            self.alpha + num_samples)
                means = test[col].map(pond)
                aux = {round(means.values[0], 2): means.index.values.tolist()}
                self.means_[col + '_' + str(count)] = aux

        return None

    def __mean_encoding(self, X):
        df_aux = pd.DataFrame()
        for col, value in self.means_.items():
            df_aux[col + '_mean_target'] = X[col].map(pd.Series(value))

        return df_aux

    def __mean_encoding_reg(self, X):
        df_aux = pd.DataFrame(index=X.index)
        for col in self.features_:
            df_aux[col + '_mean_target'] = None

        for col, val in self.means_.items():
            for k, v in val.items():
                aux = pd.Series(data=[k] * len(v), index=v)
                col_name = '_'.join(col.split('_')[:-1])
                # df_aux.iloc[aux.index, df_aux.columns.get_loc(col_name + '_meantarget')] = aux
                df_aux.loc[aux.index, col_name + '_mean_target'] = aux
        # fill the na values with the global mean
        df_aux.fillna(self.global_mean_, inplace=True)
        return df_aux

    def __check_data_type(self, X=None):
        if X is None:
            available_options = ['mean', 'mean_reg', 'mean_reg_smooth']
            if self.strategy not in available_options:
                raise ValueError("The available options for strategy are: 'mean', 'mean_reg', 'mean_reg_smooth'")
            if not isinstance(self.target, str):
                raise TypeError('target must be str Object')
        else:
            if not isinstance(X, pd.DataFrame):
                raise TypeError('X must be pd.DataFrame Object')
            if self.target not in X.columns:
                raise KeyError('The target column name is not in X DataFrame')
