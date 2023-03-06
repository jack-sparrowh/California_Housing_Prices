import os
import requests
import csv
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from scipy.stats import skew, iqr

#col_idx_dict = {housing.columns[i]:i for i in range(housing.shape[1])}

col_idx_dict = {housing.columns[i]:i for i in range(housing.shape[1])}

class AddFeatures(BaseEstimator, TransformerMixin):
    '''
    Adds new features to the dataset.
    
    Args:
        X (np.ndarray): (m, n) Data points
        
    Methods:
        fit(X)       - does nothing
        transform(X) - adds new features
    '''
    def fit(self, X):
        # return self
        return self
    
    def transform(self, X):
        '''
        Adds new features to the data.
        
        Args:
            X (np.ndarray): (m, n) Data points
            
        Returns:
            X (np.ndarray): (m, n + 6) Dataset with added features
        '''
        
        # create arrays of new features
        rooms_per_household      = X[:, col_idx_dict['total_rooms']]    / X[:, col_idx_dict['households']]
        income_per_household     = X[:, col_idx_dict['median_income']]  / X[:, col_idx_dict['households']] * X[:, col_idx_dict['population']]
        income_per_population    = X[:, col_idx_dict['median_income']]  / X[:, col_idx_dict['population']] * X[:, col_idx_dict['households']]
        bedrooms_per_rooms       = X[:, col_idx_dict['total_bedrooms']] / X[:, col_idx_dict['total_rooms']]
        population_per_household = X[:, col_idx_dict['population']]     / X[:, col_idx_dict['households']]
        rooms_per_age            = X[:, col_idx_dict['total_rooms']]    / X[:, col_idx_dict['housing_median_age']]
        
        # return concatenated dataset
        return np.c_[X, rooms_per_household, income_per_household, income_per_population, bedrooms_per_rooms, population_per_household, rooms_per_age]

    
    
class DataDropper(BaseEstimator, TransformerMixin):
    '''
    This class drops points considered outliers given the feature and method of dropping them. 
    Available methods are:
    "fixed"     - drops data that are <= choosen fixed threshold,
    "flexible"  - drops data that are <= choosen quantile,
    "optimized" - drops data with the help of a simple loss function:
                  $L(i, p) = S(i) + i^{p}$
                  where, i is the ith dropped data point, S(i) is the skewness after dropping
                  the ith data point and p is the penalty. The main assumption is that with
                  highly skewed features dropping points will monotonically lower the skewness.
                  Adding a penalty of $i^{p}$ creates a global minimum, that we seek to find,
    "skewness"  - drops data untill some fixed value of skewness is satisfied.
    
    Args:
        method (str)                     - method to use when dropping the data
        feature (str)                    - feature variable to drop data from
        val (float | int)                - value for "fixed", "flexible", and "skewness" methods.
                                           For "fixed" the value must be some point in the interval of given
                                           feature. For "flexible" method the value must be a quantile [0, 1] 
                                           which will specify the threshold for dropping points. For "skewness"
                                           the value must be a minimal skewness we want to obtain, it must be
                                           lower than the skewness of the feature itself.
        penalty (str, default "sq_root") - specifies what penalty is added to the loss after dropping ith point.
                                           If "linear" we take i to the 1th power. If "sq_root" i is raised to
                                           the power of 0.5.
    '''
    
    def __init__(self, method, feature, penalty="sq_root", val=None):
        '''
        Args:
            method (str)                     - method to use when dropping the data
            feature (str)                    - feature variable to drop data from
            val (float | int)                - value for "fixed", "flexible", and "skewness" methods.
                                               For "fixed" the value must be some point in the interval of given
                                               feature. For "flexible" method the value must be a quantile [0, 1] 
                                               which will specify the threshold for dropping points. For "skewness"
                                               the value must be a minimal skewness we want to obtain, it must be
                                               lower than the skewness of the feature itself.
            penalty (str, default "sq_root") - specifies what penalty is added to the loss after dropping ith point.
                                               If "linear" we take i to the 1th power. If "sq_root" i is raised to
                                               the power of 0.5.
        '''
        assert(method in ['fixed', 'flexible', 'optimized', 'skewness']), 'methods available: "fixed", "flexible", "optimized", "skewness".'
        assert(feature in col_idx_dict.keys()), f'feature must be one of : {list(col_idx_dict.keys())}'
        self.method = method
        self.val = val
        self.feature = feature
        self.col_index = col_idx_dict[self.feature]
        self.penalty = penalty
        
    def check_val(self, val):
        '''
        Checks if the variable val is float or int. If not it raises AssertionError.
        
        Args:
            val (float | int) - value for "fixed", "flexible", and "skewness" methods.
            
        Raises:
            AtributeError
        '''
        assert((type(val) == float) or (type(val) == int)), f'specify the value (int, float) for right parameter of {self.method} method.'
        
    def flag_outliers(self, X):
        '''
        Returns data points that are further (or less) than 3th quartile + IQR (or - IQR).
        
        Args:
            X (np.ndarray) - (m, n) data points
            
        Returns:
            X (np.ndarray) - (no. of outliers, n) flagged outliers
        '''
        _iqr = iqr(X[:, self.col_index])
        _lower_bound = np.quantile(X[:, self.col_index], 0.25) - 1.5 * _iqr
        _upper_bound = np.quantile(X[:, self.col_index], 0.75) + 1.5 * _iqr
        return X[(X[:, self.col_index] <= _lower_bound) | (X[:, self.col_index] >= _upper_bound), self.col_index]
    
    def fit(self, X):
        '''
        Stores boolean values for indices to keep. Also, stores the number of deleted data points.
        
        Args:
            X (np.ndarray) - (m, n) data points
            
        Returns:
            self
        '''
        if self.method == 'fixed':
            # make sure that fix value is specified
            self.check_val(self.val)
            # make sure that the point is in the interval of data
            _max_val = X[:, self.col_index].max()
            _min_val = X[:, self.col_index].min()
            assert((self.val <= _max_val) and (self.val >= _min_val)), 'the value must be in interval [min, max].'
            self.idx_to_keep = X[:, self.col_index] <= self.val
            
        if self.method == 'flexible':
            # make sure that flex value is specified
            self.check_val(self.val)
            # make sure that the quantile is in [0, 1]
            assert((self.val <= 1) and (self.val >= 0)), 'the value must be in interval [0, 1].'
            self.idx_to_keep = X[:, self.col_index] <= np.quantile(X[:, self.col_index], self.val)
            
        if self.method == 'optimized':
            
            assert(self.penalty in ['linear', 'sq_root']), 'penalties available: "linear", "sq_root".'
            if self.penalty == 'linear':
                self.penalty = 1
            else:
                self.penalty = 0.5
            
            loss = np.zeros(1)
            skewness_of_data = skew(X[:, self.col_index])
            sorted_data = np.sort(self.flag_outliers(X))[::-1]
            for point, data in enumerate(sorted_data):
                if point < skewness_of_data:
                    loss = np.c_[loss, skew(X[X[:, self.col_index] <= data, self.col_index]) + point ** self.penalty]
                else:
                    break
            loss = loss[loss > 0].flatten()
            self.optimization_history = loss
            self.idx_to_keep = X[:, self.col_index] <= sorted_data[loss.argmin()]
            
        if self.method == 'skewness':
            # make sure that skew value is specified
            self.check_val(self.val)
            # make sure that the skew value is lower than max skewness observed
            max_skew = skew(X[:, self.col_index])
            assert(self.val <= max_skew), f'Value for skewness is higher than the maximum skewness observed in the data: {self.val} > {round(max_skew, 1)}.'
            sorted_data = np.sort(X[:, self.col_index])[::-1]
            for point, data in enumerate(sorted_data):
                skew_after_drop = skew(X[X[:, self.col_index] <= data, self.col_index])
                if skew_after_drop <= self.val:
                    self.idx_to_keep = X[:, self.col_index] <= data
                    break
        
        self.number_deleted = X.shape[0] - self.idx_to_keep.sum()
        return self
        
    def transform(self, X):
        '''
        Mask the data with indicies to keep.
        
        Args:
            X (np.ndarray) - (m, n) data points
            
        Returns:
            X (np.ndarray) - (m - outliers, n) masked dataset
        '''        
        # return the dataset without outliers
        return X[self.idx_to_keep, :]


    
def multi_features_outliers_dropper(X, features, method, val=None, penalty=None):
    '''
    Utilizes DataDropper class to drop oultiers with fixed mehtod for multiple features.
    
    Args:
        X (np.ndarray)                   - (m, n) data points
        method (str)                     - method to use when dropping the data
        features (list)                  - list of features to drop data from
        val (float | int)                - value for "fixed", "flexible", and "skewness" methods.
        penalty (str, default "sq_root") - specifies what penalty is added to the loss after dropping ith point.
                                           If "linear" we take i to the 1th power. If "sq_root" i is raised to
                                           the power of 0.5.
    
    Returns:
        X (np.ndarray) - (m - outliers, n) - dataset without outliers
        
    Raises:
        AttributeError if features list contains values that are not in the list of original features.
    '''
    # raise an error if features list contains invalid feature names
    assert(len(set(features).intersection(col_idx_dict.keys())) == len(features)), f'features must be a list of features from the data: {list(col_idx_dict.keys())}'
    
    try:
        X = X.values
    except AttributeError:
        pass
    
    for feature in features:
        _outlier_remover = DataDropper(method=method, feature=feature, val=val, penalty=penalty)
        _outlier_remover.fit(X)
        X = _outlier_remover.transform(X)
        
    return X



class small_PCA(BaseEstimator, TransformerMixin):
    '''
    Finds principal components of centered (not standardized) data. It does not use SVD approach.
    The method finds weights of orthogonal projections of data points onto a subspace that is 
    spanned by the basis constructed of orthonormal eigenvectors with highest eigenvalues. 
    Eigenvectors and eigenvalues are obtained by eigendecomposition of covariance matrix. The size
    of the basis is specified by the value of n_components.
    
    Args:
        n_components (int) - rank of reduced data
        X (np.ndarray)     - (m, n) dataset
        
    Methods:
        fit(X)       - estimates the orthonormal eigenbasis for the subspace
        transform(X) - produces the weights of orthogonal projections
    '''
    
    def __init__(self, n_components=None):
        '''
        Params:
            n_components (int) - rank of reduced data
        '''
        
        # check if the number of components has the right type and is greater than 0
        assert(isinstance(n_components, int) and (n_components > 0)), f'n_components must be an integer greater than 0.'
        
        self.n_components = n_components
        
    def fit(self, X):
        '''
        The method finds eigendecomposition of centered data of form $Q \Lambda Q^{T}$. Then,
        n - n_components lowest eigenvalues and eigenvectors are dropped. The columns of resulting
        basis span the subspace in question. Also, the values of eigenvalues are stored as well
        as their ratio.
        
        Params:
            X (np.ndarray) - (m, n) dataset
        '''
        
        # check if number of components is lower than the number of columns of data
        assert(self.n_components <= X.shape[1]), f'n_components are greater than number of features. {self.n_components} > {X.shape[1]}'
        
        # calculate and eigendecompose the covariance matrix
        cov_matrix = np.cov((X - X.mean(axis=0)).T)
        s, Q = np.linalg.eig(cov_matrix)
        
        # get n_components with highest values of eigenvalues
        Q_r = Q[:, s.argsort()[-self.n_components:]]
        self.reduced_basis = Q_r[:, ::-1] # ---> this assures the resulting data is similar with sklearn's PCA
        self.explained_variance_ = s
        self.explained_variance_ratio_ = s / s.sum()
        return self
    
    def transform(self, X):
        '''
        Reduce the dataset by finding the weights of orthogonal projections on the subspace Col(Q_r).
        Since Q_r has orthonormal columns, the weights can be obtained from $(Q_{r}^{T}X^{T})^{T} = XQ_{r}$.
        
        Args:
            X (np.ndarray)   - (m, n) dataset
            
        Returns:
            X_r (np.ndarray) - (m, n_components) principal components
        '''
        # return mean 0 results (it is not necessary at all, but the results will be the same as with sklearn's PCA)
        return (X - X.mean(axis=0)) @ self.reduced_basis

    
    
class MulticollinearityHandler(BaseEstimator, TransformerMixin):
    '''
    This class tries to deal with multicollinearity problem implementing three different methods.
    1. Simply drop all but one feature. This method leaves the one feature that has the highest
       correlation with the target variable.
    2. Use small_PCA to reduce the dimention of the correlated features within a specified group.
    3. Uses method described by Min Tsao et al. (sorce at the end) It tries to find a vector of
       weights w that finds the overall effect of the group.
       
    The methods described above can be choosen by specifying the method argument, respectively:
    "drop"          - drops all but one feature within group
    "pca"           - uses small_PCA class
    "gr_treatement" - uses the method of M. Tsao
       
    Since two highly correlated groups were recognized, there is only option to choose one of the
    groups:
    "first"  - ['total_rooms', 'population', 'total_bedrooms', 'households']
    "second" - ['income_per_household', 'median_income', 'income_per_population']
    
    Args:
        method (str)       - method to choose when reducing the data
        group (str)        - one of ["first", "second"]
        n_components (int) - rank of reduced data when using pca method
        X (np.ndarray)     - (m, n) dataset
        
    Methods:
        get_indicies_for_group(group)
            finds column numbers of features given the specified group. 
            
        get_the_least_corr_features_group(X, gropu)
            finds the features that has the least correlation with the target variable
            
        fit(X)
            given the selected method it proceedes with finding the indexes that should be dropped
            
        transform(X)
            transforms the data using the indexes found in fit() method
            
    Sorce:
        Min Tsao "Group least squares regression for linear models with strongly correlated predictor
        variables". 
        DOI:      https://doi.org/10.1007/s10463-022-00841-7
        arXiv:    https://doi.org/10.48550/arXiv.1804.02499
        Accessed: 2/20/23
        
    '''    
    def __init__(self, method, group, n_components=None):
        '''
        Params:
            method (str)       - method to choose when reducing the data
            group (str)        - one of ["first", "second"]
            n_components (int) - rank of reduced data when using 
        '''
        assert(method in ['drop', 'pca', 'gr_teatement']), f'Method {method} is invalid. Available methods: ["drop", "pca", "gr_treatement"].'
        assert(group in ['first', 'second']), f'Group {group} is invalid. Please, choose one of available: ["first", "second"]. For more information consult documentation.'
        self.method = method
        self.group = group
        self.groups = dict(first = ['total_rooms', 'population', 'total_bedrooms', 'households'],
                           second = ['income_per_household', 'median_income', 'income_per_population'])
        self.n_components = n_components
    
    def get_indicies_for_group(self, group):
        '''
        Finds column numbers of features given the specified group. The method uses a globaly specified
        dictionary that specifies the column number given the feature name. The dictionary must be 
        specified beforehand. The main pipeline is responsible for creating this dictionary.
        
        Args:
            group (str) - one of ["first", "second"]
        
        Returns:
            _indicies (list) - list of indicies that specifies the pair feature - column number
        '''
        
        # create a matrix [feature_name index_number]
        tmp_holder = np.array([*col_idx_dict.items()])
        
        #get the indexes for given group
        _indicies = [
            tmp_holder[tmp_holder[:, 0] == col, 1].astype(int)[0] 
            for col in self.groups[group]
        ]
        
        return _indicies
    
    def get_the_least_corr_features_idx(self, X, group):
        '''
        Finds the features that has the least correlation with the target variable. The method uses a 
        globaly specified dictionary that specifies the column number given the feature name. The 
        dictionary must be specified beforehand. The main pipeline is responsible for creating this dictionary.
        
        Args:
            X (np.ndarray) - (m, n) dataset
            group (str)    - one of ["first", "second"]
            
        Returns:
            _low_corr_features_idx (np.ndarray) - array of column numbers for the least correlated with
                                                  the target variable features
        '''
        # get the group indexes
        group_indicies = self.get_indicies_for_group(group)
        # mask
        X_indexed = X[:, group_indicies + [col_idx_dict['median_house_value']]]
        # calculate correlation with the target variable
        _correlation = np.corrcoef(X_indexed.T)[-1, :-1]
        # get the indexes of the least correlated features
        _low_corr_features_idx = np.array(group_indicies)[_correlation.argsort()[:-1]]
        return _low_corr_features_idx
    
    def fit(self, X):
        
        if self.method == 'drop':
            self.idx_to_drop = self.get_the_least_corr_features_idx(X, self.group)
        
        if self.method == 'pca':
            assert(self.n_components is not None), f'Specify the number of components.'
            # get the group indexes
            self.idx_to_reduce = self.get_indicies_for_group(self.group)
            # pca should be done in transform
            
        if self.method == 'gr_treatement':
            # to be added, i have to first figure out the context of the article.
            pass
        
        return self
    
    def transform(self, X):
        
        if self.method == 'drop':
            return np.delete(X, self.idx_to_drop, axis=1)
        
        if self.method == 'pca':
            s_pca = small_PCA(n_components=self.n_components)
            X_reduced = s_pca.fit_transform(X[:, self.idx_to_reduce])
            X_concat = np.c_[np.delete(X, self.idx_to_reduce, axis=1),
                             X_reduced]
            return X_concat