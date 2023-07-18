import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy.stats import iqr, skew


class CappedTargetDropper(BaseEstimator, TransformerMixin):
    '''
    This simple class deals with capped values within variables. If there are
    capped values within the data it might be better to drop them, since there 
    is a risk of biasing the model. If the % of capped values is low (e.g. <5%),
    there should be no reason for not to drop the data, but every case must be
    considered on its own.
    
    Even though the class is called "capped target dropper", it is able to deal 
    with any variable with the help of small_Pipeline or just by passing given 
    feature.
    
    Args:
        drop : bool (default True)
            Specify if the values should be dropped.
        capped_val : float | int
            Specify the capped value. If int is passed it will be converted to
            float.
            
    Methods:
        fit(X) :
            Finds all the indices for which the values of the data should be 
            converted to nans.
        transform(X) : 
            Given the drop parameter is true, it will convert values of selected
            indicies to nans.
            
    Returns:
        X_tmp : pd.Series | np.ndarray (1, n)
            Series with converted capped values.
            
    Raises:
        AssertionError (__init__) :
            If drop is true but the capped_val is not specified, then the class
            will raise the assertion error.
        AssertionError (fit) :
            If capped_val is not present in the data given that the drop is true
            it will raise the asserion error and ask to specify the capped_val.
        AssertionError (fit, transform) : 
            If the dimention of the data is >1 then the assertion will be raised,
            since the class handles only 1-dim data.
    '''
    
    
    def __init__(self, drop=True, capped_val=None):
        '''
        Args:
            drop : bool (default True)
                Specify if the values should be dropped.
            capped_val : float | int
                Specify the capped value. If int is passed it will be converted to
                float.
                
        Raises:
            AssertionError :
                If drop is true but the capped_val is not specified, then the class
                will raise the assertion error.
        '''
        # given that drop is true, check if capped_value is specified
        if drop:
            assert(
                (capped_val is not None) and \
                (isinstance(capped_val, float) or isinstance(capped_val, int))
            ), f'If "drop" parameter is true, you must specify the value of "capped_val" parameter.'
        
        # convert capped_val to float, and store the values
        self.capped_val = float(capped_val)
        self.drop=drop
        
    def fit(self, X):
        '''
        Finds all the indices for which the values of the data should be 
        converted to nans.
        
        Args:
            X : pd.Series | np.ndarray (1, n)
                Vector of values for given feature.
                
        Returns:
            self :
            
        Raises: 
            AssertionError :
                If capped_val is not present in the data given that the drop is true
                it will raise the asserion error and ask to specify the capped_val.
            AssertionError : 
                If the dimention of the data is >1 then the assertion will be raised,
                since the class handles only 1-dim data.
        '''
        # given that drop is true, check if the value of capped_val is present in the data
        if self.drop:
            assert((X == self.capped_val).any()), f'{self.capped_val} not in the data.'
        
        # check if the data passed it 1-dimentional
        assert(X.ndim == 1), f'Data passed must be 1-dimentional.'
        
        # get the indicies that contains the capped_val
        idx_to_nan = X == self.capped_val
        self.idx_to_nan = idx_to_nan
        return self
    
    def transform(self, X):
        '''
        Given the drop parameter is true, it will convert values of selected
        indicies to nans.
        
        Args:
            X : pd.Series | np.ndarray (1, n)
                Vector of values for given feature.
        
        Returns:
            X_tmp : pd.Series | np.ndarray (1, n)
            
        Raises:
            AssertionError : 
                If the dimention of the data is >1 then the assertion will be raised,
                since the class handles only 1-dim data.
        '''
        # check if the data is 1-dimentional
        assert(X.ndim == 1), f'Data passed must be 1-dimentional.'
        
        # if drop is true convert the capped values into nans
        if self.drop:
            X_tmp = X.copy()
            X_tmp[self.idx_to_nan] = np.nan
            return X_tmp

class FeaturesAdder(BaseEstimator, TransformerMixin):
    '''
    This class adds 6 new features to the data set. The features added are:
        * 'rooms_per_household'
        * 'income_per_household'
        * 'income_per_population'
        * 'bedrooms_per_rooms'
        * 'population_per_household'
        * 'rooms_per_age'
    
    Args:
        X : pd.DataFrame
            (m, n) Data points
        
    Methods:
        fit(X) :
            does nothing
        transform(X) :
            adds new features
    '''
    def fit(self, X):
        # do nothing
        return self
    
    def transform(self, X):
        '''
        Adds new features to the data. The names of features are pointed in the 
        main doc.
        
        Args:
            X : pd.DataFrame
                (m, n) Data points
            
        Returns:
            X : pd.DataFrame
                (m, n + 6) Dataset with added features
        '''
        
        # copy the array
        X_tmp = X.copy()
        
        # create arrays of new features
        X_tmp['rooms_per_household']      = X_tmp['total_rooms']    / X_tmp['households']
        X_tmp['income_per_household']     = X_tmp['median_income']  / X_tmp['households'] * X_tmp['population']
        X_tmp['income_per_population']    = X_tmp['median_income']  / X_tmp['population'] * X_tmp['households']
        X_tmp['bedrooms_per_rooms']       = X_tmp['total_bedrooms'] / X_tmp['total_rooms']
        X_tmp['population_per_household'] = X_tmp['population']     / X_tmp['households']
        X_tmp['rooms_per_age']            = X_tmp['total_rooms']    / X_tmp['housing_median_age']
        
        # return dataset with added features
        return X_tmp
    
    
class DataDropper(BaseEstimator, TransformerMixin):
    '''
    This class drops points considered outliers given the feature and method of 
    dropping them. Available methods are:
    "fixed" :
        drops data that are <= choosen fixed threshold,
    "flexible" :
        drops data that are <= choosen quantile,
    "optimized" :
        drops data with the help of a simple loss function:$L(i, p) = S(i) + i^{p}$
        where, i is the ith dropped data point, S(i) is the skewness after 
        dropping the ith data point and p is the penalty. The main assumption is
        that with highly skewed features dropping points will monotonically lower
        the skewness. Adding a penalty of $i^{p}$ creates a global minimum, that
        we seek to find,
    "skewness":
        drops data untill some fixed value of skewness is satisfied.
    
    Args:
        method : str
            method to use when dropping the data
        val : float | int
            value for "fixed", "flexible", and "skewness" methods. For "fixed" 
            the value must be some point in the interval of given feature. For 
            "flexible" method the value must be a quantile [0, 1]  which will 
            specify the threshold for dropping points. For "skewness" the value 
            must be a minimal skewness we want to obtain, it must be lower than 
            the skewness of the feature itself.
        penalty : float | int, (default 0.5)
            specifies what penalty is added to the loss after dropping ith point.
            If int is passed it will be converted to float. The lower the power
            of penalty the more points will be dropped.
    '''
    
    def __init__(self, method, penalty=0.5, val=None):
        '''
        Args:
            method : str
                method to use when dropping the data
            val : float | int
                value for "fixed", "flexible", and "skewness" methods. For "fixed" 
                the value must be some point in the interval of given feature. For 
                "flexible" method the value must be a quantile [0, 1]  which will 
                specify the threshold for dropping points. For "skewness" the value 
                must be a minimal skewness we want to obtain, it must be lower than 
                the skewness of the feature itself.
            penalty : float | int, (default 0.5)
                specifies what penalty is added to the loss after dropping ith point.
                If int is passed it will be converted to float. The lower the power
             of penalty the more points will be dropped.
        '''
        assert(method in ['fixed', 'flexible', 'optimized', 'skewness']), 'methods available: "fixed", "flexible", "optimized", "skewness".'
        assert(float(penalty)), f'penalty must be a float or an integer.'
        self.method = method
        self.val = val
        self.penalty = float(penalty)
        self.idx_to_nan = None
        
    def check_val(self, val):
        '''
        Checks if the variable val is float or int. If not it raises AssertionError.
        
        Args:
            val : float | int
                value for "fixed", "flexible", and "skewness" methods.
            
        Raises:
            AtributeError
        '''
        assert((type(val) == float) or (type(val) == int)), f'specify the value (int, float) for right parameter of {self.method} method.'
        
    def flag_outliers(self, X):
        '''
        Returns data points that are further (or less) than 3th quartile + IQR (or - IQR).
        
        Args:
            X : pd.Series
                (m, 1) data points
            
        Returns:
            X : pd.Series
                (no. of outliers, 1) flagged outliers
        '''
        X_no_nan = X.copy()
        X_no_nan = X_no_nan[~np.isnan(X_no_nan)]
        _iqr = iqr(X_no_nan)
        _lower_bound = np.quantile(X_no_nan, 0.25) - 1.5 * _iqr
        _upper_bound = np.quantile(X_no_nan, 0.75) + 1.5 * _iqr
        return X[(X <= _lower_bound) | (X >= _upper_bound)].dropna()
    
    def fit(self, X):
        '''
        Stores boolean values for indices to keep. Also, stores the number of 
        deleted data points.
        
        Args:
            X : pd.Series
                (m, 1) data points
            
        Returns:
            self
        '''
        # make sure the data has the right type
        assert(isinstance(X, pd.Series)), f'Data should be a pd.Series.'
        
        if self.method == 'fixed':
            # make sure that fix value is specified
            self.check_val(self.val)
            # make sure that the point is in the interval of data
            _max_val, _min_val = X.max(), X.min()
            assert((self.val <= _max_val) and (self.val >= _min_val)), 'the value must be in interval [min, max].'
            self.idx_to_nan = X > self.val
            
        if self.method == 'flexible':
            # make sure that flex value is specified
            self.check_val(self.val)
            # make sure that the quantile is in [0, 1]
            assert((self.val <= 1) and (self.val >= 0)), 'the value must be in interval [0, 1].'
            self.idx_to_nan = X > np.quantile(X, self.val)
            
        if self.method == 'optimized':
            
            loss = np.zeros(1)
            skewness_of_data = np.abs(skew(X, nan_policy='omit'))
            sorted_data = np.sort(self.flag_outliers(X))[::-1]
            for point, data in enumerate(sorted_data):
                if point < skewness_of_data:
                    loss = np.c_[loss, np.abs(skew(X[X <= data])) + point ** self.penalty]
                else:
                    break
            loss = loss[loss > 0].flatten()
            self.optimization_history = loss
            self.idx_to_nan = X > sorted_data[loss.argmin()]
            
        if self.method == 'skewness':
            # make sure that skew value is specified
            self.check_val(self.val)
            # make sure that the skew value is lower than max skewness observed
            max_skew = np.abs(skew(X))
            assert(self.val <= max_skew), f'Value for skewness is higher than the maximum skewness observed in the data: {self.val} > {round(max_skew, 1)}.'
            sorted_data = np.sort(X)[::-1]
            for point, data in enumerate(sorted_data):
                skew_after_drop = np.abs(skew(X[X <= data]))
                if skew_after_drop <= self.val:
                    self.idx_to_nan = X > data
                    break
        
        self.number_deleted = self.idx_to_nan.sum()
        return self
        
    def transform(self, X):
        '''
        Mask the data with indicies to keep.
        
        Args:
            X : pd.Series
                (m, 1) data points
            
        Returns:
            X : pd.Series
                (m - no. of outliers, 1) data points
        '''
        # return the dataset without outliers
        X_tmp = X.copy()
        X_tmp[self.idx_to_nan] = np.nan
        return X_tmp

    
class small_PCA(BaseEstimator, TransformerMixin):
    '''
    Finds principal components of centered (not standardized) data. It does not use SVD approach.
    The method finds weights of orthogonal projections of data points onto a subspace that is 
    spanned by the basis constructed of orthonormal eigenvectors with highest eigenvalues. 
    Eigenvectors and eigenvalues are obtained by eigendecomposition of covariance matrix. The size
    of the basis is specified by the value of n_components.
    
    Args:
        n_components : int 
            rank of reduced data
        X : np.ndarray     
            (m, n) dataset
        
    Methods:
        fit(X) :     
            estimates the orthonormal eigenbasis for the subspace
        transform(X) : 
            produces the weights of orthogonal projections
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
    This class tries to deal with multicollinearity problem implementing three 
    different methods.
    1. Simply drop all but one feature. This method leaves the one feature that 
       has the highest correlation with the target variable.
    2. Use small_PCA to reduce the dimention of the correlated features within a
       specified group.
    3. Uses method described by Min Tsao et al. (sorce at the end) It tries to 
       find a vector of weights w that catches the overall effect of the group.
       
    The methods described above can be choosen by specifying the method argument,
    respectively:
    "drop" : 
        drops all but one feature within a passed group
    "pca" : 
        uses small_PCA class
    "gr_treatement" : 
        uses the method of M. Tsao
    
    Args:
        method : str
            method to choose when reducing the data
        n_components : int 
            rank of reduced data when using 
        X : pd.DataFrame (m, j) 
            dataset with j features that has to be treated
        
    Methods:            
        get_the_highest_corr_feature(X) :
            finds the feature that has the highest absolute correlation value 
            with the target variable
            
        fit(X) :
            given the selected method it proceedes with finding the indices that 
            should be dropped
            
        transform(X) :
            transforms the data using the indices found in fit() method
            
    ### NOTE ###
    If any nan values are present in the dataset, using "pca" method will raise 
    AssertionError.
            
    Sorce:
        Min Tsao "Group least squares regression for linear models with strongly 
        correlated predictor variables". 
        DOI:      https://doi.org/10.1007/s10463-022-00841-7
        arXiv:    https://doi.org/10.48550/arXiv.1804.02499
        Accessed: 2/20/23
        
    '''    
    def __init__(self, method, target=None, n_components=None):
        '''
        Params:
            method : str       
                method to choose when reducing the data
            n_components: int
                rank of reduced data when using 
            
        Raises:
            AssertionError :
                if given method is not one of the available: 
                ["drop", "pca", "gr_treatement", "nothing"]
        '''
        assert(method in ['drop', 'pca', 'gr_teatement', 'nothing']), f'Method {method} is invalid. Available methods: ["drop", "pca", "gr_treatement", "nothing"].'
        self.method = method
        self.target = target
        self.n_components = n_components
    
    def get_the_highest_corr_feature(self, X):
        '''
        Finds the features that has the highest absolute correlation value with 
        the target variable.
        
        Args:
            X : pd.DataFrame (m, n) 
                dataset with j features that has to be treated
            
        Returns:
            highest_corr_feature : str
                name of feature
            
        Raises:
            AssertionError : 
                If "median_house_value" is not present in the dataset passed, 
                the error will be raised, as the method calculates the correlation 
                withthis exact variable.
        '''
        # make sure target variable was passed
        assert(self.target is not None), f'Target variable must be passed as an argument in order to use "drop" method.'
        # concat data and target
        X = pd.concat([X, pd.Series(self.target, index=X.index)], axis=1)
        # find feature with the highest correlation with the target variable        
        highest_corr_feature = np.abs(X.corr()).iloc[-1, :-1].sort_values().index[-1]
        # return the feature
        return highest_corr_feature
    
    def fit(self, X):
        '''
        Given the method it checks and specifies necessary values. If "drop" 
        method is used, then get_the_highest_corr_feature method is used to find
        the right feature to keep. If "pca" method is choosen then value for
        n_components is checked and if the data does not contain any nan values.
        
        Args:
            X : pd.DataFrame (m, j)
                dataset with j features that has to be treated
        Returns:
            self
            
        Raises:
            AssertionError :
                If number of principal components is not specified or is greater
                than the number of columns. The error is also raised if the
                data contains any nan values.
        '''
        if self.method == 'drop':
            # get the feature with the highest corr value
            self.feature_to_keep = self.get_the_highest_corr_feature(X)
        
        if self.method == 'pca':
            # check if "n_components" parameter is specified
            assert((self.n_components is not None) and (self.n_components <= X.shape[1])), f'Specify the number of components that are lower or equal to the length of columns.'
            # check if there are Nans in the data
            assert(~np.isnan(X.values).flatten().any()), f'There are Nan values in the data.'
            # if "median_house_value" is in the passed data, drop the target
            if 'median_house_value' in X.columns:
                X = X.drop(columns=['median_house_value'])
            
        if self.method == 'gr_treatement':
            # to be added, i have to first figure out the context of the article.
            pass
        
        return self
    
    def transform(self, X):
        '''
        Given the method the method transforms the data. If "drop" is choosen
        then only the feature with the highest absolute correlation value is 
        left. If "pca" is choosen, then "small_PCA" class is used to find
        the specified number of principal components.
        
        Args:
            X : pd.DataFrame (m, j)
                dataset with j features that has to be treated
            
        Returns:
            high_feature_X : pd.Series | method="drop"
                (m, 1) reduced data
            X_pca_reduced : pd.Series | method="pca"
                (m, n_components) pca reduced data
        '''
        if self.method == 'drop':
            # return series of the highly corr feature
            high_feature_X = X.loc[:, self.feature_to_keep]
            return high_feature_X
        
        if self.method == 'pca':
            # use small_PCA class
            s_pca = small_PCA(n_components=self.n_components)
            # reduce the data
            X_pca_reduced = s_pca.fit_transform(X)
            # return principal components
            return X_pca_reduced
        
        if self.method == 'gr_treatement':
            #idk
            pass
        
        
class small_Pipeline(BaseEstimator, TransformerMixin):
    '''
    This class helps to deal with building small pipelines to transform the data
    in different modes. Some transformers must be fitted to the training data
    only and then used on the test data. But some must be fitted everytime 
    they are used.
    
    Args:
        list_of_transformers : list of tuples (by default, en empty list)
            List of tuples of transformer and name of column(s) fo form
            [(transformer_1, 'col'), ..., (transformer_2, 'col')].
            If 'col' is a list of columns then the list is passed into a
            multiple_column_handle method that can transform each column.
            
            
    Methods:
        multiple_column_handle(X, transformer, list_of_cols, mode) :
            Handles multiple columns passed. It separates the set into two
            smaller ones and handles them separately by transforming only one 
            set and the concatenating the results. "mode" determines if the 
            transformer should be fitted on the data or to transform the data,
            it is an internal argument that don't need to be specified. If 
            transformer is "MulticollinearityHandler", then the transformation
            must be carried in other way, because some columns are deleted 
            during the transformation. Thus, the final concatenation of the 
            dataframes must be carried without specifying the names of columns.
        fit(X) :
            Fits given transformers to the data specified by the columns passed.
        transform (X) :
            Transforms the data specified by the passed columns. The size of
            returned dataframe is determined by the kind of transformer passed.
    '''
    
    
    def __init__(self, list_of_transformers=[]):
        '''
        Args:
            list_of_transformers : list of tuples (by default [])
                List of tuples of transformer and name of column(s) fo form
                [(transformer_1, 'col'), ..., (transformer_2, 'col')].
                If 'col' is a list of columns then the list is passed into a
                multiple_column_handle method that can transform each column.
        '''
        self.list_of_transformers = list_of_transformers
        
    def multiple_column_handle(self, X, transformer, list_of_cols, mode):
        '''
        Handles multiple columns passed. It separates the set into two smaller 
        ones and handles them separately by transforming only one set and then 
        concatenating the results. "mode" determines if the transformer should 
        be fitted on the data or to transform the data, it is an internal 
        argument that don't need to be specified. If transformer is 
        "MulticollinearityHandler", then the transformation must be carried in
        other way, because some columns are deleted during the transformation. 
        Thus, the final concatenation of the dataframes must be carried without 
        specifying the names of columns.
        
        Args:
            X : pd.DataFrame (m, n) 
                datapoints
            transformer : sklearn Estimator
                sklearn Estimator. Must have fit and transform methods build-in.
            list_of_cols : list of tuples
                List of tuples of transformer and name of column(s) fo form
                [(transformer_1, 'col'), ..., (transformer_2, 'col')].
                Also, 'col' can be another list of columns.
            mode : str
                determines if the transformer should be fitted to the data, or 
                if it should transform the data passed X.
                
        Returns:
            X : pd.DataFrame (m, n)
                If the class is used to fit to the data, this method does not 
                return anything. If the class is used to transform the data it 
                will return concatenated DataFrame with columns specified in 
                list_of_cols transformed.
        '''
        # define the difference between all columns and list_of_columns
        diff_list = np.setdiff1d(X.columns, list_of_cols)
        
        # split the data
        X_tmp_only_sel_cols = X[list_of_cols]
        X_tmp_rest_of_data = X[diff_list]
        
        # given mode and transformer proceed accordingly
        if mode == 'fit':
            
            # fit the transformer on the splitted data
            transformer.fit(X_tmp_only_sel_cols) 
            
        elif (mode == 'transform') and (transformer.__class__.__name__ != 'MulticollinearityHandler'):
            
            # transform the splited data, creating a pandas dataframe with
            # the same columns as list_of_columns
            X_tmp_transformed = pd.DataFrame(transformer.transform(X_tmp_only_sel_cols),
                                             columns=list_of_cols,
                                             index=X.index)
            
            # return the concatenated results
            return pd.concat([X_tmp_rest_of_data, X_tmp_transformed], axis=1)
        
        elif (mode == 'transform') and (transformer.__class__.__name__ == 'MulticollinearityHandler'):
            
            # transform the splitted data without specifying the columns name
            X_tmp_transformed = pd.DataFrame(transformer.transform(X_tmp_only_sel_cols),
                                             index=X.index)
            
            # return the concatenated dataframe
            return pd.concat([X_tmp_rest_of_data, X_tmp_transformed], axis=1)
        
    def fit(self, X):
        '''
        Fits given transformers to the data specified by the columns passed.
        
        Args:
            X : pd.DataFrame (m, n)
                Datapoints
                
        Returns:
            self
        '''
        
        for transformer, column in self.list_of_transformers:
            if np.array(column).size == 1:
                transformer.fit(X[column])
            else:
                self.multiple_column_handle(X, transformer, column, 'fit')
        
        return self
    
    def transform(self, X):
        '''
        Transforms the data specified by the passed columns. The size of returned
        dataframe is determined by the kind of transformer passed.
        
        Args:
            X : pd.DataFrame (m, n)
                Datapoints
                
        Returns:
            X_tmp : pd.DataFrame (m, j)
                Transformed dataset
        '''
        
        X_tmp = X.copy()
        
        for transformer, column in self.list_of_transformers:
            if np.array(column).size == 1:
                X_tmp[column] = transformer.transform(X_tmp[column])
            else:
                X_tmp = self.multiple_column_handle(X_tmp, transformer, column, 'transform')
                
        return X_tmp
    
# NO DOC FOR NOW
class PreprocessPipeline(BaseEstimator, TransformerMixin):
    
    '''any transformer can be access via 
    PreprocessPipeline.name_of_transformer.list_of_transformers[0][0].statistic_in_question'''
    
    def __init__(self):
        self.median_imputer = small_Pipeline([
            (SimpleImputer(strategy='median'), 
             ['longitude', 'latitude', 'housing_median_age',
              'total_rooms', 'total_bedrooms', 'population', 
              'households', 'median_income', 'median_house_value'])
        ])
        self.feature_adder = FeaturesAdder()
        self.outliers_dropper = small_Pipeline([
            (CappedTargetDropper(capped_val=500000.0), 'median_house_value'),
            (DataDropper(method='optimized'), 'income_per_household'),
            (DataDropper(method='optimized'), 'population_per_household'),
            (DataDropper(method='optimized', penalty=0.25), 'rooms_per_household'),
            (DataDropper(method='optimized', penalty=0.25), 'rooms_per_age')
        ])
        self.standardizer = small_Pipeline([
            (StandardScaler(), 
             ['longitude', 'latitude'])
        ])
        self.power_transformer = small_Pipeline([
            (PowerTransformer(),
             ['housing_median_age', 'total_rooms',
              'total_bedrooms', 'population', 'households', 'median_income',
              'median_house_value', 'rooms_per_household', 'income_per_household',
              'income_per_population', 'bedrooms_per_rooms',
              'population_per_household', 'rooms_per_age'])
        ])
        
    def fit(self, X):
        
        self.median_imputer.fit(X)
        
        # since all the later methods are obtained with nans imputed and added
        # new features we have to transform X and add new features,
        # then fit the other classes
        X = self.median_imputer.transform(X)
        X = self.feature_adder.fit_transform(X)
        
        self.outliers_dropper.fit(X)
        self.standardizer.fit(X)
        self.power_transformer.fit(X)
        
        return self
    
    def transform(self, X):
        
        X = self.median_imputer.transform(X)
        X = self.feature_adder.fit_transform(X)
        X = self.outliers_dropper.fit_transform(X)
        X = X.dropna()
        X = self.standardizer.transform(X)
        X = self.power_transformer.transform(X)
        
        X_cat = X['ocean_proximity'].copy()
        X_num = X.drop(columns=['ocean_proximity'])
        
        X_cat.loc[X_cat == 'ISLAND'] = np.nan
        X_merged = pd.merge(X_num, 
                            pd.get_dummies(X_cat, drop_first=True), 
                            left_index=True, right_index=True)
        
        return X_merged