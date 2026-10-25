import gc
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor


class CosineKNN(BaseEstimator):
    def __init__(self, n_neighbors=45, batch_size=128):
        self.n_neighbors = n_neighbors
        self.batch_size = batch_size

    @staticmethod
    def _normalize(X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or not np.isfinite(X).all():
            raise ValueError('Expected a finite two-dimensional embedding matrix.')
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return np.ascontiguousarray(X / np.where(norms > 0, norms, 1))

    def fit(self, X, y):
        self.vectors_ = self._normalize(X)
        self.labels_ = np.asarray(y).copy()
        if (not 1 <= self.n_neighbors <= len(self.vectors_) or self.batch_size < 1
                or self.labels_.shape != (len(self.vectors_),) or not set(self.labels_) <= {0, 1, 2}):
            raise ValueError('Invalid k, batch size, or labels (expected classes 0,1,2).')
        self.n_features_in_ = self.vectors_.shape[1]
        return self

    def predict_scores(self, X):
        check_is_fitted(self, ('vectors_', 'labels_'))
        query = self._normalize(X)
        if query.shape[1] != self.n_features_in_:
            raise ValueError('Embedding dimension differs from fitting data.')
        result = np.zeros((len(query), 3), dtype=np.float64)
        with threadpool_limits(limits=2):
            for start in range(0, len(query), self.batch_size):
                cosine = query[start:start + self.batch_size] @ self.vectors_.T
                np.clip(cosine, -1., 1., out=cosine)
                cutoff = np.partition(cosine, -self.n_neighbors, axis=1)[:, -self.n_neighbors]
                for row, similarity in enumerate(cosine):
                    strict = np.flatnonzero(similarity > cutoff[row])
                    ties = np.flatnonzero(similarity == cutoff[row])[:self.n_neighbors - len(strict)]
                    nearest = np.concatenate([strict, ties])
                    nearest = nearest[np.lexsort((nearest, -similarity[nearest]))]
                    labels, values = self.labels_[nearest], similarity[nearest]
                    result[start + row] = [np.sum(np.where(labels == cls, values, 0.), dtype=np.float64)
                                           for cls in range(3)]
                del cutoff, cosine, similarity
        return result


class SortedCSR(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X=sparse.csr_matrix(X)
        X.sort_indices()
        return X

def make_meta(config):
    return make_pipeline(
        SimpleImputer(strategy='median', keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=config['C'], max_iter=1500, random_state=42),
    )


class AuthorStackingClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, svm_factory, config, columns, knn_columns):
        self.svm_factory = svm_factory
        self.config = config
        self.columns = columns
        self.knn_columns = knn_columns

    def _native_data(self, X, fit=False):
        specification = self.columns['tree']
        data = X[specification['ordered']].copy()
        data[specification['numeric']] = data[specification['numeric']].astype(np.float32)
        for column in specification['categorical']:
            data[column] = data[column].fillna('unknown').astype(str)
            if fit:
                data[column] = data[column].astype('category')
                self.category_values_[column] = list(data[column].cat.categories)
            else:
                known = self.category_values_[column]
                data[column] = pd.Categorical(data[column].where(data[column].isin(known)),
                                              categories=known)
        return data

    def _fit_bases(self, X, y, target):
        models = {}
        print('Fitting text SVM...', flush=True)
        models['svm'] = self.svm_factory().fit(X, y)
        gc.collect()
        print('Fitting residual regression...', flush=True)
        specification = self.columns['regression']
        config = self.config['regression']
        preprocessing = ColumnTransformer([
            ('num', SimpleImputer(strategy='median', keep_empty_features=True), specification['numeric']),
            ('cat', OneHotEncoder(handle_unknown='ignore', min_frequency=20, sparse_output=False),
             specification['categorical']),
        ], sparse_threshold=0)
        models['regression'] = Pipeline([
            ('prep', preprocessing),
            ('float32', FunctionTransformer(np.float32)),
            ('clf', HistGradientBoostingRegressor(
                loss=config['loss'], max_leaf_nodes=config['leaves'], max_iter=400,
                learning_rate=.05, min_samples_leaf=30, l2_regularization=5,
                early_stopping=False, random_state=42)),
        ]).fit(X, target)
        print('Fitting boosted classifier...', flush=True)
        from lightgbm import LGBMClassifier
        config = self.config['tree']
        self.category_values_ = {}
        data = self._native_data(X, fit=True)
        classifier = LGBMClassifier(
            n_estimators=config['iterations'], num_leaves=config['leaves'],
            learning_rate=config['learning_rate'], reg_lambda=config['l2'],
            class_weight={0: 3., 1: 1., 2: 3.}, min_child_samples=40,
            colsample_bytree=.9, subsample=.8, subsample_freq=1,
            random_state=42, n_jobs=4, verbosity=-1, metric='None',
        )
        classifier.fit(data, y, categorical_feature=self.columns['tree']['categorical'])
        models['tree'] = classifier
        models['category_values'] = dict(self.category_values_)
        models['knn'] = CosineKNN(self.config['knn']['k']).fit(X[self.knn_columns].to_numpy(np.float32), y)
        return models

    def _base_features(self, X, models):
        parts = []
        parts.append(models['svm'].decision_function(X))
        parts.append(models['regression'].predict(X)[:, None])
        self.category_values_ = models['category_values']
        parts.append(models['tree'].predict_proba(self._native_data(X)))
        parts.append(models['knn'].predict_scores(X[self.knn_columns].to_numpy(np.float32)))
        return np.column_stack(parts)

    def fit(self, X, y, residual_target=None):
        y = np.asarray(y)
        if residual_target is None:
            raise ValueError('Pass the training residual target separately from predictor columns.')
        target = np.asarray(residual_target)
        self.models_ = []
        values = None
        with threadpool_limits(limits=2):
            folds = GroupKFold(self.config['stack_folds'])
            for fold, (fit, valid) in enumerate(folds.split(X, groups=X.author), 1):
                print(f'Preparing stacking fold {fold}/{self.config["stack_folds"]}...', flush=True)
                models = self._fit_bases(X.iloc[fit], y[fit], target[fit])
                held_out = self._base_features(X.iloc[valid], models)
                if values is None:
                    values = np.empty((len(X), held_out.shape[1]))
                values[valid] = held_out
                self.models_.append(models)
                gc.collect()
            self.meta_ = make_meta(self.config['meta']).fit(values, y)
        self.classes_ = np.arange(3)
        return self

    def predict_proba(self, X):
        with threadpool_limits(limits=2):
            values = np.mean([self._base_features(X, models) for models in self.models_], axis=0)
            return self.meta_.predict_proba(values)

    def predict(self, X):
        probability = self.predict_proba(X)
        return (np.log(np.maximum(probability, 1e-12)) + self.config['meta']['offsets']).argmax(axis=1)
