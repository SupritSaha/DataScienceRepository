# Loading required libraries
# ====================================================================
import pandas as pd
import time
from contextlib import contextmanager
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


class stackingClassification:
    """
    Stacking of different classifiers

    Parameters
    -------------------------------------------------------
    folds : number of folds for out of fold
    classifiers : array-like, list of classifiers
    meta_classifier : second level classifier
    use_prob : boolean to signify whether to use probabilities or only predictions
    scoring : metric
    stratify : boolean to signify whether stratified sampling to be done
    """
    def __init__(self,classifiers,meta_classifier,folds=5,use_prob=True,scoring='roc_auc',stratify=True):
        self.folds = folds
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.use_prob = use_prob
        self.scoring = scoring
        self.stratify = stratify

    def fit_stacker(self,x_train,y_train,x_test):
        """
        Fit stacking model and predict

        Parameters
        -------------------------------------------------------
        x_train : training dataset without target
        y_train : target variable
        x_test : test dataset

        Returns
        -------------------------------------------------------
        res : returns predictions
        """
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)

        if self.stratify:
            n_folds = list(StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=2016).split(x_train, y_train))
        else:
            n_folds = list(KFold(n_splits=self.folds, shuffle=True, random_state=2016).split(x_train, y_train))

        train_preds = np.zeros((x_train.shape[0],len(self.classifiers)))
        test_preds = np.zeros((x_test.shape[0],len(self.classifiers)))

        for i, clf in enumerate(self.classifiers):
            test_preds_i = np.zeros((x_test.shape[0],self.folds))
            for j,(train_idx,test_idx) in enumerate(n_folds):
                x_tr = x_train[train_idx]
                y_tr = y_train[train_idx]
                x_holdout = x_train[test_idx]

                print ("Fit %s fold %d" % (str(clf).split('(')[0], j+1))
                clf.fit(x_tr,y_tr)

                if self.use_prob:
                    y_pred = clf.predict_proba(x_holdout)[:,1]
                    train_preds[test_idx, i] = y_pred
                    test_preds_i[:, j] = clf.predict_proba(x_test)[:,1]
                else:
                    y_pred = clf.predict(x_holdout)
                    train_preds[test_idx, i] = y_pred
                    test_preds_i[:, j] = clf.predict(x_test)

            test_preds[:, i] = test_preds_i.mean(axis=1)

        results = cross_val_score(self.meta_classifier, train_preds, y_train , cv = 3, scoring = self.scoring)
        print("Stacker score: %.5f" % (results.mean()))

        self.meta_classifier.fit(train_preds, y_train)
        res = self.meta_classifier.predict_proba(test_preds)[:,1]
        return res


# Example
# ================================================================

# First level models
# --------------------------------------------------------------
# RandomForest params
rf_params = {}
rf_params['n_estimators'] = 200
rf_params['max_depth'] = 6
rf_params['min_samples_split'] = 70
rf_params['min_samples_leaf'] = 30

# ExtraTrees params
et_params = {}
et_params['n_estimators'] = 155
et_params['max_features'] = 0.3
et_params['max_depth'] = 6
et_params['min_samples_split'] = 40
et_params['min_samples_leaf'] = 18

rf_model = RandomForestClassifier(**rf_params)
et_model = ExtraTreesClassifier(**et_params)

# Second level model
# ----------------------------------------------------------------
log_model = LogisticRegression()

# Get stacking results and predictions
# ----------------------------------------------------------------
with timer("Building stacking model"):
    stack = stackingClassification(folds=3,
        meta_classifier = log_model,
        classifiers = (rf_model, et_model),use_prob=False,scoring='accuracy',stratify=False)

    y_pred = stack.fit_stacker(df_train, df_target, df_test)
