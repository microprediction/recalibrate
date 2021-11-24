from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

# TODO. Comparison

if __name__=='__main__':
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)
    base_clf = GaussianNB()
    calibrated_clf = CalibratedClassifierCV(base_estimator=base_clf, cv=3)
    calibrated_clf.fit(X, y)
    CalibratedClassifierCV(base_estimator=GaussianNB(), cv=3)

    X, y = make_classification(n_samples=100, n_features=2,n_redundant=0, random_state=42)
    X_train, X_calib, y_train, y_calib = train_test_split(X, y, random_state=42)
    base_clf = GaussianNB()
    base_clf.fit(X_train, y_train)
    GaussianNB()
    calibrated_clf = CalibratedClassifierCV(
         base_estimator=base_clf,
         cv="prefit")
    calibrated_clf.fit(X_calib, y_calib)
    CalibratedClassifierCV(base_estimator=GaussianNB(), cv='prefit')
