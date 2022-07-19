from sklearn import metrics

def train_model(clf, feature_vect_train, feature_vect_test, y_train, y_test):
    clf.fit(feature_vect_train, y_train)
    predict = clf.predict(feature_vect_test)

    acc = metrics.accuracy_score(predict, y_test)
    f1 = metrics.f1_score(predict, y_test, average='weighted')
    print(metrics.classification_report(predict, y_test, digits=4))

    return acc, f1
