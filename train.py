import os
import pickle
import itertools

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from embedding import MeanEmbedding


def train_model(data, **params):
    # Check pretrained model
    model_path = os.path.join('models', params['model_name']+'.pkl')
    if os.path.isfile(model_path) and params['load']:
        model_file = open(model_path, 'rb')
        model = pickle.load(model_file)
        print('model {} found and loaded'.format(params['model_name']))
        return model

    print('model {} is training...'.format(params['model_name']))
    X_train, X_test, y_train, y_test = data
    word2vec = MeanEmbedding()

    best_score = 0
    best_params = []
    for n_est, max_depth in itertools.product(params['n_estimator'], params['max_depth']):
        clf = RandomForestClassifier(n_estimators=n_est, n_jobs=-1)

        pipe = Pipeline([
            ('word2vec', word2vec),
            ('clf', clf)
        ])
        cv_score = cross_val_score(pipe, X_train, y_train,
                                   cv=params['cv'], scoring=params['scoring'])
        print('n_est:{}, max_depth:{}, cv_score:{}'.format(n_est, max_depth, cv_score))
        cv_score = sum(cv_score) / params['cv']
        if best_score < cv_score:
            best_score = cv_score
            best_params = [n_est, max_depth]

    print('Training finished best params = n_est:{}, max_depth:{}'.format(*best_params))

    clf = LinearSVC(C=best_params[0], tol=best_params[1],
                    multi_class='ovr', max_iter=5000, dual=False)
    pipe = Pipeline([
        ('word2vec', word2vec),
        ('clf', clf)
    ])
    pipe.fit(X_train, y_train)
    print("Best parameter Score (CV f1_micro_score=%0.3f):" % pipe.score(X_test, y_test))
    print('Saving the model')
    model_file = open(model_path, 'wb')
    pickle.dump(pipe, model_file)
    return pipe


