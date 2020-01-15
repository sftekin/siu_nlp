import os
import pickle
import itertools

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from embedding import MeanEmbedding
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def train_model(data, **params):
    # Check pre-trained model
    model_path = os.path.join('models', params['model_name']+'.pkl')
    if os.path.isfile(model_path) and params['load']:
        model_file = open(model_path, 'rb')
        model = pickle.load(model_file)
        print('model {} found and loaded'.format(params['model_name']))
        return model

    print('model {} is training...'.format(params['model_name']))
    X_train, X_test, y_train, y_test = data
    word2vec = MeanEmbedding()

    if params['model_name'] == 'linear_svm':
        model_params = [params['c_list'], params['tol']]
    else:
        model_params = [params['n_estimator'], params['max_depth']]

    best_score = 0
    best_params = []
    for p1, p2 in itertools.product(*model_params):
        print('Training started..')
        if params['model_name'] == 'linear_svm':
            clf = LinearSVC(C=p1, tol=p2, multi_class='ovr',
                            max_iter=2000, dual=False)
        else:
            clf = RandomForestClassifier(n_estimators=p1, max_depth=p2)

        pipe = Pipeline([
            ('word2vec', word2vec),
            ('clf', clf)
        ])
        cv_score = cross_val_score(pipe, X_train, y_train,
                                   cv=params['cv'], scoring=params['scoring'])
        print('first_param:{}, '
              'second_param:{}, '
              'cv_score:{}'.format(p1, p2, cv_score))

        cv_score = sum(cv_score) / params['cv']
        if best_score < cv_score:
            best_score = cv_score
            best_params = [p1, p2]

    print('Training finished best params = '
          'first_param:{}, second_param:{}'.format(*best_params))

    if params['model_name'] == 'linear_svm':
        clf = LinearSVC(C=best_params[0], tol=best_params[1],
                        multi_class='ovr', max_iter=5000, dual=False)
    else:
        clf = RandomForestClassifier(n_estimators=best_params[0],
                                     max_depth=best_params[1])

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


