import os
import pickle
import itertools
import torch
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def train_clf(X, y, embed, model_name, **params):
    model_path = os.path.join('models', model_name + '.pkl')

    word_embeds = tf_embed_corpus(embed, X)

    print('model {} is training...'.format(model_name))

    if model_name == 'LinearSVM':
        model_params = [params['c_list'], params['tol']]
    else:
        model_params = [params['n_estimator'], params['max_depth']]

    best_score = 0
    best_params = []
    for p1, p2 in itertools.product(*model_params):
        if model_name == 'LinearSVM':
            clf = LinearSVC(C=p1, tol=p2, multi_class='ovr',
                            max_iter=2000, dual=False)
        else:
            clf = RandomForestClassifier(n_estimators=p1, max_depth=p2, n_jobs=-1)

        cv_score = cross_val_score(clf, word_embeds, y,
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

    if model_name == 'LinearSVM':
        clf = LinearSVC(C=best_params[0], tol=best_params[1],
                        multi_class='ovr', max_iter=5000, dual=False)
    else:
        clf = RandomForestClassifier(n_estimators=best_params[0],
                                     max_depth=best_params[1],
                                     n_jobs=-1)

    print('Training for best params')
    clf.fit(word_embeds, y)

    print('Saving the model')
    model_file = open(model_path, 'wb')
    pickle.dump(clf, model_file)
    return clf


def tf_embed_corpus(embed, X):
    word_embeds = []
    for sen in X:
        sen_vec = embed(torch.tensor(sen)).numpy()
        word_embeds.append(np.mean(sen_vec, axis=0))
    word_embeds = np.stack(word_embeds, axis=0)
    return word_embeds
