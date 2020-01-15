model_config = {
    'LinSVM': {
        'c_list': [0.1, 2, 5, 10],
        'tol': [1e-4, 1e-5],
        'cv': 3,
        'scoring': 'f1_micro',
        'model_name': 'linear_svm',
        'load': False
    },
    'RandomForest': {
        'n_estimator': [1000, 3000],
        'max_depth': [1000, 3000],
        'cv': 3,
        'scoring': 'f1_micro',
        'model_name': 'random_forest',
        'load': False
    }
}
