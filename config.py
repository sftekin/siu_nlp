model_config = {
    'LinSVM': {
        'c_list': [1],
        'tol': [1e-4],
        'cv': 3,
        'scoring': 'f1_micro',
        'model_name': 'linear_svm',
        'load': False
    },
    'RandomForest': {
        'n_estimator': [3000],
        'max_depth': [1000],
        'cv': 3,
        'scoring': 'f1_micro',
        'model_name': 'random_forest',
        'load': False
    }
}
