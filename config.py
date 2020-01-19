model_config = {
    'LinearSVM': {
        'c_list': [1],
        'tol': [1e-4],
        'cv': 3,
        'scoring': 'f1_micro',
        'load': False
    },
    'RandomForest': {
        'n_estimator': [3000],
        'max_depth': [1000],
        'cv': 3,
        'scoring': 'f1_micro',
        'load': False
    },
    'LSTM': {
        'n_layers': 1,
        'lstm_dim': 256,
        'drop_prob': 0.5,
        'output_dim': 1,
    }
}

batch_params = {
    'batch_size': 32,
    'num_works': 0,
    'shuffle': True,
}

train_params = {
    'n_epoch': 20,
    'clip': 5,
    'lr': 0.00007,
    'seq_len': 15,
    'eval_every': 500
}