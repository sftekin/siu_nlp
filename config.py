model_config = {
    'n_layers': 1,
    'lstm_dim': 256,
    'feature_dim': 2,
    'drop_prob': 0.5,
    'batch_size': 16,
    'output_dim': 1,
}

batch_params = {
    'batch_size': 16,
    'num_works': 0,
    'shuffle': True,
}

train_params = {
    'n_epoch': 50,
    'clip': 5,
    'lr': 0.003,
    'seq_len': 15,
    'eval_every': 500
}