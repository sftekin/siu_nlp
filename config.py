model_config = {
    'LSTM': {
        'n_layers': 1,
        'lstm_dim': 256,
        'drop_prob': 0.5,
        'output_dim': 1,
    }
}

batch_params = {
    'batch_size': 64,
    'num_works': 0,
    'shuffle': True,
}

train_params = {
    'n_epoch': 20,
    'clip': 5,
    'lr': 0.00007,
    'seq_len': 15,
    'eval_every': 50
}
