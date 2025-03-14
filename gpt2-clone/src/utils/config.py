# filepath: /gpt2-clone/gpt2-clone/src/utils/config.py
MODEL_PARAMS = {
    'num_heads': 8,
    'd_model': 512,
    'd_ff': 2048,
    'dropout_rate': 0.1,
    'max_position_embeddings': 512,
    'vocab_size': 50257,
}

TRAINING_CONFIG = {
    'batch_size': 32,
    'learning_rate': 5e-5,
    'num_epochs': 3,
    'max_grad_norm': 1.0,
}

DATA_PATHS = {
    'train_data': 'data/train.txt',
    'valid_data': 'data/valid.txt',
}