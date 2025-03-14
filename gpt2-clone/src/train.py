def main():
    import numpy as np
    from model.gpt2 import GPT2
    from utils.config import Config
    from utils.data_utils import load_data

    # Load configuration
    config = Config()

    # Load data
    train_data, val_data = load_data(config.data_path)

    # Initialize model
    model = GPT2(config)

    # Training loop
    for epoch in range(config.num_epochs):
        for batch in train_data:
            # Forward pass
            loss = model.train_on_batch(batch)

            # Print loss
            print(f'Epoch: {epoch + 1}, Loss: {loss}')

    # Save the model
    model.save(config.model_save_path)

if __name__ == "__main__":
    main()