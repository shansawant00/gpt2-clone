# Configuration settings for the GPT-2 clone project

class Config:
    def __init__(self):
        self.learning_rate = 5e-5
        self.batch_size = 32
        self.epochs = 3
        self.max_seq_length = 512
        self.model_name = "gpt2"
        self.save_model_path = "./models/"
        self.logging_steps = 100
        self.gradient_accumulation_steps = 1
        self.warmup_steps = 0
        self.weight_decay = 0.01
        self.fp16 = False  # Set to True if using mixed precision training