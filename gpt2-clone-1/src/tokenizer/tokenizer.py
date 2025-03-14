class Tokenizer:
    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)

    def load_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = f.read().splitlines()
        return {word: idx for idx, word in enumerate(vocab)}

    def encode(self, text):
        tokens = text.split()
        return [self.vocab.get(token, self.vocab.get('<unk>')) for token in tokens]

    def decode(self, token_ids):
        reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        return ' '.join([reverse_vocab.get(token_id, '<unk>') for token_id in token_ids])