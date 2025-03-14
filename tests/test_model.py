import unittest
from src.model.gpt2 import GPT2Model

class TestGPT2Model(unittest.TestCase):

    def setUp(self):
        self.model = GPT2Model()

    def test_model_initialization(self):
        self.assertIsNotNone(self.model)

    def test_forward_pass(self):
        input_data = "Hello, world!"
        output = self.model.forward(input_data)
        self.assertIsNotNone(output)

if __name__ == '__main__':
    unittest.main()