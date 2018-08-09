import tensorflow as tf


class Trainer():
    def __init__(self, model, optimizer, learning_rate):
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        
