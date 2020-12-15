import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, inputs, targets, learning_rate=0.01):
        self.weights = np.random.randn(3, 1)
        self.features = np.c_[ np.ones((len(inputs), 1)), inputs ]
        self.targets = targets
        
        self.learning_rate = learning_rate
        
        self.x_axis = [ ]
        self.y_axis = [ ]
        
    def model(self):
        return 1 / (1 + np.exp(- (np.matmul(self.features, self.weights))))
    
    def loss(self):
        m = len(self.targets)
        prediction = self.model()
        cost = np.sum(self.targets * np.log(prediction) + (1-self.targets) * np.log(1 - prediction))
        return - cost / m
    
    def train(self, epochs):
        for epoch in range(epochs):
            
            self.x_axis.append(epoch)
            self.y_axis.append(self.loss())
            
            gradient = np.matmul(self.features.T, (self.model() - self.targets))
            
            self.weights -= self.learning_rate * gradient 
    
    def draw(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title('Logistic Regression Loss/Time')
        ax1.set_ylabel('LOSS : J(A)')
        ax1.set_xlabel('EPOCH')
        ax1.plot(b.x_axis, b.y_axis, linestyle='dashed', color='#6ffc03')
