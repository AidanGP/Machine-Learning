import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, inputs, targets, learning_rate=0.01):
        shape = inputs.shape[1] + 1
        self.weights = np.random.randn(shape, 1)
        self.features = np.c_[ np.ones((len(inputs), 1)), inputs ]
        self.targets = targets
        
        self.learning_rate = learning_rate
        
        self.x_axis = [ ]
        self.y_axis = [ ]
        
    def model(self):
        return 1 / (1 + np.exp(- 1 * (np.matmul(self.features, self.weights))))
    
    def loss(self):
        m = len(self.targets)
        prediction = self.model()
        cost = np.sum(self.targets * np.log(prediction) + (1 - self.targets) * np.log(1 - prediction))
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
        ax1.plot(self.x_axis, self.y_axis, linestyle='dashed', color='#6ffc03')
        
    def get_equation(self):
        eqn = []
        coefficients = self.weights.flatten()
        for term in range(len(coefficients)):
            coeff = str("%.2f" % coefficients[term])
            X = '(' + coeff + ' x X' + str(term) + ')'
            eqn.append(X)
        return 'y = ' + 'sigmoid(' + ' + '.join(eqn) + ')'
    
    def get_results(self, inputs):
        features = np.c_[ np.ones((len(inputs), 1)), inputs ]
        model = 1 / (1 + np.exp(- 1 * (np.matmul(features, self.weights))))
        return [(inputs[i].tolist(), "%.2f" % model[i]) for i in range(len(model))]
