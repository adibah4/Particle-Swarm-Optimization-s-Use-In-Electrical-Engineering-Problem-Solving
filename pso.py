import sys                                                          # To manipulate different parts of the Python runtime environment
import numpy                                                        # To perform a wide variety of mathematical operations on arrays                                        
import matplotlib.pyplot as plt                                     # To make matplotlib works like MATLAB
import pyswarms as ps                                               # To solve a non-linear equation by restructuring it as an optimization problem

from sklearn import metrics                                         # To measure classification performance to create the confusion matrix
from sklearn.metrics import classification_report                   # To measure the quality of predictions from a classification algorithm
from scipy.optimize import fsolve                                   # To finds the root of a given function by using general non-linear solver

actual      = numpy.random.binomial(1,.9,size = 300)                # Draw samples from a binomial distribution for actual values
predicted   = numpy.random.binomial(1,.9,size = 300)                # Draw samples from a binomial distribution for predicted values

# Different measures include: Accuracy, Precision, Sensitivity (Recall), Specificity, and the F-score
Accuracy            = metrics.accuracy_score(actual, predicted)                 # To show the score of accuracy for both actual and predicted values measures how often the model is correct
Precision           = metrics.precision_score(actual, predicted)                # To show the score of precision values, how repeatable a measurement is
Sensitivity_recall  = metrics.recall_score(actual, predicted)                   # To show the score of sensitivity (sometimes called Recall) measures how good the model is at predicting positive results which are positives that have been incorrectly predicted as negative
Specificity         = metrics.recall_score(actual, predicted, pos_label=0)      # To show the score of specificity measures how good the model is at predicting negative results which are negatives that have been incorrectly predicted as positive
F1_score            = metrics.f1_score(actual, predicted)                       # To show the  F-score which is the "harmonic mean" of precision and sensitivity

confusion_matrix    = metrics.confusion_matrix(actual, predicted)               # Use the confusion matrix function on our actual and predicted values once the metrices is imported

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True]) # To create a more interpretable visual display
cm_display.plot()                                                   # To display the plot
plt.title('Confusion Matrix of PSO')                                # To label the title of figure as 'Confusion Matrix of PSO'
plt.show()                                                          # To show the plot


def cost_function(I):       # To predict the cost that will be experienced at a certain activity level
    
    # Fixed parameters
    U = 10                  # Voltage value of the source
    R = 100                 # The resistance of the resistor
    I_s = 9.4e-12           # Reverse bias saturation current of silicon diode at room temperature,  T = 300K
    v_t = 25.85e-3          # Thermal voltage at room temperature,  T = 300K

    c = abs(U - v_t * numpy.log(abs(I[:, 0] / I_s)) - R * I[:, 0])  
    return c                # Return c value

# Setting the Optimizer
options = {'c1': 0.5, 'c2': 0.3, 'w':0.3}                                           # Set-up hyperparameters
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=1, options=options)  # Call instance of PSO
cost, pos = optimizer.optimize(cost_function, iters=30)                             # Perform optimization

print('pos :' , pos[0])     # Print out the position
print('cost :' ,cost)       # Print out the cost function

# Checking the solution 
x = numpy.linspace(0.001, 0.1, 100).reshape(100, 1)
y = cost_function(x)

plt.plot(x, y)                      # Plot the graph with x and y-axis
plt.xlabel('Current I [A]')         # Label x-axis with "Current I [A]"
plt.ylabel('Cost')                  # Label y-axis with "Cost"
plt.show()                          # Show the output graph

c = lambda I: abs(10 - 25.85e-3 * numpy.log(abs(I / 9.4e-12)) - 100 * I)           
initial_guess = 0.09                                                               # initial guess value needed for the numerical method 
current_I = fsolve(func=c, x0=initial_guess)                                       # Current formula 
print('current I :' , current_I[0])                                                # Print out the Current value (I)

# To print all the calculations of accurancy, precision, sensitivity_recall, specificity and F1_score 
print({"Accuracy":Accuracy,"Precision":Precision,"Sensitivity_recall":Sensitivity_recall,"Specificity":Specificity,"F1_score":F1_score}) 