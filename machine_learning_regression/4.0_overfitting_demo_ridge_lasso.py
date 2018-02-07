
# coding: utf-8

# # Overfitting demo
# 
# ## Create a dataset based on a true sinusoidal relationship
# Let's look at a synthetic dataset consisting of 30 points drawn from the sinusoid $y = \sin(4x)$:

# In[1]:

import graphlab
import math
import random
import numpy
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')


# Create random values for x in interval [0,1)

# In[2]:

random.seed(98103)
n = 30
x = graphlab.SArray([random.random() for i in range(n)]).sort()


# Compute y

# In[3]:

y = x.apply(lambda x: math.sin(4*x))


# Add random Gaussian noise to y

# In[4]:

random.seed(1)
e = graphlab.SArray([random.gauss(0,1.0/3.0) for i in range(n)])
y = y + e


# ### Put data into an SFrame to manipulate later

# In[5]:

data = graphlab.SFrame({'X1':x,'Y':y})
data


# ### Create a function to plot the data, since we'll do it many times

# In[6]:

def plot_data(data):    
    plt.plot(data['X1'],data['Y'],'k.')
    plt.xlabel('x')
    plt.ylabel('y')

plot_data(data)


# ## Define some useful polynomial regression functions

# Define a function to create our features for a polynomial regression model of any degree:

# In[7]:

def polynomial_features(data, deg):
    data_copy=data.copy()
    for i in range(1,deg):
        data_copy['X'+str(i+1)]=data_copy['X'+str(i)]*data_copy['X1']
    return data_copy


# Define a function to fit a polynomial linear regression model of degree "deg" to the data in "data":

# In[8]:

def polynomial_regression(data, deg):
    model = graphlab.linear_regression.create(polynomial_features(data,deg), 
                                              target='Y', l2_penalty=0.,l1_penalty=0.,
                                              validation_set=None,verbose=False)
    return model


# Define function to plot data and predictions made, since we are going to use it many times.

# In[9]:

def plot_poly_predictions(data, model):
    plot_data(data)

    # Get the degree of the polynomial
    deg = len(model.coefficients['value'])-1
    
    # Create 200 points in the x axis and compute the predicted value for each point
    x_pred = graphlab.SFrame({'X1':[i/200.0 for i in range(200)]})
    y_pred = model.predict(polynomial_features(x_pred,deg))
    
    # plot predictions
    plt.plot(x_pred['X1'], y_pred, 'g-', label='degree ' + str(deg) + ' fit')
    plt.legend(loc='upper left')
    plt.axis([0,1,-1.5,2])


# Create a function that prints the polynomial coefficients in a pretty way :)

# In[10]:

def print_coefficients(model):    
    # Get the degree of the polynomial
    deg = len(model.coefficients['value'])-1

    # Get learned parameters as a list
    w = list(model.coefficients['value'])

    # Numpy has a nifty function to print out polynomials in a pretty way
    # (We'll use it, but it needs the parameters in the reverse order)
    print 'Learned polynomial for degree ' + str(deg) + ':'
    w.reverse()
    print numpy.poly1d(w)


# ## Fit a degree-2 polynomial

# Fit our degree-2 polynomial to the data generated above:

# In[11]:

model = polynomial_regression(data, deg=2)


# Inspect learned parameters

# In[12]:

print_coefficients(model)


# Form and plot our predictions along a grid of x values:

# In[13]:

plot_poly_predictions(data,model)


# ## Fit a degree-4 polynomial

# In[14]:

model = polynomial_regression(data, deg=4)
print_coefficients(model)
plot_poly_predictions(data,model)


# ## Fit a degree-16 polynomial

# In[15]:

model = polynomial_regression(data, deg=16)
print_coefficients(model)


# ###Woah!!!!  Those coefficients are *crazy*!  On the order of 10^6.

# In[16]:

plot_poly_predictions(data,model)


# ### Above: Fit looks pretty wild, too.  Here's a clear example of how overfitting is associated with very large magnitude estimated coefficients.

# # 

# # 

#  # 

#  # 

# # Ridge Regression

# Ridge regression aims to avoid overfitting by adding a cost to the RSS term of standard least squares that depends on the 2-norm of the coefficients $\|w\|$.  The result is penalizing fits with large coefficients.  The strength of this penalty, and thus the fit vs. model complexity balance, is controled by a parameter lambda (here called "L2_penalty").

# Define our function to solve the ridge objective for a polynomial regression model of any degree:

# In[17]:

def polynomial_ridge_regression(data, deg, l2_penalty):
    model = graphlab.linear_regression.create(polynomial_features(data,deg), 
                                              target='Y', l2_penalty=l2_penalty,
                                              validation_set=None,verbose=False)
    return model


# ## Perform a ridge fit of a degree-16 polynomial using a *very* small penalty strength

# In[18]:

print data
model = polynomial_ridge_regression(data, deg=16, l2_penalty=1e-25)
print_coefficients(model)


# In[19]:

plot_poly_predictions(data,model)


# ## Perform a ridge fit of a degree-16 polynomial using a very large penalty strength

# In[20]:

model = polynomial_ridge_regression(data, deg=16, l2_penalty=100)
print_coefficients(model)


# In[21]:

plot_poly_predictions(data,model)


# ## Let's look at fits for a sequence of increasing lambda values

# In[22]:

for l2_penalty in [1e-25, 1e-10, 1e-6, 1e-3, 1e2]:
    model = polynomial_ridge_regression(data, deg=16, l2_penalty=l2_penalty)
    print 'lambda = %.2e' % l2_penalty
    print_coefficients(model)
    print '\n'
    plt.figure()
    plot_poly_predictions(data,model)
    plt.title('Ridge, lambda = %.2e' % l2_penalty)


# In[23]:

data


# ## Perform a ridge fit of a degree-16 polynomial using a "good" penalty strength

# We will learn about cross validation later in this course as a way to select a good value of the tuning parameter (penalty strength) lambda.  Here, we consider "leave one out" (LOO) cross validation, which one can show approximates average mean square error (MSE).  As a result, choosing lambda to minimize the LOO error is equivalent to choosing lambda to minimize an approximation to average MSE.

# In[24]:

# LOO cross validation -- return the average MSE
def loo(data, deg, l2_penalty_values):
    # Create polynomial features
    data = polynomial_features(data, deg)
    
    # Create as many folds for cross validatation as number of data points
    num_folds = len(data)
    folds = graphlab.cross_validation.KFold(data,num_folds)
    
    # for each value of l2_penalty, fit a model for each fold and compute average MSE
    l2_penalty_mse = []
    min_mse = None
    best_l2_penalty = None
    for l2_penalty in l2_penalty_values:
        next_mse = 0.0
        for train_set, validation_set in folds:
            # train model
            model = graphlab.linear_regression.create(train_set,target='Y', 
                                                      l2_penalty=l2_penalty,
                                                      validation_set=None,verbose=False)
            
            # predict on validation set 
            y_test_predicted = model.predict(validation_set)
            # compute squared error
            next_mse += ((y_test_predicted-validation_set['Y'])**2).sum()
        
        # save squared error in list of MSE for each l2_penalty
        next_mse = next_mse/num_folds
        l2_penalty_mse.append(next_mse)
        if min_mse is None or next_mse < min_mse:
            min_mse = next_mse
            best_l2_penalty = l2_penalty
            
    return l2_penalty_mse,best_l2_penalty


# Run LOO cross validation for "num" values of lambda, on a log scale

# In[25]:

l2_penalty_values = numpy.logspace(-4, 10, num=10)
l2_penalty_mse,best_l2_penalty = loo(data, 16, l2_penalty_values)


# Plot results of estimating LOO for each value of lambda

# In[26]:

plt.plot(l2_penalty_values,l2_penalty_mse,'k-')
plt.xlabel('$\ell_2$ penalty')
plt.ylabel('LOO cross validation error')
plt.xscale('log')
plt.yscale('log')


# Find the value of lambda, $\lambda_{\mathrm{CV}}$, that minimizes the LOO cross validation error, and plot resulting fit

# In[27]:

best_l2_penalty


# In[28]:

model = polynomial_ridge_regression(data, deg=16, l2_penalty=best_l2_penalty)
print_coefficients(model)


# In[29]:

plot_poly_predictions(data,model)


# # 

# # 

# # 

# # 

# # Lasso Regression

# Lasso regression jointly shrinks coefficients to avoid overfitting, and implicitly performs feature selection by setting some coefficients exactly to 0 for sufficiently large penalty strength lambda (here called "L1_penalty").  In particular, lasso takes the RSS term of standard least squares and adds a 1-norm cost of the coefficients $\|w\|$.

# Define our function to solve the lasso objective for a polynomial regression model of any degree:

# In[30]:

def polynomial_lasso_regression(data, deg, l1_penalty):
    model = graphlab.linear_regression.create(polynomial_features(data,deg), 
                                              target='Y', l2_penalty=0.,
                                              l1_penalty=l1_penalty,
                                              validation_set=None, 
                                              solver='fista', verbose=False,
                                              max_iterations=3000, convergence_threshold=1e-10)
    return model


# ## Explore the lasso solution as a function of a few different penalty strengths

# We refer to lambda in the lasso case below as "l1_penalty"

# In[31]:

for l1_penalty in [0.0001, 0.01, 0.1, 10]:
    model = polynomial_lasso_regression(data, deg=16, l1_penalty=l1_penalty)
    print 'l1_penalty = %e' % l1_penalty
    print 'number of nonzeros = %d' % (model.coefficients['value']).nnz()
    print_coefficients(model)
    print '\n'
    plt.figure()
    plot_poly_predictions(data,model)
    plt.title('LASSO, lambda = %.2e, # nonzeros = %d' % (l1_penalty, (model.coefficients['value']).nnz()))


# Above: We see that as lambda increases, we get sparser and sparser solutions.  However, even for our non-sparse case for lambda=0.0001, the fit of our high-order polynomial is not too wild.  This is because, like in ridge, coefficients included in the lasso solution are shrunk relative to those of the least squares (unregularized) solution.  This leads to better behavior even without sparsity.  Of course, as lambda goes to 0, the amount of this shrinkage decreases and the lasso solution approaches the (wild) least squares solution.

# In[ ]:




# In[ ]:



