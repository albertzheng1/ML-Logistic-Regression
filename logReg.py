import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as lin
import pandas as pd
import scipy.optimize as op
import math

# Get X from the initial 2d array of data
def getX(mat):
    X = np.array(mat[:, 0:mat.shape[1] - 1])
    return X

# Get y from the initial 2d array of data
def getY(mat):
    y = np.array([mat[:, mat.shape[1] - 1]])
    y = y.T  # T can only be called on numpy arrays, not arrays
    return y

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Defined hypothesis for logistic regression
def h(x, theta):
    return sigmoid( np.dot(x, theta) )

# Computes cost function with theta (theta_0 to theta_n) as input value
# Postcondition: Returns float value
def computeCost(X, y, theta):
    m = y.shape[0]
    sum = 0.0

    # Implement cost function
    # log refers to natural log
    for i in range(0, m):
        if y[i] == 1:
            sum += -np.log(h(np.array([X[i, :]]), theta))
        elif y[i] == 0:
            sum += -np.log(1 - h(np.array([X[i, :]]), theta))

    res = (1 / m) * sum[0, 0]

    return round(res, 100)

# Gradient descent (along with the optimization module in SciPy) both allow for minimizing the cost function
def gradDesc(X, y, theta, alpha, num_iters, returnJ):
    theta = theta.astype(float)
    m = y.shape[0]

    # Create space to store values of error with each iteration
    J = np.array([])
    J = np.append(J, computeCost(X, y, theta))

    # Run this for a specified number of iterations
    for k in range(0, num_iters):
        # Initialize delta
        delta = np.zeros((theta.shape[0],1), dtype='float64')

        # Define delta
        for i in range(0, m):

            # Update delta_0 to delta_n+1
            for d in range(0,delta.shape[0]):
                delta[d,0] += ( h(np.array([X[i, :]]), theta) - y[i, 0] ) * X[i, d]

        # Update theta
        theta -= alpha * (1 / m) * delta

        # Update cost using the theta corresponding to the current iteration
        J = np.append(J, computeCost(X, y, theta))

    # Decide whether this function returns optimized theta or values of J
    if returnJ == False:
        return theta
    else:
        return J

# If there are too few training examples, function does not work well for some reason
# Return normalized X (excluding x_0)
def featureNorm(X):
    res = np.copy(X)
    for i in range(1,X.shape[1]):
        res[:, i] = ( res[:, i] - np.mean(res[:, i]) ) / np.std(res[:, i])
    return res

# Predict whether the result is 0 or 1 using learned parameters theta
# If X is normalized, make sure to input normalized values of new x values
def predict(x, theta_min):
    x = x.astype(float)
    if h(x,theta_min) >= 0.5:
        return 1.0
    elif h(x,theta_min) < 0.5:
        return 0.0

# Compute percentage of training examples that are correctly predicted by the optimized decision boundary
# If X is normalized, use normalized X as a parameter instead of original X
def computeAccuracy(X, y, theta_min):
    num_correct = 0
    for i in range(0,X.shape[0]):
        if predict( np.array([X[i,:]]),theta_min ) == y[i,0]:
            num_correct += 1
    return num_correct / X.shape[0]


# Adds more features to build a more flexible classifier
# May cause overfitting
# Precondition:
#   X initially only has features x_0, x_1, x_2
#   deg >= 2
def addFeatures(X, degree):
    m = X.shape[0]
    res = np.copy(X)

    # Starting at degree 2, for each degree, the exponent of x_1 goes from high to low, while the
    # exponent of x_2 goes from low to high
    # The sum of exponents in each degree tier equals the degree (eg deg 2: 2+0,1+1,0+2)
    # Each pair of d,k represents a new feature
    for d in range(2, degree+1):
        for k in range(0,d+1):

            # Temporary column to store values for a particular feature. Once filled up, will be appended to
            # X as a brand new feature
            temp = np.zeros((m, 1))

            # x_1 ^ (d - k) * x_2 ^ k
            for i in range(0,m):
                temp[i,0] = X[i,1] ** (d-k) * X[i,2] ** k

            # Append brand new feature as a column vector to X
            res = np.append(res, temp, axis=1)

    return res

# Intuitive tool to visualize the features being added
# Assume X initially only has features x_0, x_1, x_2
# Precondition:
#   deg >= 2
def visualizeAddFeat(deg):
    for d in range(2, deg + 1):
        for k in range(0, d + 1):
            print(f"x_1 ** ({d - k}) * x_2 ** {k}")
        print()

# Used to create headers for csv files, when there is a high number of features
def makeFeatHeads(deg):
    for d in range(0, deg + 1):
        for k in range(0, d + 1):
            print(f"x1^{d - k} * x2^{k}, ", end='')

# Computes regularized cost function with theta (theta_0 to theta_n) as input value
# Combats overfitting
# Postcondition: Returns float value
def computeRegCost(X, y, theta, lamb):
    m = X.shape[0]
    n = X.shape[1]
    term1 = 0.0
    term2 = 0.0

    # Implement cost function
    # log refers to natural log
    for i in range(0, m):
        term1 += y[i, 0] * np.log(h(np.array([X[i, :]]), theta)) \
                 + (1 - y[i, 0]) * np.log(1 - h(np.array([X[i, :]]), theta))

    # Do not regularize the theta_0 parameter
    for j in range(1,n):
        term2 += theta[j,0] ** 2

    res = (-1/m) * term1[0,0] + (lamb/(2*m)) * term2

    return round(res, 100)

# Regularized gradient descent to find the optimal theta
def regGradDesc(X, y, theta, alpha, lamb, num_iters, returnJ):
    theta = theta.astype(float)
    m = X.shape[0]

    # Create space to store values of error with each iteration
    J = np.array([])
    J = np.append(J, computeRegCost(X, y, theta, lamb))

    # Run this for a specified number of iterations
    for k in range(0, num_iters):

        # Initialize delta
        delta = np.zeros((theta.shape[0],1), dtype='float64')

        # Define delta
        for i in range(0, m):

            # Update delta_0
            delta[0, 0] += (h(np.array([X[i, :]]), theta) - y[i, 0]) * X[i, 0]

            # Update the entries delta_1 to delta_n+1
            for d in range(1,delta.shape[0]):
                delta[d,0] += ( h(np.array([X[i, :]]), theta) - y[i, 0] ) * X[i, d]

        # Update theta
        theta = theta * (1 - alpha * (lamb/m)) - alpha * (1 / m) * delta

        # Update cost using the theta corresponding to the current iteration
        J = np.append(J, computeRegCost(X, y, theta, lamb))

    # Decide whether this function returns optimized theta or values of J
    if returnJ == False:
        return theta
    else:
        return J

def mapFeat(x1,x2,degree):
    res = np.ones(1)
    for d in range(1,degree+1):
        for k in range(d+1):
            temp= x1 ** (d-k) * x2 ** k
            res = np.hstack((res,temp))
    return res

# Plots scatter plot and decision boundary
# If using feature normalization, make sure to input normalized mat and X
def plotLogReg(mat,X,theta_min):
    neg_vals = np.zeros((1, mat.shape[1]))
    pos_vals = np.zeros((1, mat.shape[1]))
    for i in range(0, mat.shape[0]):
        if mat[i, mat.shape[1] - 1] == 0:
            neg_vals = np.append(neg_vals, [mat[i]], axis=0)
        elif mat[i, mat.shape[1] - 1] == 1:
            pos_vals = np.append(pos_vals, [mat[i]], axis=0)
    neg_vals = np.delete(neg_vals, 0, axis=0)
    pos_vals = np.delete(pos_vals, 0, axis=0)

    plt.scatter(neg_vals[:, 1], neg_vals[:, 2], s=10, c='r', marker='o', label='Negative')
    plt.scatter(pos_vals[:, 1], pos_vals[:, 2], s=20, c='g', marker='x', label='Positive')

    if theta_min.shape[0] == 3:
        x_db = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
        y_db = -(theta_min[0] + theta_min[1] * x_db) / theta_min[2]
        plt.plot(x_db, y_db, 'k')
    elif theta_min.shape[0] > 3:
        u = np.linspace(np.min(X[:, 1]) - 0.25, np.max(X[:, 1]) + 0.25, 50)
        v = np.linspace(np.min(X[:, 1]) - 0.25, np.max(X[:, 1]) + 0.25, 50)
        z = np.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = mapFeat(u[i], v[j], 6) @ theta_min
        plt.contour(u, v, z.T, 0)

    plt.style.use('seaborn-whitegrid')
    plt.xlabel('x_1')
    plt.ylabel('x_2')
    plt.title('')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
