from logReg import *

################################################## INPUT ##################################################

# Iterations
num_iters = 1500

# Learning rate
alpha = 0.3

# Lambda
lamb = 1

# Read data
df = pd.read_csv('logReg2_1.csv')

################################################## DEFINE ##################################################

# Convert the appropriate data in dataframe to numpy 2d array
mat = df.loc[:,:].to_numpy()
mat = mat.astype(float)

'''
# Raw data does not have a predefined x_0 (which always equals one). The following can be used to create a new 
# text file that adds an x_0 column to the data.
mat = np.insert(mat, 0, 1, axis=1)
np.savetxt("logreg1.csv", mat, delimiter=",")
'''

# X is a matrix of m training examples (each row) and n+1 features (each column)
X = getX(mat)
m = X.shape[0] # Number of training examples
n = X.shape[1] # Number of features (inc. x_0)

# Initial guess for theta (num of entries must equal num of features (inc. x_0) )
theta = np.zeros((X.shape[1],1))

# Feature normalize values if necessary (it usually is)
X_norm = featureNorm(X)

# y is a column vector of all outputs y1 to ym, corresponding with each training example
y = getY(mat)

'''
# Adding more features to X allows for more fitting flexibility of the decision boundary. The following can be
# used to create new txt files of the same original data, but with new features (eg x_1^2, x_1*x_2, x_2^2)
X_feat = addFeatures(X,6)
mat_feat = np.append(X_feat, y, axis=1)
np.savetxt("logreg2_1.csv", mat_feat, delimiter=",")
'''

# Create a normalized version of mat (useful for plotting data)
mat_norm = np.copy(mat)
mat_norm[:,0:mat_norm.shape[1]-1] = np.copy(X_norm)

################################################## EXECUTE ##################################################
# Find theta such that error is minimized using (regularized) gradient descent
#theta_min = gradDesc(X_norm, y, theta, alpha, num_iters, returnJ=False)
theta_min = regGradDesc(X, y, theta, alpha, lamb, num_iters, returnJ=False)
print('Theta using gradient descent: ')
print(theta_min)

# Compute accuracy of regression
acc = computeAccuracy(X,y,theta_min)
print(f'Accuracy: {acc*100}%')

# Return cost for each iteration (should decrease over time)
#J001 = gradDesc(X_norm,y,theta,alpha=0.01,num_iters=num_iters, returnJ=True)
#J001 = regGradDesc(X,y,theta,alpha=0.01,lamb=1,num_iters=num_iters, returnJ=True)
#J003 = gradDesc(X_norm,y,theta,alpha=0.03,num_iters=num_iters, returnJ=True)
#J003 = regGradDesc(X,y,theta,alpha=0.03,lamb=1,num_iters=num_iters, returnJ=True)
#J01 = gradDesc(X_norm,y,theta,alpha=0.1,num_iters=num_iters, returnJ=True)
#J01 = regGradDesc(X,y,theta,alpha=0.1,lamb=1,num_iters=num_iters, returnJ=True)
#J03 = gradDesc(X_norm,y,theta,alpha=0.3,num_iters=num_iters, returnJ=True)
#J03 = regGradDesc(X,y,theta,alpha=0.3,lamb=1,num_iters=num_iters, returnJ=True)

################################################## PLOT ##################################################
# Uncomment to plot logistic regression
plotLogReg(mat,X,theta_min)

# Uncomment to plot speed of convergence
'''
x = np.arange(0, num_iters + 1)
plt.plot(x,J001, label='0.01')
plt.plot(x,J003, label='0.03')
plt.plot(x,J01, label='0.1')
plt.plot(x,J03, label='0.3')

plt.style.use('seaborn-whitegrid')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.title('Speed of convergence')
plt.xlim(0, num_iters + 1)
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
'''
