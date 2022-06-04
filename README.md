# ML-Logistic-Regression
This code demonstrates the basics of machine learning by implementing algorithms for logistic regression of multiple variables. Given a training set of data with any number of features, it can learn parameters that fit any nonlinear function to the data and make predictions about discrete values. This is useful for predicting categories or groups that data belongs in, such as whether or not an NBA team will win their next game. While this code can work for any number of variables, logistic regression is best used when there are only two features because the data can be easily plotted, and because neural networks are better "in general" for data sets with a high number of features.

**Steps to implement logistic regression using this code:**
1. Obtain raw initial data from somewhere and put it into a csv file.
2. Modify the data by adding the x_0 feature. Also, feature normalize the data if necessary. Note that this code separates the initial X and feature normalized X into two distinct variables.
3. Choose a learning rate by plotting the speed of convergence of the error function with respect to the number of iterations.
4. Depending on the situation, apply a regularized or noon-regularized gradient descent algorithm to return the optimal parameters that minimizes the error function.
5. Plot logistic regression using the optimal parameters.
6. Check accuracy of program.

**Reference**
- xj_i is the ith element of the jth training example
- Training examples range from x1 to xm
- Each training example has elements ranging from x_0 to x_n+1
- Use numpy 2d arrays as preferred data structure whenever possible
- If learning rates are too large, J can diverge and blow up, resulting in really large values the computer cannot comprehend
- Csv files MUST include headers (x_0, x_1, ..., y) or else the first training ex will be cut off

**Data**
- Data Set 1: logReg1.csv, requires feat norrm, linear decision boundary
- Data Set 2: logReg2.csv, do not do feat norm, circular decision boundary
  - Data Set 2_1: logReg2_1.csv, added 25 more features

**Examples:**

**Testing different learning rates for data set 1**

![Figure_7](https://user-images.githubusercontent.com/106856325/172010002-4dee1c72-47e1-408a-9400-a0fd5cd6cb10.png)

**Plotting decision boundary to predict data set 1 using the learned parameters**

![Figure_6](https://user-images.githubusercontent.com/106856325/172008981-2cdf73f8-832b-4ca5-a511-6977dea2ce06.png)

**Accuracy for data set 1: 89.0%**

Testing different learning rates for data set 2_1

![Figure_8](https://user-images.githubusercontent.com/106856325/172012051-e6a11d9c-e437-485f-a550-8d605cae5035.png)

**Plotting decision boundary to predict data set 2_1 using the learned parameters**

![Figure_9](https://user-images.githubusercontent.com/106856325/172012526-531efa2b-3c7a-46a7-901d-26a6cc1b192a.png)

**Accuracy for data set 2_1: 82.20338983050848%**

**Conclusion: Logistic regression is pretty solid.**
