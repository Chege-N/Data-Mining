#!/usr/bin/env python
# coding: utf-8

# # Exercise 2: Logistic Regression
# 
# > In this exercise, you will implement logistic regression and apply it to two different datasets.

# ## 1. Logistic Regression
# 
# > In this part of the exercise, you will build a logistic regression model to predict whether a student gets admitted into a university.
# Suppose that you are the administrator of a university department and you want to determine each applicant’s chance of admission based on their results on two exams. You have historical data from previous applicants that you can use as a training set for logistic regression. For each training example, you have the applicant’s scores on two exams and the admissions decision.
# Your task is to build a classification model that estimates an applicant’s probability of admission based the scores from those two exams. 
# 
# ### 1.1 Visualizing the Data

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import pandas as pd
df = pd.read_csv('ex2data1.txt', sep=',', header=None)
df.columns = ['exam_score_1', 'exam_score_2', 'label']


# In[3]:


df.describe().T


# In[4]:


plt.figure(figsize=(7,5))
ax = sns.scatterplot(x='exam_score_1', y='exam_score_2', hue='label', data=df, style='label', s=80)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[1:], ['Not admitted', 'Admitted'])
plt.title('Scatter plot of training data')
plt.show(ax)


# ### 1.2 Implementation
# 
# #### 1.2.1 Sigmoid Function
# 
# Logistic regression hypothesis: 
# 
# $$h_\theta(x) = g(\theta^Tx)$$
# 
# $$g(z) = \frac{1}{1+e^{-z}}$$

# In[5]:


def sigmoid(z):
    z = np.array(z)
    return 1 / (1+np.exp(-z))


# Plot of sigmoid function:

# In[6]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
z = np.linspace(-10, 10, 100)
sig = sigmoid(z)
plt.figure(figsize=(9, 3))
plt.plot([-10, 10], [0, 0], "k-")
plt.plot([-10, 10], [0.5, 0.5], "k:")
plt.plot([-10, 10], [1, 1], "k:")
plt.plot([0, 0], [-1.1, 1.1], "k-")
plt.plot(z, sig, "b-", linewidth=2)
plt.xlabel("z")
plt.axis([-10, 10, -0.1, 1.1])
plt.show()


# #### 1.2.2 Cost Function and Gradient
# 
# Cost function in logistic regression is:
# 
# $$J(\theta) = -\frac{1}{m}\sum_{i=1}^m[y^i log(h_\theta(x^i))+(1-y^i)log(1-h_\theta(x^i))]$$
# 
# Vectorized implementation:
# 
# $h = g(X\theta)$
# 
# $J(\theta) = \frac{1}{m}(-y^T log(h)-(1-y)^Tlog(1-h))$
# 
# 
# 
# The gradient of the cost is a vector of the same length as $\theta$ where $j^{th}$ element (for $j=0,1,...,n$) is defined as follows:
# 
# $$\nabla J(\theta) = \frac{1}{m} \sum_{i=1}^m ((h_\theta(x^i) - y^i) \cdot x_j^i)$$
# 
# Vectorized:
# $\nabla J(\theta) = \frac{1}{m} \cdot X^T \cdot (g(X\theta)-y)$

# In[7]:


def cost_function(theta, X, y):
    m = y.shape[0]
    theta = theta[:, np.newaxis] #trick to make numpy minimize work
    h = sigmoid(X.dot(theta))
    J = (1/m) * (-y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h)))

    diff_hy = h - y
    grad = (1/m) * diff_hy.T.dot(X)

    return J, grad


# In[8]:


m = df.shape[0]
X = np.hstack((np.ones((m,1)),df[['exam_score_1', 'exam_score_2']].values))
y = np.array(df.label.values).reshape(-1,1)
initial_theta = np.zeros(shape=(X.shape[1]))


# In[9]:


cost, grad = cost_function(initial_theta, X, y)
print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros):')
print(grad.T)
print('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628')


# In[10]:


test_theta = np.array([-24, 0.2, 0.2])
[cost, grad] = cost_function(test_theta, X, y)

print('Cost at test theta:', cost)
print('Expected cost (approx): 0.218')
print('Gradient at test theta:')
print(grad.T)
print('Expected gradients (approx):\n 0.043\n 2.566\n 2.647')


# #### 1.2.3 Learning Parameters using an optimization solver
# "Conjugate gradient", "BFGS", and "L-BFGS" are more sophisticated, faster ways to optimize $\theta$ that can be used instead of gradient descent. 

# In[11]:


import scipy.optimize as opt
def optimize_theta(X, y, initial_theta):
    opt_results = opt.minimize(cost_function, initial_theta, args=(X, y), method='TNC',
                               jac=True, options={'maxiter':400})
    return opt_results['x'], opt_results['fun']


# In[12]:


opt_theta, cost = optimize_theta(X, y, initial_theta)


# In[13]:


print('Cost at theta found by fminunc:', cost)
print('Expected cost (approx): 0.203')
print('theta:\n', opt_theta.reshape(-1,1))
print('Expected theta (approx):')
print(' -25.161\n 0.206\n 0.201')


# #### Decision boundary

# In[14]:


plt.figure(figsize=(7,5))
ax = sns.scatterplot(x='exam_score_1', y='exam_score_2', hue='label', data=df, style='label', s=80)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[1:], ['Not admitted', 'Admitted'])
plt.title('Training data with decision boundary')

plot_x = np.array(ax.get_xlim())
plot_y = (-1/opt_theta[2]*(opt_theta[1]*plot_x + opt_theta[0]))
plt.plot(plot_x, plot_y, '-', c="green")
plt.show(ax)


# #### 1.2.4 Evaluating Logistic Regression

# Predict whether a particular student will be admitted

# In[15]:


prob = sigmoid(np.array([1, 45, 85]).dot(opt_theta))
print('For a student with scores 45 and 85, we predict an admission probability of', prob)
print('Expected value: 0.775 +/- 0.002');


# Accuracy on training set

# In[16]:


def predict(X, theta):
    y_pred = [1 if sigmoid(X[i, :].dot(theta)) >= 0.5 else 0 for i in range(0, X.shape[0])]
    return y_pred


# In[17]:


X = np.hstack((np.ones((m,1)),df[['exam_score_1', 'exam_score_2']].values))

y_pred_prob = predict(X, opt_theta)
f'Train accuracy: {np.mean(y_pred_prob == df.label.values) * 100}'


# #### 1.2.5 Equivalent code using Scikit-Learn:

# In[18]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver='newton-cg', max_iter=400)
log_reg.fit(df[['exam_score_1', 'exam_score_2']].values,
            df.label.values)


# In[19]:


log_reg.intercept_, log_reg.coef_


# Sklearn logistic regression accuracy:

# In[20]:


log_reg.score(df[['exam_score_1', 'exam_score_2']].values,
              df.label.values)


# ## 2. Regularized Logistic Regression
# 
# > In this part of the exercise, you will implement regularized logistic regression to predict whether microchips from a fabrication plant passes quality assur- ance (QA). During QA, each microchip goes through various tests to ensure it is functioning correctly.
# Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests. From these two tests, you would like to determine whether the microchips should be accepted or rejected. To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.
# 
# ### 2.1 Visualizing the Data

# In[22]:


df2 = pd.read_csv('ex2data2.txt', sep=',', header=None)
df2.columns = ['test_1', 'test_2', 'label']


# In[23]:


df2.describe().T


# In[24]:


plt.figure(figsize=(7,5))
ax = sns.scatterplot(x='test_1', y='test_2', hue='label', data=df2, style='label', s=80)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[1:], ['accepted', 'rejected'])
plt.title('Scatter plot of training data')
plt.show(ax)


# ### 2.2 Feature Mapping
# 
# > One way to fit the data better is to create more features from each data point. We will map the features into all polynomial terms of $x_1$ and $x_2$ up to the sixth power. As a result of this mapping, our vector of two features (the scores on two QA tests) has been transformed into a 28-dimensional vector. A logistic regression classifier trained on this higher-dimension feature vector will have a more complex decision boundary and will appear nonlinear when drawn in our 2-dimensional plot.

# In[25]:


def map_feature(X1, X2, degree):
    X1 = np.array(X1).reshape(-1,1)
    X2 = np.array(X2).reshape(-1,1)
    
    out = np.ones((X1.shape[0], 1))
    for i in range(1, degree+1):
        for j in range(0, i+1):
            p = (X1**(i-j)) * (X2**j)
            out = np.append(out, p, axis=1)
    return out


# In[26]:


X_p = map_feature(df2.test_1.values, df2.test_2.values, 6)
X_p.shape


# ### 2.3 Cost Function and Gradient
# 
# Regularized cost function in logistic regression:
# 
# $$J(\theta) = -\frac{1}{m}\sum_{i=1}^m[y^i log(h_\theta(x^i))+(1-y^i)log(1-h_\theta(x^i))] + \frac{\lambda}{2m} \sum_{j=1}^n\theta_j^2$$
# 
# Gradient:
# 
# $$\frac{\partial J(\theta)}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)} - y^{(i)})\cdot x_j^{(i)} \ \text{for j=0}$$
# 
# $$\frac{\partial J(\theta)}{\partial \theta_0} = (\frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)} - y^{(i)})\cdot x_j^{(i)}) + \frac{\lambda}{m}\theta_j \ \text{for j$\ge$1}$$

# In[27]:


def cost_function_reg(theta, X, y, lambda_reg):
    m = y.shape[0]
    theta = theta[:, np.newaxis] 
    h = sigmoid(X.dot(theta))
    J = (1/m) * (-y.T.dot(np.log(h)) - (1-y).T.dot(np.log(1-h))) + (lambda_reg/(2*m)) * np.sum(theta[1:]**2)

    diff_hy = h - y
    grad = (1/m) * diff_hy.T.dot(X) + ((lambda_reg/m) * theta.T)
    grad[0, 0] = (1/m) * diff_hy.T.dot(X[:, 0])

    return J, grad


# #### 2.3.1 Learning Parameters

# In[28]:


import scipy.optimize as opt
def optimize_theta_reg(X, y, initial_theta, lambda_reg):
    opt_results = opt.minimize(cost_function_reg, initial_theta, args=(X, y, lambda_reg), method='TNC', jac=True, options={'maxiter':400})
    return opt_results['x'], opt_results['fun']


# In[29]:


m = df.shape[0]
X = X_p
y = np.array(df2.label.values).reshape(-1,1)
initial_theta = np.zeros(shape=(X.shape[1]))


# In[30]:


lambda_reg = 1
cost, grad = cost_function_reg(initial_theta, X, y, lambda_reg)
print('Cost at initial theta (zeros):', cost)
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - top 5:')
print(grad.T[:5])
print('Expected gradients top 5(approx):\n 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115')


# In[31]:


lambda_reg = 10
initial_theta = np.ones(shape=(X.shape[1]))
cost, grad = cost_function_reg(initial_theta, X, y, lambda_reg)
print('Cost at initial theta:', cost)
print('Expected cost (approx): 3.16')
print('Gradient at theta - top 5:')
print(grad.T[:5])
print('Expected gradients top 5(approx):\n 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922')


# ### 2.4 Plotting the Decision Boundary

# In[32]:


lambda_reg = [1, 10, 100, 0]
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15,4))
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)

for il, l in enumerate(lambda_reg):
    theta_opt, cost = optimize_theta_reg(X, y, initial_theta, l)
    z = np.zeros((u.shape[0], v.shape[0]))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = map_feature(u[i], v[j], 6).dot(theta_opt)
    
    sns.scatterplot(x='test_1', y='test_2', hue='label', data=df2, style='label', s=80, ax=axs[il])
    
    axs[il].contour(u, v, z.T, levels=[0], colors='green')
    axs[il].set_title('$\lambda={}$'.format(l))
fig.tight_layout()
plt.show()


# ### 2.5 Accuracy on Training Set

# In[33]:


lambda_reg = 1
theta, cost = optimize_theta_reg(X, y, initial_theta, lambda_reg)
theta


# In[34]:


y_pred_prob = predict(X, theta)
f'Train accuracy: {np.mean(y_pred_prob == df2.label.values) * 100}'


# ### 2.6 Equivalent Code using Scikit-Learn:

# In[35]:


from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(solver='newton-cg', max_iter=400)
log_reg.fit(X[:,1:], df2.label.values)


# In[36]:


log_reg.intercept_, log_reg.coef_


# In[37]:


log_reg.score(X[:,1:], df2.label.values)


# In[ ]:




