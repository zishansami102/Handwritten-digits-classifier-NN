import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle

def sigmoid(a):
	return 1/(1 + np.exp(-a))

def sigmGrad(a):
	return sigmoid(a)*(1-sigmoid(a))

#	getting paramater's gradient
def getgrad(X, y, theta1, theta2, regconst):
		(n, m) = X.shape
		a1 = np.vstack((np.ones((1,m)),X))
		z2 = np.dot(theta1.T, a1)
		a2 = np.vstack((np.ones((1,m)),sigmoid(z2)))
		a3 = sigmoid(np.dot(theta2.T, a2))
		theta1F = theta1[1:,:]
		theta2F = theta2[1:,:]

		cost = np.sum(np.sum( -y*np.log(a3) - (1-y)*np.log(1-a3) ))/m + (regconst/(2*m))*(sum(sum(theta1F*theta1F))+sum(sum(theta2F*theta2F)))
		d3 = a3 - y
		d2 = (theta2F.dot(d3))*sigmGrad(z2)

		tri2 = d3.dot(a2.T)
		tri1 = d2.dot(a1.T)
		theta2Grad = tri2.T/m + (regconst/m)*np.vstack((np.zeros((1,l2)),theta2F))
		theta1Grad = tri1.T/m + (regconst/m)*np.vstack((np.zeros((1,l1)),theta1F))

		return [theta1Grad, theta2Grad, cost]

# gradient Descent
def gradDescent(X, y, theta1, theta2, regconst, num_iter, alpha):	#slowest optimizer among others (#CS231n)
	cost = np.zeros(num_iter)
	for i in range(0, num_iter):
		[theta1Grad, theta2Grad, cost[i]] = getgrad(X, y, theta1, theta2, regconst)
		theta1 = theta1 - alpha*theta1Grad
		theta2 = theta2 - alpha*theta2Grad
		# printing the current status
		if (i+1)%(num_iter*0.1) == 0:
			per = float(i+1)/num_iter*100
			print(str(per)+"% Completed, Cost:"+str(cost[i]))
	return [theta1, theta2, cost]

#	Momentum Grad Descent (#CS231n)
def MgradDescent(X, y, theta1, theta2, regconst, num_iter, alpha, mu):
	cost = np.zeros(num_iter)
	v1 = 0
	v2 = 0
	for i in range(0, num_iter):
		[theta1Grad, theta2Grad, cost[i]] = getgrad(X, y, theta1, theta2, regconst)
		v1 = mu*v1 -  alpha*theta1Grad
		v2 = mu*v2 -  alpha*theta2Grad
		theta1 += v1
		theta2 += v2
		# printing the current status
		if (i+1)%(num_iter*0.1) == 0:
			per = float(i+1)/num_iter*100
			print(str(per)+"% Completed, Cost:"+str(cost[i]))
	return [theta1, theta2, cost]


def predict(X, theta1, theta2):	
	(n, m) = X.shape
	a1 = np.vstack((np.ones((1,m)),X))
	z2 = np.dot(theta1.T, a1)
	a2 = np.vstack((np.ones((1,m)),sigmoid(z2)))
	a3 = sigmoid(np.dot(theta2.T, a2))
	return a3

def initialize(L_in, L_out):
	epsilon = 0.25
	return np.random.random((L_in+1, L_out))*2*epsilon - epsilon
	# return np.random.random((L_in+1, L_out))/np.sqrt(L_in+1)	#Xavier Initialization CS231n lec 5


##############################################################################################################################
################################################## ------ START HERE --------  ###############################################
########################################### ----------  NEURAL NETWORK ---------------  ######################################
################# ----ARCHITECTURE PROPOSED : [INPUT - HIDDEN1 - SIGMOID - OUTPUT - SIGMOID]---- #############################
##############################################################################################################################


l1 = 50		# number of neurons in hidden layer
l2 = 10		# number of neurons in outer layer


mat = sio.loadmat('data.mat')
X_dash = mat['X']	# m * n
y_dash = mat['y']	# m * 1

X = X_dash.T	# n * m
tempY = y_dash.T	# 1 * m
(n, m) = X.shape

tempY = tempY*(tempY!=10)	# changin value with 10 back to 0.

y = np.zeros((l2,m))	# l2 * m
for i in range(0,m):
	y[tempY[0,i], i]= 1


alpha = 1.2	#learning rate (OPTIMIZED)
regconst = 1.3	#regularization constant 	(OPTIMIZED)
num_iter = 500	# number of times gradient descent to be return

initial_theta1 =  initialize(n, l1)
initial_theta2 = initialize(l1, l2)


# out = gradDescent(X, y , initial_theta1, initial_theta2, regconst, num_iter, alpha)
out = MgradDescent(X, y , initial_theta1, initial_theta2, regconst, num_iter, alpha, 0.5)

with open('output.pickle', 'wb') as f:
	pickle.dump(out, f)


# pickle_in = open('output.pickle', 'rb')
# out = pickle.load(pickle_in)

[theta1, theta2, cost] = out
a3 = predict(X, theta1, theta2)
# print(a3[:,0:4])
h = np.empty((1,m))
for i in range(0,m):
	h[0,i] = np.argmax(a3[:,i])



acc = np.mean((h==tempY)*np.ones(h.shape))*100
print("With Alpha:"+str(alpha)+", Regularization Const:"+str(regconst)+", Num Iteration:"+str(num_iter))
print("Accuracy:"+str(acc))

plt.plot(cost)
plt.ylabel('cost')
plt.xlabel('iteration')
plt.show()

















