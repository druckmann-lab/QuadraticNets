import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable 
import copy
import numpy as np
import os
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
#import matplotlib.pyplot as plt
import math
import time
#from scipy.io import savemat
import collections
import shutil
import networkFiles as NF
import torchvision
import torchvision.transforms as transforms


##################################################################
# Parameters:
# 		* input_dist: Sets distribution of inputs {'gaussian', 'non-planted', 'natural_scences'}
# 		* data_gen: Specifies method for generating outputs
#		* out_file: Path to save results
#		* use_gpu: Boolean flag for using GPU
##################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--data_gen', default='identity', type=str, metavar='PATH',
					help='method for generating outpus')
parser.add_argument('--input_dist', default='guassian', type=str, metavar='PATH',
					help='sets the input distribution')
parser.add_argument('--out_file', default='Results/test.pth.tar', type=str, metavar='PATH',
					help='path to save result block')
parser.add_argument('--use_gpu', action='store_true')



def main(args):
	##################################################################
	# Generate and train a one layer quadratic NN
	# Goal is to see if GD finds global minimum
	# 
	# 		
	##################################################################

	# Load in arguments
	data_gen = args.data_gen
	input_dist = args.input_dist
	out_file = args.out_file


	nsamp = 1500 # Sample number
	ntrial = 20
	lower_hidden = 0
	upper_hidden = 20 # Upper bound on the
	input_dim = 10 # Input dimension


	training_epochs = 30000
	training_epochs_lsq = 700
	batch = 2000
	learning = 1e-3


	dtype = torch.FloatTensor
	if args.use_gpu:
		print('GPU is used')
		dtype = torch.cuda.FloatTensor


	if ((input_dist=='natural_scenes') or (input_dist=='non_planted_natural')):
		print('Using natural scenes')
		# Import cifar
		transform = transforms.Compose(
		[transforms.ToTensor(),
		 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

		cifarset = torchvision.datasets.CIFAR10(root='./data', train=True,
												download=True, transform=transform)
		cifarloader = torch.utils.data.DataLoader(cifarset, batch_size=5,
												  shuffle=True)

	
	# Save out metadata about the experiments
	results = {}
	results['Meta'] = {}
	results['Meta']['Data Generation'] = data_gen 
	results['Meta']['Input Distribution'] = input_dist 

	results['Meta']['Sample Number'] = nsamp 
	results['Meta']['Lower Hidden'] = upper_hidden 
	results['Meta']['Upper Hidden'] = upper_hidden 
	results['Meta']['Input Dimension'] = input_dim 
	############################################
	# Step 1: generate A's and training examples
	#
	############################################

	# Set the data_generation
	if (data_gen == 'identity'):
		print('Generating data with identity matrix')
		W = np.eye(input_dim)
		v = np.ones((1, input_dim))
	elif(data_gen == 'identity_mixed_signs'):
		print('Generating data with identity matrix, mixed signs')
		W = np.eye(input_dim)
		v = np.random.binomial(1, 0.5, (1, input_dim))		
		v[v==0] = -1
	elif(data_gen == 'gaussian_matrix'):
		print('Generating data with random Gaussian matrix')
		W = np.random.normal(size = (input_dim, input_dim))/np.sqrt(input_dim)
		v = np.random.binomial(1, 0.5, (1, input_dim))		
		v[v==0] = -1
	elif(data_gen == 'identity_lowrank'):
		print('Generating data with low-rank identity matrix')
		Id = np.eye(input_dim)
		W = Id[0:rank, :]
		v = np.ones((1, rank))
	elif(data_gen == 'identity_lowrank_mixed_signs'):
		print('Generating data with low-rank identity matrix, mixed signs')
		Id = np.eye(input_dim)
		W = Id[0:rank, :]
		v = np.random.binomial(1, 0.5, (1, rank))		
		v[v==0] = -1
	elif(data_gen == 'gaussian_lowrank'):
		print('Generating data with random Gaussian matrix')
		W = np.random.normal(size = (rank, input_dim))/np.sqrt(input_dim)
		v = np.random.binomial(1, 0.5, (1, rank))		
		v[v==0] = -1
	else:
		print('The specified data generation method not recognized; using identity.')
		W = np.eye(input_dim)
		v = np.ones((1, input_dim))

	print(W)
	print(v)




	trainFeatures = np.zeros((nsamp, input_dim))
	trainLabels = np.zeros((nsamp, 1))

	
	for j in range(nsamp):

		if (input_dist == 'gaussian'):
			x = np.random.normal(size = (input_dim, 1))/(np.sqrt(input_dim))
			x[0] = 1
			y = v.dot(np.square(W.dot(x)))
		elif (input_dist == 'non_planted'):
			x = np.random.normal(size = (input_dim, 1))/(np.sqrt(input_dim))
			x[0] = 1
			y = 2*np.random.normal(size = (1))
		elif (input_dist == 'natural_scenes'):
			data, labels = iter(cifarloader).next()

			image = np.random.choice(5, input_dim)
			i_pixel = np.random.choice(32, input_dim)
			j_pixel = np.random.choice(32, input_dim)

			x = data[image, 1, i_pixel, j_pixel]
			
			y = v.dot(np.square(W.dot(x)))
		elif (input_dist == 'non_planted_natural'):
			data, labels = iter(cifarloader).next()

			image = np.random.choice(5, input_dim)
			i_pixel = np.random.choice(32, input_dim)
			j_pixel = np.random.choice(32, input_dim)

			x = data[image, 1, i_pixel, j_pixel]

			y = 2*np.random.normal(size = (1))
		else:
			x = np.random.normal(size = (input_dim, 1))
			y = v.dot(np.square(W.dot(x)))


		trainFeatures[j, :] = np.transpose(x)
		trainLabels[j, :] = y
		


	trainFeaturesTorch = torch.from_numpy(trainFeatures).type(dtype)
	trainLabelsTorch = torch.from_numpy(trainLabels).type(dtype)

	trainLoader = [trainLabelsTorch]

	# Solve the least squares problem via the QR factorization
	X = Veronese(trainFeaturesTorch,input_dim,2,nsamp).numpy()

	

	m, c, r, s  = np.linalg.lstsq(X,trainLabels,rcond = None)
	rel_error = ((trainLabels-np.matmul(X,m))**2).mean()/(np.transpose(trainLabels)**2).mean()
	# Q, R = torch.qr(tensorFeaturesTorch)
	# temp = torch.mm(Q.t(), trainLabelsTorch)

	# # beta is the least squares solution
	# beta = torch.mm(torch.inverse(R), temp)
	# #q, _ = torch.lstsq(trainLabelsTorch, tensorFeaturesTorch)
	# #beta = q[0:100, :]

	# print((torch.mm(tensorFeaturesTorch, beta) - trainLabelsTorch)**2)

	# residual_all = ((torch.mm(tensorFeaturesTorch, beta) - trainLabelsTorch)**2).mean()
	# residual_all_lsq = residual_all*(1/(trainLabelsTorch**2).mean())


	results['Meta']['Q'] = W
	results['Meta']['lambda'] = v
	results['Meta']['Features'] = trainFeaturesTorch
	results['Meta']['Labels'] = trainLabelsTorch
	results['Meta']['lsq_weights'] = m
	results['Meta']['lsq_residual'] = rel_error


	for trial in range(ntrial):

		print(rel_error)
		results[trial] = {}

		############################################
		# Set up the quadratic network in Pytorch and train on the data set using GD

		for hidden_iter in range(lower_hidden, upper_hidden):

			hidden = hidden_iter + 1

			print('Trial %i: %i Hidden Units \n' % (trial, hidden))

			results[trial][hidden_iter] = {}



			######## Solve with regular network
			print('Regular QNN \n')
			model = NF.QuadraticNet(input_dim, hidden, 1)
			model.type(dtype)

			eta = torch.norm(trainLabelsTorch)**2

			loss_fn = nn.MSELoss()
			optimizer = optim.Adam(model.parameters(), lr = learning)

			I = torch.eye(input_dim).type(dtype)

			epoch_trace = []
			residual_trace = []

			results[trial][hidden_iter]['Quadratic'] = {}

			for epoch in range(training_epochs):
				#loader = DataLoader(trainset, batch_size=batch, shuffle=True)
				model.train()
				for y in trainLoader:
					# Run the model forward to compute scores and loss.
					output = model(trainFeaturesTorch, dtype).type(dtype)
					loss = loss_fn(output, y) + eta*loss_fn(torch.mm(model.hiddenLayer.weight.t(), model.hiddenLayer.weight), I)

					# Run the model backward and take a step using the optimizer.
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()


				if (epoch % 2000 == 0):

					model.eval()

						
					output = model(trainFeaturesTorch, dtype).detach().cpu().numpy()
					residual_all = np.sum((trainLabels - output)**2)

					residual_all_nn = np.asscalar(residual_all*(1/(np.linalg.norm(trainLabels)**2)))


					print('Epoch %i: Current residual: %.6f' % (epoch, residual_all_nn))
					epoch_trace = np.append(epoch_trace, epoch)
					residual_trace = np.append(residual_trace, residual_all_nn)



			model.eval()

			output = model(trainFeaturesTorch, dtype).detach().cpu().numpy()
			residual_all = np.sum((trainLabels - output)**2)

			residual_all_nn = np.asscalar(residual_all*(1/(np.linalg.norm(trainLabels)**2)))

			trace = np.vstack((epoch_trace, residual_trace))


			results[trial][hidden_iter]['Quadratic']['Model'] = model.state_dict()
			results[trial][hidden_iter]['Quadratic']['Residual'] = residual_all_nn
			results[trial][hidden_iter]['Quadratic']['Trace'] = trace


		torch.save(results, out_file)



def Veronese(x,d,k,T):
	# This makes an unrolled version of the kth order symetric tensor generated by x
	#	d: input dimension (dimension of x)
	#	k: order of tensor
	#	T: number of samples

		X = torch.Tensor(T,d**k)
		for t in range(T):
			temp = x[t,:].squeeze()
			len = d
			# print(temp)
			for i in range(k-1):
				len = len*d
				temp = torch.ger(temp,x[t,:].squeeze()).view(len)
				# print(temp)
			X[t,:] = temp
		return X


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)





