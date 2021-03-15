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
from scipy.io import savemat
import collections
import shutil
import networkFiles as NF
import torchvision
import torchvision.transforms as transforms
import TensorGenerator


##################################################################
# Parameters:
# 		* data_file: Path to dataset 
#		* out_file: Path to save results
#		* use_gpu: Boolean flag for using GPU
##################################################################



# Layers is set by hiddenPlanted, which specifies the number of hidden units
#	to use in the data generation.  layers = # of hidden layers, so layers = 1
#	would be two QL layers


parser = argparse.ArgumentParser()
parser.add_argument('--data_gen', default='gaussian', type=str, metavar='PATH',
					help='method for generating outputs')
# parser.add_argument('--input_dist', default='guassian', type=str, metavar='PATH',
# 					help='sets the input distribution')
parser.add_argument('--out_file', default='Results/test.pth.tar', type=str, metavar='PATH',
					help='path to save result block')
parser.add_argument('--use_gpu', action='store_true')



def main(args):
	# Load in arguments
	data_gen = args.data_gen
	#input_dist = args.input_dist
	out_file = args.out_file

	print(data_gen)


	nsamp = 1000 # Sample number
	ntrial = 10
	lower_hidden = 4
	upper_hidden = 100 # Upper bound on the
	input_dim = 9 # Input dimension
	output_dim = 1
	



	training_epochs = 25000
	batch = nsamp
	learning_start = 1e-3

	# Set up arrays to store results
	residual_nn = np.zeros((upper_hidden, ntrial))
	residual_lsq = np.zeros((upper_hidden, ntrial))




	dtype = torch.FloatTensor
	if args.use_gpu:
		print('GPU is used')
		dtype = torch.cuda.FloatTensor


	############################################
	# Step 1: generate A's and training examples
	#
	############################################


	if data_gen == 'gaussian':
		print('Generation using Gaussian tensor')
		trainFeatures = np.random.normal(size = (nsamp, input_dim))/(np.sqrt(input_dim))
		trainFeatures[:, 0] = 1
		trainFeaturesTorch = torch.from_numpy(trainFeatures).type(dtype)

		X = Veronese(trainFeaturesTorch,input_dim,4,nsamp).numpy()

		A = np.random.normal(size = (input_dim**4, 1))

		trainLabels = X.dot(A)

		trainLabelsTorch = torch.from_numpy(trainLabels).type(dtype)

		trainLoader = [trainLabelsTorch]

		m, c, r, s  = np.linalg.lstsq(X,trainLabels,rcond = None)
		relError = ((trainLabels-np.matmul(X,m))**2).mean()/(np.transpose(trainLabels)**2).mean()

		print(relError)


	elif data_gen == 'random':
		print('Generation using random data')
		y, x, relError = TensorGenerator.generate_random(d = input_dim, r = 20, NumSamples = nsamp, k = 4)

		print(relError)

		y = y - np.mean(y)

		trainLabels = np.transpose(y)
		trainFeatures = np.transpose(x)

		trainFeaturesTorch = torch.from_numpy(trainFeatures).type(dtype)
		trainLabelsTorch = torch.from_numpy(trainLabels).type(dtype)

		trainLoader = [trainLabelsTorch]



	# # Calculate the least squares solution
	# z = torch.from_numpy(x.transpose())
	# X = Veronese(z,input_dim,4,nsamp).numpy()

	# X_other = Veronese(trainFeaturesTorch,input_dim,4,nsamp).cpu().numpy()

	# #print(np.linalg.norm(X - X_other))

	# m, c, r, s  = np.linalg.lstsq(X,trainLabels,rcond = None)
	# rel_error = ((np.transpose(y)-np.matmul(X,m))**2).mean()/(np.transpose(y)**2).mean()

	# Save out trial metadata
	results = {}
	results['Meta'] = {}

	results['Meta']['Sample Number'] = nsamp 
	results['Meta']['Lower Hidden'] = upper_hidden 
	results['Meta']['Upper Hidden'] = upper_hidden 
	results['Meta']['Input Dimension'] = input_dim 

	results['Meta']['Features'] = trainFeaturesTorch
	results['Meta']['Labels'] = trainLabelsTorch
	results['Meta']['lsq_residual'] = relError

	torch.save(results, out_file)
	

	for trial in range(ntrial):

		results[trial] = {}

		for hidden_iter in range(lower_hidden, upper_hidden, 5):

			hidden = hidden_iter + 1

			print('%i Hidden Units, Trial %i \n' % (hidden, trial))
			results[trial][hidden_iter] = {}


			layers = np.array(hidden)



			#######################################################################################
			# Train network with orthogonal regularization
			print('Regular QNN (Orthogonal) \n')

			learning = learning_start
			model = NF.QuadraticNet2(input_dim, hidden, output_dim)
			model.type(dtype)
			loss_fn = nn.MSELoss()
			optimizer = optim.Adam(model.parameters(), lr = learning)

			epoch_trace = []
			residual_trace = []

			results[trial][hidden_iter]['QuadraticOrth'] = {}

			I_out = torch.eye(hidden).type(dtype)
			I_1 = torch.eye(input_dim**2).type(dtype)


			eta = 10000 #torch.norm(trainLabelsTorch)**2

			# Set to the initialization specified by the paper
			#print(model.Layer2_linear.weight.data.size())
			# model.Layer1.weight.data = torch.eye(input_dim*hidden, input_dim).type(dtype)
			# model.Layer2.weight.data = torch.eye(hidden, hidden).type(dtype)
			# model.Layer2_linear.weight.data = torch.zeros(1, hidden).type(dtype)


			best = 1.0
			

			for epoch in range(training_epochs):
				model.train()
				for y in trainLoader:
					output, aWeights = model(trainFeaturesTorch, dtype)

					# # Form big matrix of A's
					# aWeights = torch.zeros((hidden, input_dim**2)).type(dtype)


					# for q in range(hidden):
					# 	temp = torch.mm(model.Layer1.weight.t(), torch.diag(model.Layer1_linear.weight[q, :]))
					# 	temp2 = torch.mm(temp, model.Layer1.weight)
					# 	aWeights[q, :] = temp2.reshape([-1])


					loss = ((loss_fn(output, y)) + eta*loss_fn(torch.mm(model.Layer2.weight.t(), model.Layer2.weight), I_out) +
						eta*loss_fn(torch.mm(aWeights.t(), aWeights), I_1))
					# Run the model backward and take a step using the optimizer.
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()


				if (epoch % 2000 == 0):


					model.eval()
					output, aWeights = model(trainFeaturesTorch, dtype)
					output = output.detach().cpu().numpy()
					residual_all = np.sum((trainLabels - output)**2)

					residual_all_nn = np.asscalar(residual_all*(1/(np.linalg.norm(trainLabels)**2)))

					if (residual_all_nn < best):
						best = residual_all_nn


					print('Epoch %i: Current residual: %.6f' % (epoch, residual_all_nn))
					epoch_trace = np.append(epoch_trace, epoch)
					residual_trace = np.append(residual_trace, residual_all_nn)

				adjustLearningRate(optimizer, epoch, learning)



			model.eval()

			output, aWeights = model(trainFeaturesTorch, dtype)
			output = output.detach().cpu().numpy()
			residual_all = np.sum((trainLabels - output)**2)

			residual_all_nn = np.asscalar(residual_all*(1/(np.linalg.norm(trainLabels)**2)))		

			print('[Trial: %i] Residual NN: %.6f' % (trial, residual_all_nn))

			trace = np.vstack((epoch_trace, residual_trace))

			#results[trial][hidden_iter]['QuadraticOrth']['Model'] = model.state_dict()
			results[trial][hidden_iter]['QuadraticOrth']['Residual'] = residual_all_nn
			results[trial][hidden_iter]['QuadraticOrth']['Trace'] = trace
			results[trial][hidden_iter]['QuadraticOrth']['Best'] = best


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

def adjustLearningRate(optimizer, epoch, learning):
	lr = learning * (0.5 ** (epoch // 20000))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)





