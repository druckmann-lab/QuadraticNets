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
# 		* load_path: Path to dataset 
#		* out_file: Path to save results
#		* use_gpu: Boolean flag for using GPU
##################################################################


parser = argparse.ArgumentParser()
parser.add_argument('--load_number', default=1, type=int, metavar='PATH',
					help='sets which data to load')
parser.add_argument('--load_path', default='test', type=str, metavar='PATH',
					help='sets where to load data from')
parser.add_argument('--out_file', default='Results/test.pth.tar', type=str, metavar='PATH',
					help='path to save result block')
parser.add_argument('--use_gpu', action='store_true')



def main(args):

	# Load in arguments
	load_path = args.load_path
	load_number = args.load_number
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


	data = torch.load(load_path)

	print(data[0]['Features'].type())

	
	# Save out metadata about the experiments
	results = {}
	results['Meta'] = {}
	
	A = data[load_number]['A']
	trainFeaturesTorch = data[load_number]['Features']
	trainLabelsTorch = data[load_number]['Labels']
	m = data[load_number]['lsq_weights']
	rel_error = data[load_number]['lsq_residual']

	trainLoader = [trainLabelsTorch]
	trainLabels = trainLabelsTorch.cpu().numpy()

	print(trainFeaturesTorch.type())


	results['Meta']['A'] = A
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


			####### Solve with norm network
			print('QNN with Norm \n')
			results[trial][hidden_iter]['QuadraticNorm'] = {}
			model = NF.QuadraticNetNorm(input_dim, hidden, 1)
			model.type(dtype)

			loss_fn = nn.MSELoss()
			optimizer = optim.Adam(model.parameters(), lr = learning)

			epoch_trace = []
			residual_trace = []

			for epoch in range(training_epochs):
				#loader = DataLoader(trainset, batch_size=batch, shuffle=True)
				model.train()
				for y in trainLoader:
					# Run the model forward to compute scores and loss.
					output = model(trainFeaturesTorch, dtype).type(dtype)
					loss = loss_fn(output, y).type(dtype)

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

			trace = np.vstack((epoch_trace, residual_trace))

			model.eval()

			output = model(trainFeaturesTorch, dtype).detach().cpu().numpy()
			residual_all = np.sum((trainLabels - output)**2)

			residual_all_nn = np.asscalar(residual_all*(1/(np.linalg.norm(trainLabels)**2)))

			results[trial][hidden_iter]['QuadraticNorm']['Model'] = model.state_dict()
			results[trial][hidden_iter]['QuadraticNorm']['Residual'] = residual_all_nn
			results[trial][hidden_iter]['QuadraticNorm']['Trace'] = trace


			######## Solve with regular network
			print('Regular QNN \n')
			model = NF.QuadraticNet(input_dim, hidden, 1)
			model.type(dtype)

			loss_fn = nn.MSELoss()
			optimizer = optim.Adam(model.parameters(), lr = learning)

			epoch_trace = []
			residual_trace = []

			results[trial][hidden_iter]['Quadratic'] = {}


			for epoch in range(training_epochs):
				#loader = DataLoader(trainset, batch_size=batch, shuffle=True)
				model.train()
				for y in trainLoader:
					# Run the model forward to compute scores and loss.
					output = model(trainFeaturesTorch, dtype).type(dtype)
					loss = loss_fn(output, y).type(dtype)

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



			######## Solve with orthogonal regularization
			print('Orthogonal QNN \n')
			model = NF.QuadraticNet(input_dim, hidden, 1)
			model.type(dtype)

			eta = torch.norm(trainLabelsTorch)**2

			loss_fn = nn.MSELoss()
			optimizer = optim.Adam(model.parameters(), lr = learning)

			I = torch.eye(input_dim).type(dtype)

			epoch_trace = []
			residual_trace = []

			results[trial][hidden_iter]['QuadraticOrth'] = {}

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


			results[trial][hidden_iter]['QuadraticOrth']['Model'] = model.state_dict()
			results[trial][hidden_iter]['QuadraticOrth']['Residual'] = residual_all_nn
			results[trial][hidden_iter]['QuadraticOrth']['Trace'] = trace


			######## Solve with orthogonal regularization and initialization
			print('Orthogonal QNN with Initialization \n')
			model = NF.QuadraticNet(input_dim, hidden, 1)
			model.type(dtype)

			model.hiddenLayer.weight.data = torch.eye(hidden, 10).type(dtype)
			model.output.weight.data = torch.zeros(1, hidden).type(dtype)

			eta = torch.norm(trainLabelsTorch)**2

			loss_fn = nn.MSELoss()
			optimizer = optim.Adam(model.parameters(), lr = learning)

			I = torch.eye(input_dim).type(dtype)

			epoch_trace = []
			residual_trace = []

			results[trial][hidden_iter]['QuadraticOrthInit'] = {}

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


			results[trial][hidden_iter]['QuadraticOrthInit']['Model'] = model.state_dict()
			results[trial][hidden_iter]['QuadraticOrthInit']['Residual'] = residual_all_nn
			results[trial][hidden_iter]['QuadraticOrthInit']['Trace'] = trace


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





