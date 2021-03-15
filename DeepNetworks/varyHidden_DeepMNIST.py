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


####### Notes
# Layers is set by hiddenPlanted, which specifies the number of hidden units
#	to use in the data generation.  layers = # of hidden layers, so layers = 1
#	would be two QL layers


parser = argparse.ArgumentParser()
parser.add_argument('--data_file', default='ExpData/dataset47.pth.tar', type=str, metavar='PATH',
					help='path to save result block')
parser.add_argument('--out_file', default='Results/test.pth.tar', type=str, metavar='PATH',
					help='path to save result block')
parser.add_argument('--use_gpu', action='store_true')



def main(args):
	# Load in arguments
	data_file = args.data_file
	out_file = args.out_file



	#nsamp = 1000 # Sample number
	ntrial = 10
	lower_hidden = 4
	upper_hidden = 125 # Upper bound on the
	input_dim = 11 # Input dimension
	output_dim = 1
	



	training_epochs = 20000
	
	learning_start = 1e-3




	dtype = torch.FloatTensor
	if args.use_gpu:
		print('GPU is used')
		dtype = torch.cuda.FloatTensor


	### Load in MNIST data 
	dataset = torch.load(data_file)
	trainFeaturesTorch = dataset['TrainFeatures']
	trainLabelsTorch = dataset['TrainLabels']
	nsamp = trainFeaturesTorch.size(0)
	batch = nsamp

	### Normalize the training data and concatenate with 1
	#trainFeaturesTorch = trainFeaturesTorch/torch.max(torch.norm(trainFeaturesTorch, dim = 1))
	trainFeaturesTorch = trainFeaturesTorch[:, 0:input_dim-1]
	trainFeaturesTorch = (trainFeaturesTorch.t()/(torch.norm(trainFeaturesTorch, dim = 1))).t()
	trainFeaturesTorch = torch.cat((torch.ones(nsamp, 1).type(torch.DoubleTensor), trainFeaturesTorch), dim = 1)
	#trainFeaturesTorch = trainFeaturesTorch[:, 0:input_dim]

	testFeaturesTorch = dataset['TestFeatures']
	testLabelsTorch = dataset['TestLabels']
	ntest = testFeaturesTorch.size(0)

	print(nsamp)
	print(ntest)



	### Also normalize the test feature and concantenate with 1
	#testFeaturesTorch = testFeaturesTorch/torch.max(torch.norm(trainFeaturesTorch, dim = 1))
	testFeaturesTorch = testFeaturesTorch[:, 0:input_dim-1]
	testFeaturesTorch = (testFeaturesTorch.t()/(torch.norm(testFeaturesTorch, dim = 1))).t()

	testFeaturesTorch = torch.cat((torch.ones(ntest, 1).type(torch.DoubleTensor), testFeaturesTorch), dim = 1)
	

	### Extract the number of samples
	sz = list(trainFeaturesTorch.size())
	sz_test = list(testFeaturesTorch.size())


	trainFeatures = trainFeaturesTorch.numpy()
	trainLabels = trainLabelsTorch.numpy()
	testFeatures = testFeaturesTorch.numpy()
	testLabels = testLabelsTorch.numpy()


	# This is the code that finds linear regression solution
	relError, m = TensorGenerator.BestLinearMap(np.transpose(trainLabels), np.transpose(trainFeatures), input_dim, 4,  sz[0])
	X = Veronese(trainFeaturesTorch, input_dim,4,sz[0]).numpy()
	accuracy = (trainLabels == np.sign(np.matmul(X,m))).mean()
	X = Veronese(testFeaturesTorch, input_dim,4,sz_test[0]).numpy()

	genAcc = (testLabels == np.sign(np.matmul(X,m))).mean()

	print(relError)
	print(accuracy)
	print(genAcc)



	trainFeaturesTorch = trainFeaturesTorch.type(dtype)
	trainLabelsTorch = trainLabelsTorch.type(dtype)

	testFeaturesTorch = testFeaturesTorch.type(dtype)
	testLabelsTorch = testLabelsTorch.type(dtype)


	trainLoader = [trainLabelsTorch]

	# Save out trial metadata
	results = {}
	results['Meta'] = {}

	results['Meta']['lsq_residual'] = relError 
	results['Meta']['Accuracy'] = accuracy
	results['Meta']['Generalization'] = genAcc

	torch.save(results, out_file)

	# results['Meta']['Sample Number'] = nsamp 
	# results['Meta']['Lower Hidden'] = upper_hidden 
	# results['Meta']['Upper Hidden'] = upper_hidden 
	# results['Meta']['Input Dimension'] = input_dim 

	# results['Meta']['Features'] = trainFeaturesTorch
	# results['Meta']['Labels'] = trainLabelsTorch
	# #results['Meta']['lsq_residual'] = relError


	for trial in range(ntrial):

		results[trial] = {}

		for hidden_iter in range(lower_hidden, upper_hidden, 5):

			hidden = hidden_iter + 1

			print('%i Hidden Units, Trial %i \n' % (hidden, trial))
			results[trial][hidden_iter] = {}


			layers = np.array(hidden)



			#######################################################################################
			# Train network without norm
			print('Regular QNN \n')

			learning = learning_start
			model = NF.QuadraticNet2(input_dim, hidden, output_dim)
			model.type(dtype)
			loss_fn = nn.MSELoss()
			optimizer = optim.Adam(model.parameters(), lr = learning)

			epoch_trace = []
			residual_trace = []

			results[trial][hidden_iter]['Quadratic'] = {}

			model.eval()
			output = model(trainFeaturesTorch, dtype).detach().cpu().numpy()

			I_out = torch.eye(hidden).type(dtype)
			I_1 = torch.eye(input_dim**2).type(dtype)

			eta = 10000



			best = 1.0
			

			for epoch in range(training_epochs):
				model.train()
				for y in trainLoader:
					output = model(trainFeaturesTorch, dtype).type(dtype)
					output = output.squeeze()

					aWeights = torch.zeros((hidden, input_dim**2)).type(dtype)

					for q in range(hidden):
						temp = torch.mm(model.Layer1.weight.t(), torch.diag(model.Layer1_linear.weight[q, :]))
						temp2 = torch.mm(temp, model.Layer1.weight)
						aWeights[q, :] = temp2.reshape([-1])


					loss = ((loss_fn(output, y)) + eta*loss_fn(torch.mm(model.Layer2.weight.t(), model.Layer2.weight), I_out) +
						eta*loss_fn(torch.mm(aWeights.t(), aWeights), I_1))

					# Run the model backward and take a step using the optimizer.
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()


				if (epoch % 2000 == 0):


					model.eval()
					output = model(trainFeaturesTorch, dtype).squeeze().detach().cpu().numpy()
					residual_all = np.sum((trainLabels - output)**2)

					residual_all_nn = np.asscalar(residual_all*(1/(np.linalg.norm(trainLabels)**2)))

					if (residual_all_nn < best):
						best = residual_all_nn


					print('Epoch %i: Current residual: %.6f' % (epoch, residual_all_nn))
					epoch_trace = np.append(epoch_trace, epoch)
					residual_trace = np.append(residual_trace, residual_all_nn)

				#adjustLearningRate(optimizer, epoch, learning)



			model.eval()

			output = model(trainFeaturesTorch, dtype).squeeze().detach().cpu().numpy()
			output_test = model(testFeaturesTorch, dtype).squeeze().detach().cpu().numpy()
			residual_all = np.sum((trainLabels - output)**2)

			residual_all_nn = np.asscalar(residual_all*(1/(np.linalg.norm(trainLabels)**2)))	

			accuracy = (trainLabels == np.sign(output)).mean()
			genAcc = (testLabels == np.sign(output_test)).mean()	

			print('[Trial: %i] Residual NN: %.5f, Accuracy: %.5f, Generalization: %.5f' % (trial, residual_all_nn, accuracy, genAcc))

			trace = np.vstack((epoch_trace, residual_trace))

			#results[trial][hidden_iter]['Quadratic']['Model'] = model.state_dict()
			results[trial][hidden_iter]['Quadratic']['Residual'] = residual_all_nn
			results[trial][hidden_iter]['Quadratic']['Trace'] = trace
			results[trial][hidden_iter]['Quadratic']['Best'] = best
			results[trial][hidden_iter]['Quadratic']['Accuracy'] = accuracy
			results[trial][hidden_iter]['Quadratic']['Generalization'] = genAcc


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
	lr = learning * (0.5 ** (epoch // 100000))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


if __name__ == '__main__':
	args = parser.parse_args()
	main(args)





