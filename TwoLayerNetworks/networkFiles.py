import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
from torch.autograd import Variable


## Netowrk Types ##

class QuadraticNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(QuadraticNet, self).__init__()

		self.hiddenLayer = nn.Linear(D_in, H, bias = False)

		self.output = nn.Linear(H, D_out, bias = False)
		self.tanh = nn.Tanh()

	def forward(self, x, dtype):
		x = self.hiddenLayer(x)
		y_pred = self.output(x**2)
		return y_pred

class QuadraticNetNorm(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(QuadraticNetNorm, self).__init__()

		self.hiddenLayer = nn.Linear(D_in, H, bias = False)
		self.scalar = nn.Parameter(torch.randn(1), requires_grad=True)

		self.output = nn.Linear(H, D_out, bias = False)
		self.tanh = nn.Tanh()

	def forward(self, x, dtype):
		h = self.hiddenLayer(x)
		y_pred = self.output(h**2) + self.scalar * torch.norm(x, p=2, dim = 1, keepdim = True)**2
		return y_pred


class DeepNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out, layers):
		super(DeepNet, self).__init__()
		d = collections.OrderedDict()

		# First hidden layer:
		d.update({('Layer0', nn.Linear(D_in, H))})
		d.update({('Tanh0', nn.Tanh())})
		
		# Intermediate hidden layers
		for i in range(1,layers):
			d.update({('Layer'+str(i), nn.Linear(H, H))})
			d.update({('Tanh'+str(i), nn.Tanh())})


		self.hiddenLayers = nn.Sequential(d)

		self.output = nn.Linear(H, D_out)
		self.tanh = nn.Tanh()

	def forward(self, x, dtype):
		x = self.hiddenLayers(x)
		y_pred = self.tanh(self.output(x))
		return y_pred


# class DeepNetInput(torch.nn.Module):
# 	def __init__(self, D_in, H, D_out, layers):
# 		super(DeepNetInput, self).__init__()
# 		d = collections.OrderedDict()
# 		self.input_size = D_in
# 		self.hidden_size = H
# 		self.output_size = D_out
		
# 		# Intermediate hidden layers
# 		for i in range(0,layers):
# 			d.update({('Layer'+str(i), inputLayers(self.input_size, self.hidden_size))})


# 		self.hiddenLayers = inputSequential(d)

# 		self.outputLayer = nn.Linear(self.hidden_size, self.output_size)
# 		self.tanh = nn.Tanh()

# 	def forward(self, x, dtype):
# 		#dtype = torch.cuda.FloatTensor
# 		u = Variable(torch.zeros(self.hidden_size).type(dtype))
# 		u = self.hiddenLayers(u, x)
# 		y_pred = self.tanh(self.outputLayer(u))
# 		return y_pred



# class Recurrent(torch.nn.Module):
# 	def __init__(self, D_in, H, D_out, layers):
# 		super(Recurrent, self).__init__()
# 		self.iteratedLayer = RepeatedLayers(D_in, H, layers)
# 		self.outputLayer = nn.Linear(H, D_out)
# 		self.tanh = nn.Tanh()
# 		self.hidden_size = H

# 	def forward(self, x, dtype):
# 		#dtype = torch.cuda.FloatTensor
# 		u = Variable(torch.zeros(self.hidden_size).type(dtype))
# 		u = self.iteratedLayer(u, x)
# 		y_pred = self.tanh(self.outputLayer(u))
# 		return y_pred


# class RecurrentScaled(torch.nn.Module):
# 	def __init__(self, D_in, H, D_out, layers):
# 		super(RecurrentScaled, self).__init__()
# 		self.iteratedLayer = RepeatedLayersScaled(D_in, H, layers)
# 		self.outputLayer = nn.Linear(H, D_out)
# 		self.tanh = nn.Tanh()
# 		self.hidden_size = H

# 	def forward(self, x, dtype):
# 		#dtype = torch.cuda.FloatTensor
# 		u = Variable(torch.zeros(self.hidden_size).type(dtype))
# 		u = self.iteratedLayer(u, x)
# 		y_pred = self.tanh(self.outputLayer(u))
# 		return y_pred

# class RecurrentScaledMasked(torch.nn.Module):
# 	def __init__(self, D_in, H, D_out, layers, imageSize, boundarySize):
# 		super(RecurrentScaledMasked, self).__init__()
# 		weightMask = generateSquareWeightMask(imageSize, boundarySize)
# 		self.iteratedLayer = RepeatedLayersScaledMasked(D_in, H, layers, weightMask)
# 		self.outputLayer = nn.Linear(H, D_out)
# 		self.tanh = nn.Tanh()
# 		self.hidden_size = H

# 	def forward(self, x, dtype):
# 		#dtype = torch.cuda.FloatTensor
# 		u = Variable(torch.zeros(self.hidden_size).type(dtype))
# 		u = self.iteratedLayer(u, x)
# 		y_pred = self.tanh(self.outputLayer(u))
# 		return y_pred


# class RecurrentScaledGrid(torch.nn.Module):
# 	def __init__(self, D_in, H, D_out, layers, imageSize):
# 		super(RecurrentScaledGrid, self).__init__()
# 		weightMask = generateGridWeightMask(imageSize)
# 		self.iteratedLayer = RepeatedLayersScaledMasked(D_in, H, layers, weightMask)
# 		self.outputLayer = nn.Linear(H, D_out)
# 		self.tanh = nn.Tanh()
# 		self.hidden_size = H

# 	def forward(self, x, dtype):
# 		#dtype = torch.cuda.FloatTensor
# 		u = Variable(torch.zeros(self.hidden_size).type(dtype))
# 		u = self.iteratedLayer(u, x)
# 		y_pred = self.tanh(self.outputLayer(u))
# 		return y_pred


# class RecurrentScaledMultiplicative(torch.nn.Module):
# 	def __init__(self, D_in, H, D_out, layers):
# 		super(RecurrentScaledMultiplicative, self).__init__()
# 		self.iteratedLayer = RepeatedLayersMultiplicative(D_in, H, layers)
# 		self.outputLayer = nn.Linear(H, D_out)
# 		self.tanh = nn.Tanh()
# 		self.hidden_size = H

# 	def forward(self, x, dtype):
# 		#dtype = torch.cuda.FloatTensor
# 		u = Variable(torch.zeros(self.hidden_size).type(dtype))
# 		u = self.iteratedLayer(u, x)
# 		y_pred = self.tanh(self.outputLayer(u))
# 		return y_pred













# ## Classes used to construct network types ##

# class inputLayers(nn.Module):
# 	def __init__(self, D_input, hidden):
# 		super(inputLayers, self).__init__()
# 		self.hiddenWeight = nn.Linear(hidden, hidden)
# 		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
# 		self.tanh = nn.Tanh()

# 	def forward(self, x, y):
# 		u = F.tanh(self.hiddenWeight(x) + self.inputWeight(y))
# 		return u

# class inputSequential(nn.Sequential):
# 	def forward(self, inputOne, inputTwo):
# 		hidden  = inputOne
# 		for module in self._modules.values():
# 			hidden = module(hidden, inputTwo)
# 		return hidden

# class RepeatedLayers(torch.nn.Module):
# 	def __init__(self, D_input, hidden, layers):
# 		super(RepeatedLayers, self).__init__()
# 		self.iteration = layers
# 		self.hiddenWeight = nn.Linear(hidden, hidden)
# 		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
# 		self.tanh = nn.Tanh()

# 	def forward(self, initial_hidden, input):
# 		u = initial_hidden.clone()
# 		for _ in range(0, self.iteration):
# 			v = self.hiddenWeight(u) + self.inputWeight(input)
# 			u = self.tanh(v)
# 		return u


# class RepeatedLayersScaled(torch.nn.Module):
# 	def __init__(self, D_input, hidden, layers):
# 		super(RepeatedLayersScaled, self).__init__()
# 		self.iteration = layers
# 		self.hiddenWeight = nn.Linear(hidden, hidden)
# 		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
# 		self.tanh = nn.Tanh()
# 		self.scalar = nn.Parameter(torch.ones(1)*2, requires_grad=True)

# 	def forward(self, initial_hidden, input):
# 		u = initial_hidden.clone()
# 		for _ in range(0, self.iteration):
# 			v = self.hiddenWeight(u) + self.inputWeight(input)
# 			u = self.tanh(v * self.scalar.expand_as(v))
# 		return u

# class RepeatedLayersMultiplicative(torch.nn.Module):
# 	def __init__(self, D_input, hidden, layers):
# 		super(RepeatedLayersMultiplicative, self).__init__()
# 		self.iteration = layers
# 		self.hiddenWeight = nn.Linear(hidden, hidden)
# 		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
# 		self.tanh = nn.Tanh()
# 		self.scalar = nn.Parameter(torch.ones(1)*2, requires_grad=True)

# 	def forward(self, initial_hidden, input):
# 		u = initial_hidden.clone()
# 		for _ in range(0, self.iteration):
# 			v = self.hiddenWeight(u)*self.inputWeight(input)
# 			u = self.tanh(v * self.scalar.expand_as(v))
# 		return u


# class RepeatedLayersMasked(torch.nn.Module):
# 	def __init__(self, D_input, hidden, layers, weightMask):
# 		super(RepeatedLayersMasked, self).__init__()
# 		self.iteration = layers
# 		self.hiddenWeight = nn.Parameter(torch.Tensor(hidden, hidden))
# 		self.hiddenBias = nn.Parameter(torch.Tensor(hidden))
# 		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
# 		self.weightMask = weightMask
# 		self.tanh = nn.Tanh()

# 	def forward(self, initial_hidden, input):
# 		u = initial_hidden.clone()
# 		for _ in range(0, self.iteration):
# 			v = F.linear(u, (self.weightMask * self.hiddenWeight), self.hiddenBias) + self.inputWeight(input)
# 			u = self.tanh(v)
# 		return u


# # class RepeatedLayersScaledMasked(torch.nn.Module):
# # 	def __init__(self, D_input, hidden, layers, weightMask):
# # 		super(RepeatedLayersScaledMasked, self).__init__()
# # 		self.iteration = layers
# # 		self.hiddenWeight = nn.Parameter(torch.Tensor(hidden, hidden))
# # 		self.hiddenBias = nn.Parameter(torch.Tensor(hidden))
# # 		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
# # 		self.weightMask = weightMask
# # 		self.tanh = nn.Tanh()
# # 		self.scalar = nn.Parameter(torch.ones(1)*2, requires_grad=True)

# # 	def forward(self, initial_hidden, input):
# # 		u = initial_hidden.clone()
# # 		for _ in range(0, self.iteration):
# # 			v = F.linear(u, (self.weightMask * self.hiddenWeight), self.hiddenBias) + self.inputWeight(input)
# # 			u = self.tanh(v * self.scalar.expand_as(v))
# # 		return u

# class RepeatedLayersScaledMasked(torch.nn.Module):
# 	def __init__(self, D_input, hidden, layers, weightMask):
# 		super(RepeatedLayersScaledMasked, self).__init__()

# 		self.mask = weightMask
# 		self.invertMask = torch.ones((hidden, hidden)).type(torch.cuda.ByteTensor) - self.mask
# 		self.iteration = layers
# 		self.hiddenWeight = nn.Linear(hidden, hidden)
# 		self.inputWeight = nn.Linear(D_input, hidden, bias=False)
# 		self.tanh = nn.Tanh()
# 		self.scalar = nn.Parameter(torch.ones(1)*2, requires_grad=True)
# 		self.hiddenWeight.weight.data[self.invertMask] = 0
# 		self.hiddenWeight.weight.register_hook(self.backward_hook)


		

# 	def forward(self, initial_hidden, input):
# 		u = initial_hidden.clone()
# 		for _ in range(0, self.iteration):
# 			v = self.hiddenWeight(u) + self.inputWeight(input)
# 			u = self.tanh(v * self.scalar.expand_as(v))
# 		return u

# 	def backward_hook(self, grad):
# 		out = grad.clone()
# 		out[self.invertMask] = 0
# 		return out









