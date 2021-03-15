import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import collections
from torch.autograd import Variable


## Netowrk Types ##

##########################################
# These are networks with one QL layer with various activation types

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


class ThirdOrderNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(ThirdOrderNet, self).__init__()

		self.hiddenLayer = nn.Linear(D_in, H, bias = False)

		self.output = nn.Linear(H, D_out, bias = False)


	def forward(self, x, dtype):
		x = self.hiddenLayer(x)
		y_pred = self.output(x**3)
		return y_pred


class FourthOrderNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(FourthOrderNet, self).__init__()

		self.hiddenLayer = nn.Linear(D_in, H, bias = False)

		self.output = nn.Linear(H, D_out, bias = False)
		self.tanh = nn.Tanh()

	def forward(self, x, dtype):
		x = self.hiddenLayer(x)
		y_pred = self.output(x**4)
		return y_pred



class EigthOrderNet(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(EigthOrderNet, self).__init__()

		self.hiddenLayer = nn.Linear(D_in, H, bias = False)

		self.output = nn.Linear(H, D_out, bias = False)
		self.tanh = nn.Tanh()

	def forward(self, x, dtype):
		x = self.hiddenLayer(x)
		y_pred = self.output(x**8)
		return y_pred





class QuadraticForm(torch.nn.Module):
	def __init__(self, d, out):
		super(QuadraticForm, self).__init__()

		self.A = nn.Bilinear(d, d, out, bias = False)
		#self.A.weight.data = A_init

	def forward(self, x):
		y_pred = self.A(x, x)
		return y_pred




##########################################
# These networks are from trying to hard code the number of layers.
# The better network for this is QuadraticNet_VaryHidden

class QuadraticNet2(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(QuadraticNet2, self).__init__()

		self.Layer1 = nn.Linear(D_in, H*D_in, bias = False)
		self.Layer1_linear = nn.Linear(H*D_in, H, bias = False)
		
		self.Layer2 = nn.Linear(H, H*D_out, bias = False)
		self.Layer2_linear = nn.Linear(H*D_out, D_out, bias = False)

		self.input_dim = D_in
		self.hidden = H


	def forward(self, x, dtype):
		x = self.Layer1(x)
		h = self.Layer1_linear(x**2)
		x = self.Layer2(h)
		y_pred = self.Layer2_linear(x**2)

		# Form big matrix of A's
		aWeights = torch.zeros((self.hidden, self.input_dim**2)).type(dtype)


		for q in range(self.hidden):
			temp = torch.mm(self.Layer1.weight.t(), torch.diag(self.Layer1_linear.weight[q, :]))
			temp2 = torch.mm(temp, self.Layer1.weight)
			aWeights[q, :] = temp2.reshape([-1])

		return y_pred, aWeights




class QuadraticNet2_Norm(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(QuadraticNet2_Norm, self).__init__()

		self.Layer1 = nn.Linear(D_in, H*D_in, bias = False)
		self.Layer1_linear = nn.Linear(H*D_in, H, bias = False)
		self.scalar1 = nn.Parameter(torch.randn(1), requires_grad=True)
		
		self.Layer2 = nn.Linear(H, H*D_out, bias = False)
		self.Layer2_linear = nn.Linear(H*D_out, D_out, bias = False)
		self.scalar2 = nn.Parameter(torch.randn(1), requires_grad=True)
		self.scalar3 = nn.Parameter(torch.randn(1), requires_grad=True)


	def forward(self, x, dtype):
		h1 = self.Layer1(x)
		h2 = self.Layer1_linear(h1**2) + self.scalar1 * torch.norm(x, p=2, dim = 1, keepdim = True)**2
		h3 = self.Layer2(h2)
		y_pred = self.Layer2_linear(h3**2) + self.scalar2 * torch.norm(h2, p=2, dim = 1, keepdim = True)**2 + \
			self.scalar3 * torch.norm(x, p=2, dim = 1, keepdim = True)**4
		return y_pred



class QuadraticNet3(torch.nn.Module):
	def __init__(self, D_in, H1, H2, D_out):
		super(QuadraticNet3, self).__init__()

		self.Layer1 = nn.Linear(D_in, H1*D_in, bias = False)
		self.Layer1_linear = nn.Linear(H1*D_in, H1, bias = False)

		self.Layer2 = nn.Linear(H1, H2*H1, bias = False)
		self.Layer2_linear = nn.Linear(H2*H1, H2, bias = False)
		
		self.Layer3 = nn.Linear(H2, H2*D_out, bias = False)
		self.Layer3_linear = nn.Linear(H2*D_out, D_out, bias = False)


	def forward(self, x, dtype):
		h = self.Layer1(x)
		h = self.Layer1_linear(h**2)
		h = self.Layer2(h)
		h = self.Layer2_linear(h**2)
		h = self.Layer3(h)
		y_pred = self.Layer3_linear(h**2)
		return y_pred



class QuadraticNet_simple(torch.nn.Module):
	def __init__(self, D_in, H, D_out):
		super(QuadraticNet_simple, self).__init__()

		d = collections.OrderedDict()

		d.update({('Layer1', nn.Linear(D_in, H, bias = False))})
		d.update({('Quadratic1', QuadraticActivation())})
		d.update({('Layer1_linear', nn.Linear(H, D_out, bias = False))})


		# d.update({('Layer2', nn.Linear(H, H*D_out, bias = False))})
		# d.update({('Quadratic2', QuadraticActivation())})
		# d.update({('Layer2_linear', nn.Linear(H*D_out, D_out, bias = False))})

		self.hiddenLayers = nn.Sequential(d)


	def forward(self, x, dtype):

		y_pred = self.hiddenLayers(x)
		return y_pred


class QuadraticDeep(torch.nn.Module):
	def __init__(self, D_in, D_out, layers, dtype):
		super(QuadraticDeep, self).__init__()

		d = collections.OrderedDict()
		self.layers = layers.size

		for l in range(self.layers):
			if (l == 0):
				if (self.layers == 1):
					H = layers
				else:
					H = layers[0]
				d.update({('Layer' + str(l), nn.Linear(D_in, H*D_in, bias = False).type(dtype))})
				d.update({('Quadratic' + str(l), QuadraticActivation())})
				d.update({('Layer_linear' + str(l), nn.Linear(H*D_in, H, bias = False).type(dtype))})
				# d.update({('Layer' + str(l), nn.Linear(D_in, 100, bias = False).type(dtype))})
				# d.update({('Quadratic' + str(l), QuadraticActivation())})
				# d.update({('Layer_linear' + str(l), nn.Linear(100, H, bias = False).type(dtype))})
			else:
				H_prev = layers[l - 1]
				H = layers[l]
				d.update({('Layer' + str(l), nn.Linear(H_prev, H_prev*H, bias = False).type(dtype))})
				d.update({('Quadratic' + str(l), QuadraticActivation())})
				d.update({('Layer_linear' + str(l), nn.Linear(H_prev*H, H, bias = False).type(dtype))})

		if (self.layers == 1):
			H = layers
		else:
			H = layers[self.layers - 1]
		d.update({('LayerOut', nn.Linear(H, D_out*H, bias = False).type(dtype))})
		d.update({('QuadraticOut', QuadraticActivation())})
		d.update({('Layer_linearOut', nn.Linear(H*D_out, D_out, bias = False).type(dtype))})

		self.hiddenLayers = nn.Sequential(d)
		


	def forward(self, x, dtype):
		y_pred = self.hiddenLayers(x)
		return y_pred



class QuadraticTest(torch.nn.Module):
	def __init__(self, D_in, H, D_out, layers, dtype):
		super(QuadraticTest, self).__init__()

		self.quadratic = {}
		self.linear = {}
		self.layers = layers

		for l in range(self.layers):
			if (l == 0):
				self.quadratic[0] = nn.Linear(D_in, H*D_in).type(dtype)
				self.linear[0] = nn.Linear(H*D_in, H).type(dtype)
			else:
				self.quadratic[l] = nn.Linear(H, H*H).type(dtype)
				self.linear[l] = nn.Linear(H*H, H).type(dtype)

		self.output_quadratic = nn.Linear(H, D_out*H).type(dtype)
		self.output = nn.Linear(H*D_out, D_out).type(dtype)
		


	def forward(self, x, dtype):


		for l in range(self.layers):
			x = (self.quadratic[l](x))**2
			x = self.linear[l](x)

		x = self.output_quadratic(x)
		y_pred = self.output(x)
		return y_pred



class QuadraticDeep_VaryHidden(torch.nn.Module):
	def __init__(self, D_in, D_out, layers, dtype):
		super(QuadraticDeep_VaryHidden, self).__init__()

		self.quadratic = {}
		self.linear = {}
		self.layers = layers.size

		for l in range(self.layers):
			if (l == 0):
				H = layers[0]
				self.quadratic[0] = nn.Linear(D_in, H*D_in, bias = False).type(dtype)
				self.linear[0] = nn.Linear(H*D_in, H, bias = False).type(dtype)
			else:
				H_prev = layers[l - 1]
				H = layers[l]
				self.quadratic[l] = nn.Linear(H_prev + 1, (H_prev+1)*H, bias = False).type(dtype)
				self.linear[l] = nn.Linear((H_prev+1)*H, H, bias = False).type(dtype)

		H = layers[self.layers - 1]
		self.output_quadratic = nn.Linear(H+1, D_out*H, bias = False).type(dtype)
		self.output = nn.Linear(H*D_out, D_out, bias = False).type(dtype)
		


	def forward(self, x, batch, dtype):
		h = x
		if (batch == 1):
			one = torch.ones([1]).type(dtype)
		else:
			one = torch.ones([batch, 1]).type(dtype)


		for l in range(self.layers):
			h = (self.quadratic[l](h))**2
			h = self.linear[l](h)
			if (batch == 1):
				h = torch.cat((one, h))
			else:
				h = torch.cat((one, h), dim = 1)

		h = self.output_quadratic(h)
		y_pred = self.output(h)
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



class QuadraticActivation(nn.Module):

	def __init__(self):
		super(QuadraticActivation, self).__init__()


	def forward(self, x):
		return x**2









