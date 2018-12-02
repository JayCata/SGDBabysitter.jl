# Load X and y variable
using SGDBabysitter
using Pkg
using HDF5
using Plots

data=h5open("test\\mnist.hdf5","r")
xtrain=transpose(read(data,"x_train"))
ytrain=read(data,"t_train")
xvalid=transpose(read(data,"x_valid"))
yvalid=read(data,"t_valid")

n = size(xtrain,1)
xtrain = [ones(n,1) xtrain]
d = size(xtrain,2)
k=length(unique(ytrain))
# Choose network structure and randomly initialize weights
include("NeuralNetclass.jl")
nHidden = [3]
nParams = NeuralNet_nParams_class(d,nHidden,k)
W = randn(nParams,1)
f,g = NeuralNet_backprop_class(W, xtrain[1,:], ytrain[1], nHidden,k)
# Train with stochastic gradient
maxIter = 10000
j=1
checks=[]
valid=[]
stepSize = 1e-4


for t in 1:maxIter
	global W, j
	# The stochastic gradient update:
	i = rand(1:n,2)
	f,g = NeuralNet_backprop(W, xtrain[i,:], ytrain[i], nHidden)
	W = W - stepSize*g

	# Every few iterations, plot the data/model:
	if (mod(t-1,round(maxIter/50)) == 0)
		print("Training iteration = $(t-1)",)

		yhat = NeuralNet_predict(W,[ones(size(xvalid,1)) xvalid],nHidden)

		push!(valid,sum((yhat-yvalid).^2)/length(yvalid))
		push!(checks,j)
		j=j+1
	end
end
plot(checks,valid)
