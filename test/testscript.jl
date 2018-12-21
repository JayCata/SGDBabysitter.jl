using Plots
using HDF5
using Pkg
using Statistics

data=h5open("test/mnist.hdf5","r")
xtrain=transpose(read(data,"x_train"))
ytrain=read(data,"t_train")
xvalid=transpose(read(data,"x_valid"))
yvalid=read(data,"t_valid")

n = size(xtrain,1)
xtrain = [ones(n,1) xtrain]
d = size(xtrain,2)

# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [3,3,4,3]
# nHidden = [5,5,5,5]
nParams = NeuralNet_nParams(d,nHidden)
w = randn(nParams,1)

# Train with stochastic gradient
include("sgdfunc.jl")
maxIter = 10000

#----Uncomment below for hyperparameter search:----#

# aTest = [0.1, 0.01, 0.001, 0.0001, 0.00001]
# BTest = [1, 5, 10, 20, 100] #200 500 1000
#
# bestvalid = Inf
# bestw = []
# bestparams = [0,0]
# for B in BTest
#     for a in aTest
#         global bestvalid, bestw, bestparams
#         wb, validerr = VanillaSGD(NeuralNet_backprop, NeuralNet_predict, maxIter, nHidden,
#                                     nParams, xtrain, ytrain, xvalid, yvalid, a, B)
#         if (validerr[end]<bestvalid)
#             bestvalid=validerr[end]
#             bestw=wb
#             bestparams = [a,B]
#         end
#     end
# end

#best hyperparams found to be [0.0001, 20]

wbSGDB, validSGDB = SGDBabysitter(NeuralNet_backprop, NeuralNet_predict,
maxIter, nHidden, nParams, xtrain, ytrain, xvalid, yvalid)
savefig("SGDB_MNIST.pdf")

wbVan, validVan = VanillaSGD(NeuralNet_backprop, NeuralNet_predict, maxIter, nHidden, nParams, xtrain, ytrain, xvalid, yvalid, bestparams[1], convert(Int64, bestparams[2]))
savefig("Vanilla_MNIST.pdf")
