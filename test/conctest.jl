using CSV
using Plots
using Pkg
using Random

concdata = CSV.read("/home/puranjay/ubc/term3/SGDBabysitter.jl/test/Concrete_Data.csv", delim=',')

X = concdata[1:8]
y = concdata[9]
n = size(X)[1]
idxshuffle = shuffle(1:n)
X = X[idxshuffle, :]
y = y[idxshuffle, :]
splitval=750
xtrain = convert(Array{Float64}, X[1:splitval, :])
ytrain = convert(Array{Float64}, y[1:splitval, :])

xvalid = convert(Array{Float64}, X[splitval+1:end, :])
yvalid = convert(Array{Float64}, y[splitval+1:end, :])

xtrain = [ones(splitval,1) xtrain]
# xvalid = [ones(n-splitval,1) xvalid]
d = size(xtrain,2)

include("NeuralNet.jl")
nHidden = [3,3,4,3]
nParams = NeuralNet_nParams(d,nHidden)
w = randn(nParams,1)

include("sgdfunc.jl")
maxIter = 10000

aTest = [0.1, 0.01, 0.001, 0.0001, 0.00001]
BTest = [1, 5, 10, 20, 100] #200 500 1000

bestvalid = Inf
bestw = []
bestparams3 = [0,0]
for B in BTest
    for a in aTest
        global bestvalid, bestw, bestparams3
        wb, validerr = VanillaSGD(NeuralNet_backprop, NeuralNet_predict, maxIter, nHidden, nParams, xtrain, ytrain, xvalid, yvalid, a, B)
        if (validerr[end]<bestvalid)
            bestvalid=validerr[end]
            bestw=wb
            bestparams3 = [a,B]
        end
    end
end

#best hyperparams found to be [0.0001, 100]
w_vconc, valid_vconc = VanillaSGD(NeuralNet_backprop, NeuralNet_predict, maxIter, nHidden,nParams, xtrain, ytrain, xvalid, yvalid, bestparams3[1], convert(Int64, bestparams3[2]))
savefig("Vanilla_Concrete.pdf")
w_conc, valid_conc = SGDBabysitter(NeuralNet_backprop, NeuralNet_predict, maxIter, nHidden, nParams, xtrain, ytrain, xvalid, yvalid)
savefig("SGDB_Concrete.pdf")
yhat_conc = NeuralNet_predict(w_conc,[ones(size(xvalid,1)) xvalid],nHidden)

yhat_vconc = NeuralNet_predict(w_vconc,[ones(size(xvalid,1)) xvalid],nHidden)
