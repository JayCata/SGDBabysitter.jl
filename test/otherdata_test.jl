using CSV
using DataFrames
# df = CSV.read("parkinsons_updrs", header=false, delim=',')
# df = readtable("~/test/parkinsons_updrs.csv")
medData = CSV.read("/home/puranjay/ubc/term3/SGDBabysitter.jl/test/parkinsons_updrs.csv", delim=',')

# deletecols!(medData, :motor_UPDRS)

y1 = medData[[:total_UPDRS]]
y2 = medData[[:motor_UPDRS]]
deletecols!(medData, [:motor_UPDRS, :total_UPDRS])
xtrain = convert(Array,first(medData, 4000))
ytrain1 = convert(Array,first(y1, 4000))
ytrain2 = convert(Array,first(y2, 4000))
xvalid = convert(Array,last(medData, 1875))
yvalid1 = convert(Array,last(y1, 1875))
yvalid2 = convert(Array,last(y2, 1875))

n2 = size(xtrain,1)
xtrain = [ones(n2,1) xtrain]
d2 = size(xtrain,2)

# Choose network structure and randomly initialize weights
include("NeuralNet.jl")
nHidden = [3,3,4,3]
# nHidden = [5,5,5,5]
nParams = NeuralNet_nParams(d2,nHidden)
w = randn(nParams,1)

# Train with stochastic gradient
include("sgdfunc.jl")
maxIter = 10000

#----Uncomment below for hyperparameter search:----#

aTest = [0.1, 0.01, 0.001, 0.0001, 0.00001]
BTest = [1, 5, 10, 20, 100] #200 500 1000

bestvalid = Inf
bestw = []
bestparams2 = [0,0]
for B in BTest
    for a in aTest
        global bestvalid, bestw, bestparams2
        wb, validerr = VanillaSGD(NeuralNet_backprop, NeuralNet_predict, maxIter, nHidden, nParams, xtrain, ytrain1, xvalid, yvalid1, a, B)
        if (validerr[end]<bestvalid)
            bestvalid=validerr[end]
            bestw=wb
            bestparams2 = [a,B]
        end
    end
end

#best hyperparams found to be [0.1, 10]

wb_park, valid_park = SGDBabysitter(NeuralNet_backprop, NeuralNet_predict,
maxIter, nHidden, nParams, xtrain, ytrain1, xvalid, yvalid1)
savefig("SGDB_Parkinsons.pdf")

wb_vpark, valid_vpark = VanillaSGD(NeuralNet_backprop, NeuralNet_predict, maxIter, nHidden,nParams, xtrain, ytrain1, xvalid, yvalid1, bestparams2[1], convert(Int64, bestparams2[2]))
savefig("Vanilla_Parkinsons.pdf")
