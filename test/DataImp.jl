using Distributions
using Random
using Plots
#set parameters for data generation
μ=1
σ=1
#Sample size and numbe rof ismulations
n=100000
numsim=10
nHidden = [3,4,10,12,3,4,5,9,8,7,6,4,3]
#list for keeping track of simulations
βlist=[]
βbiaslist=[]
βnnlist=[]
misslist=[]
#Neural Net Parameters and commands
maxIter = 10000
include("NeuralNet.jl")
include("sgdfunc.jl")

for j in 1:numsim
#Draw random varaibles
normdist=Normal(μ,σ)
X1=rand(normdist,n)
X2=rand(normdist,n)
ϵ1=rand(Normal(0,1),n)
ϵ2=rand(Normal(0,1),n)
#form dependent variables
X3=X1.^2+X2.^2+ϵ1
y=2*X1-3*X2+4*X3+ϵ2


#Check Consistency
X=hcat(X1,X2,X3)
β=inv(X'*X)*X'*y
push!(βlist,β)

#Systematic Removal
indexl= [] #Contains list of indices for nonmissing data
indexm=[] #Contains list of indices for missing data
for i in 1:n
    k=rand(Binomial(1,.5))
    if (5>y[i]>-1 && 2>X3[i]>-1) && k==true #removal conditions
        push!(indexm,i)
    else
        push!(indexl,i)
    end
end

#Number of missing
push!(misslist,n-length(indexl))
#Show Bias
X=hcat(X1[indexl],X2[indexl],X3[indexl])
ymiss=y[indexl]
βbias=inv(X'*X)*X'*ymiss
push!(βbiaslist,βbias)

#Fill in Missing Values
#Shuffle for randomness maybe not necesarry

#Create training and validation sets
numtrain=convert(Int,round(.7*length(indexl)))
xtrain=hcat(X1[indexl[1:numtrain]],X2[indexl[1:numtrain]])
x3train=(X3[indexl[1:numtrain]])
xvalid=hcat(X1[indexl[(numtrain+1):length(indexl)]],X2[indexl[(numtrain+1):length(indexl)]])
x3valid=(X3[indexl[numtrain+1:length(indexl)]])

#Number of training examples
ntrain = size(xtrain,1)
#add bias variables
xtrain = [ones(ntrain,1) xtrain]
#number of features
dtrain = size(xtrain,2)
#number and size of each hidden layer

#get number of total weights
nParams = NeuralNet_nParams(dtrain,nHidden)
#intialize weight matrix



#Run neural network to get weights
w, valid = SGDBabysitter(NeuralNet_backprop, NeuralNet_predict,
maxIter, nHidden, nParams, xtrain, x3train, xvalid, x3valid)

#predict on missing data
X3nn=NeuralNet_predict(w,hcat(ones(length(indexm)),X1[indexm],X2[indexm]),nHidden)
Xnn=vcat(hcat(X1[indexl],X2[indexl],X3[indexl]),hcat(X1[indexm],X2[indexm],X3nn))
ynn=vcat(y[indexl],y[indexm])
βnn=inv(Xnn'*Xnn)*Xnn'*ynn
push!(βnnlist,βnn)
end
print(sum(βbiaslist)/numsim,sum(βlist)/numsim,sum(βnnlist)/numsim,sum(misslist)/numsim, " ")
