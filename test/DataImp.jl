using Distributions
using Random
using Plots
#set parameters for data generation

μ=0
σ=1
#Sample size, number of simulations, number and size of hidden layers
n=100000
numsim=100
nHidden = [6,15,20,15,6]
#list for keeping track of simulations
βlist=[]
βbiaslist=[]
βnnbabysitlist=[]
βnnvanlist=[]
ValErrBS=[]
ValErrVan=[]
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
ϵ2=rand(Normal(0,10),n)
#form dependent variables
X3=X1+X2+ϵ1
y=2*X1-3*X2+4*X3+ϵ2


#Check OLS estimates for all data
X=hcat(X1,X2,X3)
β=inv(X'*X)*X'*y
push!(βlist,β)

#Systematic Removal
indexl= [] #Contains list of indices for nonmissing data
indexm=[] #Contains list of indices for missing data
for i in 1:n
    k=rand(Binomial(1,.5))
    if (y[i]>1 && ϵ2[i]<5) && k==true #removal conditions
        push!(indexm,i)
    else
        push!(indexl,i)
    end
end

#Number of missing
push!(misslist,length(indexm))

#Get Bias β
Xmiss=hcat(X1[indexl],X2[indexl],X3[indexl])
ymiss=y[indexl]
βbias=inv(Xmiss'*Xmiss)*Xmiss'*ymiss
push!(βbiaslist,βbias)



#Create training and validation sets (70% training/30% validation)
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


#get number of total weights
nParams = NeuralNet_nParams(dtrain,nHidden)



#Run neural network to get weights with Vanilla SGD
bestvanval=Inf
wbestvan=randn(nParams)
for α in [.1, .01,.001,.0001,.00001]
w,validvan=VanillaSGD(NeuralNet_backprop,NeuralNet_predict,maxIter,nHidden, nParams,xtrain,x3train,xvalid,x3valid, α ,10)
if validvan < bestvanval
wbestvan=w
bestvanval=validvan
end
end
push!(ValErrVan,bestvanval)
#predict on missing data
X3nnvan=NeuralNet_predict(wbestvan,hcat(ones(length(indexm)),X1[indexm],X2[indexm]),nHidden)
#form new X and y
Xnnvan=vcat(hcat(X1[indexl],X2[indexl],X3[indexl]),hcat(X1[indexm],X2[indexm],X3nnvan))
ynnvan=vcat(y[indexl],y[indexm])

#Get vanilla SGD OLS estimates
βnnvan=inv(Xnnvan'*Xnnvan)*Xnnvan'*ynnvan
push!(βnnvanlist,βnnvan)

#Run neural network to get weights with SGDbabysitter
w, validBB = SGDBabysitter(NeuralNet_backprop, NeuralNet_predict,
maxIter, nHidden, nParams, xtrain, x3train, xvalid, x3valid,500)
push!(ValErrBS,validBB)
#predict on missing data
X3nnbabysit=NeuralNet_predict(w,hcat(ones(length(indexm)),X1[indexm],X2[indexm]),nHidden)
#form new X and y
Xnnbabysit=vcat(hcat(X1[indexl],X2[indexl],X3[indexl]),hcat(X1[indexm],X2[indexm],X3nnbabysit))
ynnbabysit=vcat(y[indexl],y[indexm])

#Get SGDBabysitter OLS estimates
βnnbabysit=inv(Xnnbabysit'*Xnnbabysit)*Xnnbabysit'*ynnbabysit
push!(βnnbabysitlist,βnnbabysit)


end
#Final report of all OLS estimates
print("All Data Coefficients: ", sum(βlist)/numsim, " Coefficients with Missing Data: ", sum(βbiaslist)/numsim," Coefficients with NN Imputation (SGDBabysitter): ",sum(βnnbabysitlist)/numsim," Coefficients with NN Imputation (VanillaSGD): " ,sum(βnnvanlist)/numsim," Average Number of Missing Variables: ", sum(misslist)/numsim, " ")
print("Average Validation Error SGDBabysitter: ", sum(ValErrBS)/numsim, " Average Validation Error VanillaSGD ", sum(ValErrVan)/numsim, " ", "Average Test Error SGDBabysitter: ", sum((X3nnbabysit-X3[indexm]).^2)/numsim, " Average Validation Error VanillaSGD ", sum((X3nnvan-X3[indexm]).^2)/numsim,)
