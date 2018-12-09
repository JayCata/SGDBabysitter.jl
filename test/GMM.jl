using Distributions
using ForwardDiff
dof=1
n=1000
chidist=Chisq(dof)
X1=rand(chidist,n)
print("First Moment Conditions is E[X1-$dof]=0")

mhat(d)=(1/n)sum(X1-d)

#First Step
W=1
f(d)=mhat(d)*W*mhat(d)
g(d)=ForwardDiff(f(d),d)

pr

include("sgdfunc.jl")
W,valid = SGDBabysitter(g, nn_predict::Function, maxIter, nHidden,
                        nParams, xtrain, ytrain, xvalid, yvalid )
