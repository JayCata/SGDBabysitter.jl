#include("example_nnet.jl")
using LinearAlgebra
function ComputeValid(predict_func::Function, W, xvalid, yvalid, valid)
    yhat = predict_func(W,[ones(size(xvalid,1)) xvalid],nHidden)
    push!(valid,sum((yhat-yvalid).^2)/length(yvalid))

end

function BS_Select(;validation_array,f_array, gradcalc::Function,xtrain,ytrain,W,nHidden, alist, Blist)
    #Get most recent step and batch sizes
    a=alist[end]
    B=Blist[end]
    n,d=size(xtrain)

    #Get Gradient Angle
    i=rand(1:n,B)
    j=rand(1:n,B)
    _,gi=gradcalc(W, xtrain[i,:],ytrain[i],nHidden)
    _,gj=gradcalc(W,xtrain[j,:],ytrain[j], nHidden)
    costheta=dot(gi,gj)/(norm(gi)*norm(gj))
    #theta=acos(costheta)



    #alpha decreases
    if length(validation_array) >= 4 && validation_array[end-2]>validation_array[end]
        a_adjust = (1 - abs(validation_array[end-2] - validation_array[end])/validation_array[end-2])^(1/4)
        a = a*a_adjust
    end
    push!(alist,a)
    push!(Blist,B)
    print(a," ")


    return a,B
end

function SGDBabysitter(gradcalc::Function, nn_predict::Function, maxIter, nHidden,
                        nParams, xtrain, ytrain, xvalid, yvalid )

    n = size(xtrain,1)
    valid=[]
    flist=[]
    alist=[]
    Blist=[]
    #placehold for initialization
    B=100
    a = 1e-4
    push!(Blist,B)
    push!(alist,a)
    W= randn(nParams,1)

    for t in 1:maxIter

    	i = rand(1:n,B)
    	f,g = gradcalc(W, xtrain[i,:], ytrain[i], nHidden)
    	W = W - a*g
        #Every few iterations, compute validation and append f,g,validation

          if (mod(t-1,round(maxIter/100)) == 0)
              # print("Training iteration = $(t-1) ")
              push!(flist,f)
              ComputeValid(nn_predict, W, xvalid, yvalid, valid)
          end
        if (mod(t-1,round(maxIter/50)) == 0)
            a,B=BS_Select(validation_array=valid,f_array=flist, gradcalc=gradcalc,xtrain=xtrain,ytrain=ytrain,W=W,nHidden=nHidden, alist=alist, Blist=Blist)

          end
    end

    display(plot(1:length(valid), valid))

    return W, valid
end
