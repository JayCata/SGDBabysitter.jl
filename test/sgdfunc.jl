using LinearAlgebra
using Plots
using Distributions
function VanillaSGD(gradcalc::Function, nn_predict::Function, maxIter, nHidden,
                        nParams, xtrain, ytrain, xvalid, yvalid,a=.0001, B=1)
    print("Vanilla SGD Running...")
    n = size(xtrain,1)
    W= randn(nParams,1)
    flist=[]
    valid=[]
    wbest=zeros(nParams)
    vallow=Inf
    for t in 1:maxIter

    	i = rand(1:n,B)
    	f,g = gradcalc(W, xtrain[i,:], ytrain[i], nHidden)
    	W = W - a*g

        if (mod(t-1,round(maxIter/100)) == 0)
            # print("Training iteration = $(t-1) ")
            push!(flist,f)
            ComputeValid(nn_predict, W, xvalid, yvalid, valid)
            if valid[end]<vallow
                wbest=W
            end
        end
    end
    display(plot(1:length(valid), valid))
    return wbest, valid
end

function SGDBabysitter(gradcalc::Function, nn_predict::Function, maxIter, nHidden,
                        nParams, xtrain, ytrain, xvalid, yvalid, maxbatch=500)

    n = size(xtrain,1)
    valid=[]
    flist=[]
    alist=[]
    Blist=[]

    a,B=BS_initialize(gradcalc, nn_predict, maxIter, nHidden,
                            nParams, xtrain, ytrain, xvalid, yvalid)
    push!(Blist,B)
    push!(alist,a)
    W= randn(nParams,1)
    wbest=zeros(nParams)
    vallow=Inf

    for t in 1:maxIter

    	i = rand(1:n,B)
    	f,g = gradcalc(W, xtrain[i,:], ytrain[i], nHidden)
    	W = W - a*g
        #Every few iterations, compute validation and append f,g,validation

          if (mod(t-1,round(maxIter/100)) == 0)
              # print("Training iteration = $(t-1) ")
              push!(flist,f)
              ComputeValid(nn_predict, W, xvalid, yvalid, valid)
              if valid[end]<vallow
                  wbest=W
              end
          end
        if (mod(t-1,round(maxIter/50)) == 0)
            a,B=BS_Select(validation_array=valid,f_array=flist, gradcalc=gradcalc,xtrain=xtrain,ytrain=ytrain,W=W,nHidden=nHidden, alist=alist, Blist=Blist,maxbatch=maxbatch)

          end
    end

    display(plot(1:length(valid), valid))

    return wbest, valid
end



#INITIALIZATION FUNCTION--------------------------------------------------------
function BS_initialize(gradcalc::Function, nn_predict::Function, maxIter, nHidden,
                        nParams, xtrain, ytrain, xvalid, yvalid)
switch=1
n = size(xtrain,1)
B=5
a=rand(Uniform(.95,.9999999))
j=1
while switch==1 && j<10
    W= randn(nParams,1)
    flist=[]
    validlist=[]
for t in 1:10
    i = rand(1:n,B)
    f,g = gradcalc(W, xtrain[i,:], ytrain[i], nHidden)
    W = W - a*g
    push!(flist,f)
    ComputeValid(nn_predict, W, xvalid, yvalid, validlist)
end
if validlist[end]<(4/6)*validlist[1] && flist[end]<(4/5)flist[1]
    switch=0
else
    a=rand(Uniform(.5,.75))*a
    print("Initializing... ")
    j+=1
end
end
return a,B
end
#Selection Function-------------------------------------------------------------
function BS_Select(;validation_array,f_array, gradcalc::Function,xtrain,ytrain,W,nHidden, alist, Blist, maxbatch)
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
    if costheta>0
    theta=acos(costheta-.001)
    else
    theta=acos(costheta+.001)
    end


    #case 1 alpha decreases based on validation error
    if length(validation_array) >= 4 && validation_array[end-2]>validation_array[end]
        a_adjust = (1 - abs(validation_array[end-2] - validation_array[end])/validation_array[end-2])^2
        a = a*a_adjust
    end

    #case 2 alpha decrease if validation error does not
    if length(validation_array) >= 4 && (validation_array[end-4] < validation_array[end]||validation_array[end-3] < validation_array[end]) && (validation_array[end-2]<validation_array[end])
        a=(3/4)*a
    end

    #case 3 batchsize as a function of theta
    if exp(-1/(pi-theta))==NaN
        B=maxbatch
    else
    B=convert(Int64,round(maxbatch*(1/(1+5*exp(-1/(pi-theta))))))
    end
    push!(alist,a)
    push!(Blist,B)
    print("Step Size: ", a," ","Batch Size: " B)
    return a,B
end
#Compute Validation-------------------------------------------------------------
function ComputeValid(predict_func::Function, W, xvalid, yvalid, valid)
    yhat = predict_func(W,[ones(size(xvalid,1)) xvalid],nHidden)
    push!(valid,sum((yhat-yvalid).^2)/length(yvalid))
 end
