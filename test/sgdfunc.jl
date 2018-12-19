using LinearAlgebra
using Plots
using Distributions
using Random

function VanillaSGD(gradcalc::Function, nn_predict::Function, maxIter, nHidden,nParams, xtrain, ytrain, xvalid, yvalid,a=.0001, B=1)
    print("Vanilla SGD Running...")
    n = size(xtrain,1)
    W= randn(nParams,1)
    flist=[]
    valid=[]
    wbest=zeros(nParams)
    vallow=Inf
    for t in 1:maxIter

    	i = Base.rand(1:n,B)
    	f,g = gradcalc(W, xtrain[i,:], ytrain[i], nHidden)
    	W = W - a*g

        if (mod(t-1,round(maxIter/100)) == 0)
            push!(flist,f)
            ComputeValid(nn_predict, W, xvalid, yvalid, valid)
            if valid[end]<vallow
                wbest=W
                vallow=valid[end]
            end
        end
    end
    display(plot(1:length(valid), valid))
    return wbest, vallow
end

function SGDBabysitter(gradcalc::Function, nn_predict::Function, maxIter, nHidden,nParams, xtrain, ytrain, xvalid, yvalid, maxbatch=500)

    n = size(xtrain,1)
    valid=[]
    flist=[]
    alist=[]
    Blist=[]

    a,B,W=BS_initialize(gradcalc, nn_predict, maxIter, nHidden,
                            nParams, xtrain, ytrain, xvalid, yvalid)
    push!(Blist,B)
    push!(alist,a)
    wbest=zeros(nParams)
    vallow=Inf
    triedDec=false
    for t in 1:maxIter

        idxshuffle = shuffle(1:n)
        i = rand(idxshuffle, B)
    	# i = rand(1:n,B)

    	f,g = gradcalc(W, xtrain[i,:], ytrain[i], nHidden)
    	W = W - a*g

        #Every few iterations, compute validation and append f,g,validation
        if (mod(t-1,round(maxIter/100)) == 0)
              push!(flist,f)
              ComputeValid(nn_predict, W, xvalid, yvalid, valid)
              if valid[end]<vallow
                  wbest=W
                  vallow = valid[end]
              end
        end
        if (mod(t-1,round(maxIter/50)) == 0)
            a,B,W,triedDec=BS_Select(validation_array=valid,f_array=flist, gradcalc=gradcalc,xtrain=xtrain,ytrain=ytrain,W=W,nHidden=nHidden, alist=alist, Blist=Blist,maxbatch=maxbatch, wbest=wbest,  maxIter=maxIter,triedDec=triedDec)

          end
    end

    display(plot(1:length(valid), valid))

    return wbest, vallow
end

#INITIALIZATION FUNCTION--------------------------------------------------------
function BS_initialize(gradcalc::Function, nn_predict::Function, maxIter, nHidden,nParams, xtrain, ytrain, xvalid, yvalid)
switch=1
n = size(xtrain,1)
B=5
a=rand(Uniform(.95,.9999999))
j=1
W_init = randn(nParams,1)
W = zeros(nParams,1)
while switch==1 && j<10
    W=W_init
    flist=[]
    validlist=[]
    for t in 1:10
        i = rand(1:n,B)
        f,g = gradcalc(W, xtrain[i,:], ytrain[i], nHidden)
        W = W - a*g
        push!(flist,f)
        ComputeValid(nn_predict, W, xvalid, yvalid, validlist)
    end
    if validlist[end]<(4/5)*validlist[1] && flist[end]<(4/5)*flist[1]
        switch=0
    else
        a=rand(Uniform(.5,.75))*a
        print("Initializing... ")
        j+=1
    end
end
return a,B,W
end

#Selection Function-------------------------------------------------------------

# bestyet = 1
function BS_Select(;validation_array,f_array, gradcalc::Function,xtrain,ytrain,W,nHidden, alist, Blist, maxbatch, wbest,  maxIter,triedDec)
    global ctr
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

    tol = 1e-5
    if costheta > 1
        costheta = 1 - tol
    elseif costheta < -1
        costheta = -1 + tol
    end
    theta = acos(costheta)

    #Case 1: alpha decrease if validation error does not

    if length(validation_array) >= 4 && ((validation_array[end-3] < validation_array[end]||validation_array[end-2] < validation_array[end]) && (validation_array[end-1]<validation_array[end])) || length(validation_array) >= 4 && ((mean(validation_array[end-3:end-1]) <= 1.05*validation_array[end]) && (mean(validation_array[end-3:end-1]) >= 0.95*validation_array[end]))
        if triedDec==false
            a=(3/4)*a
            triedDec = true
        elseif triedDec==true
            #go back to best w
            W = wbest
            a = (4/5)*a
            triedDec = false
        end

    #---- Alterative to case 1 ----#
    #Don't delete, might use later
    #Tries increasing alpha

    # decLimit = 5 #round(maxIter/20)
    # threshold = 12 #round(maxIter/10)
    #
    # if length(validation_array) >= 4 && ((validation_array[end-3] < validation_array[end]||validation_array[end-2] < validation_array[end]) && (validation_array[end-1]<validation_array[end])) || ((mean(validation_array[end-3:end-1]) <= 1.05*validation_array[end]) && (mean(validation_array[end-3:end-1]) >= 0.95*validation_array[end]))
    #     if ctr <= decLimit
    #         a = (4/5)*a
    #         ctr += 1
    #         print("case 2 part 1")
    #         print(" Ctr is :",ctr, " ")
    #     elseif ctr > decLimit && ctr <= threshold
    #         a = (6/4)*a
    #         ctr += 1
    #         print("case 2 part 2")
    #     elseif ctr > threshold
    #         W = wbest
    #         ctr = 0
    #         print("case 2 part 3")
    #     end


    #Case 2: alpha decreases based on validation error
    elseif length(validation_array) >= 4 && ((validation_array[end-3] < validation_array[end]||validation_array[end-2] < validation_array[end]) && (validation_array[end-1]<validation_array[end]))
        a_adjust = (1 - abs(validation_array[end-2] - validation_array[end])/validation_array[end-2])
        a = a*a_adjust
        print("case 1")
    end

    #case 3 batchsize as a function of theta
    if exp(-.5/(pi-theta))==NaN
        B=maxbatch
    else
        B=convert(Int64,round(maxbatch*(1/(1+round(exp(-.5/(pi-theta+.00001)),digits=5)))))
    end
    push!(alist,a)
    push!(Blist,B)
    print("Step Size: ", a," ","Batch Size: ", B, " ")
    return a,B,W,triedDec
end

#Compute Validation-------------------------------------------------------------
function ComputeValid(predict_func::Function, W, xvalid, yvalid, valid)
    yhat = predict_func(W,[ones(size(xvalid,1)) xvalid],nHidden)
    push!(valid,sum((yhat-yvalid).^2)/length(yvalid))
 end
