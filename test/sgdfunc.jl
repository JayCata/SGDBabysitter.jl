#include("example_nnet.jl")

function ComputeValid(predict_func::Function, W, xvalid, yvalid, valid)
    yhat = predict_func(W,[ones(size(xvalid,1)) xvalid],nHidden)
    push!(valid,sum((yhat-yvalid).^2)/length(yvalid))
    # push!(checks,j)
end

function BS_Select(val_array)
    return
end

function SGDBabysitter(gradcalc::Function, nn_predict::Function, maxIter, nHidden,
                        nParams, xtrain, ytrain, xvalid, yvalid )
    n = size(xtrain,1)
    # j=1
    # checks=[]
    valid=[]
    stepSize = 1e-4
    W= randn(nParams,1)

    for t in 1:maxIter
    	#global W, j
    	# The stochastic gradient update:
    	i = rand(1:n)
    	f,g = gradcalc(W, xtrain[i,:], ytrain[i], nHidden)
    	W = W - stepSize*g

    	# Every few iterations, plot the data/model:
    	# if (mod(t-1,round(maxIter/50)) == 0)
    	# 	print("Training iteration = $(t-1)",)
        #
    	# 	yhat = nn_predict(W,[ones(size(xvalid,1)) xvalid],nHidden)
        #
    	# 	push!(valid,sum((yhat-yvalid).^2)/length(yvalid))
    	# 	push!(checks,j)
    	# 	j=j+1
    	# end
          if (mod(t-1,round(maxIter/50)) == 0)
              print("Training iteration = $(t-1)",)
              ComputeValid(nn_predict, W, xvalid, yvalid, valid)
              # j=j+1
          end
    end
    display(plot(1:length(valid), valid))
    #plot(checks,valid)
    return W, valid
end
