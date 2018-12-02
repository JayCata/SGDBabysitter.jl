# We use nHidden as a vector, containing the number of hidden units in each layer
using DataFrames
# Function that returns total number of parameters
function NeuralNet_nParams_class(d,nHidden,k)

	# Connections from inputs to first hidden layer
	nParams = d*nHidden[1]

	# Connections between hidden layers
	for h in 2:length(nHidden)
		nParams += nHidden[h-1]*nHidden[h]
	end

	# Connections from last hidden layer to output
	nParams += nHidden[end]*k

end

# Compute squared error and gradient
# for a single training example (x,y)
# (x is assumed to be a column-vector)
function NeuralNet_backprop_class(bigW,x,y,nHidden,k)
	d = length(x)
	nLayers = length(nHidden)

	#### Reshape 'bigW' into vectors/matrices
	W1 = reshape(bigW[1:nHidden[1]*d],nHidden[1],d)
	ind = nHidden[1]*d
	Wm = fill([],(nLayers-1))
	for layer in 2:nLayers
		Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*nHidden[layer-1]],nHidden[layer],nHidden[layer-1])
		ind += nHidden[layer]*nHidden[layer-1]
	end
	w = reshape(bigW[ind+1:end],k,nHidden[nLayers])

	### Bin y variables

	#### Define activation function and its derivative
	h(z) = tanh.(z)
	dh(z) = (sech.(z)).^2


	#### Forward propagation
	z = fill([],(nLayers))
	z[1] = W1*x
	for layer in 2:nLayers
		z[layer] = Wm[layer-1]*h(z[layer-1])
	end
	yhat,_ = findmax(w*h(z[end]))
	yhat=yhat-1


	f,r = softmax(w,x,y,k)

	#### Backpropagation
	dr = r
	err = dr

	# Output weights
	Gout = err*h(z[end])

	Gm = fill([],(nLayers-1))
	if nLayers > 1
		# Last Layer of Hidden Weights
		backprop = err*(dh(z[end]).*w)
		Gm[end] = backprop*h(z[end-1])'

		# Other Hidden Layers
		for layer in nLayers-2:-1:1
			backprop = (Wm[layer+1]'*backprop).*dh(z[layer+1])
			Gm[layer] = backprop*h(z[layer])'
		end

		# Input Weights
		backprop = (Wm[1]'*backprop).*dh(z[1])
		G1 = backprop*x'
	else
		# Input weights
		print(size(err),size(x'))
		G1 = err*(dh(z[1]).*w)*x'
	end

	#### Put gradients into vector
	g = zeros(size(bigW))
	g[1:nHidden[1]*d] = G1
	ind = nHidden[1]*d
	for layer in 2:nLayers
		g[ind+1:ind+nHidden[layer]*nHidden[layer-1]] = Gm[layer-1]
		ind += nHidden[layer]*nHidden[layer-1]
	end
	g[ind+1:end] = Gout

	return (f,g)
end

# Computes predictions for a set of examples X

function NeuralNet_predict_class(bigW,Xhat,nHidden,k)
	(t,d) = size(Xhat)
	nLayers = length(nHidden)

	#### Reshape 'bigW' into vectors/matrices
	W1 = reshape(bigW[1:nHidden[1]*d],nHidden[1],d)
	ind = nHidden[1]*d
	Wm = fill([],(nLayers-1))
	for layer in 2:nLayers
		Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*nHidden[layer-1]],nHidden[layer],nHidden[layer-1])
		ind += nHidden[layer]*nHidden[layer-1]
	end
	w = reshape(bigW[ind+1:end],k,nHidden[nLayers])

	#### Define activation function and its derivative
	h(z) = tanh.(z)
	dh(z) = (sech.(z)).^2

	#### Forward propagation on each example to make predictions
	yhat = zeros(t,1)
	for i in 1:t
		# Forward propagation
		z = fill([],(nLayers))
		z[1] = W1*Xhat[i,:]
		for layer in 2:nLayers
			z[layer] = Wm[layer-1]*h(z[layer-1])
		end
		yhat[i] = w'*h(z[end])
	end
	return yhat
end

function softmax(w,X,y,k)
	d=length(X)

	yunique=unique(y)
	f=0.0
	g=zeros(k,d)
	for i in 1:n
		print(yunique.==y[i])
		for j in length(yunique)
			if yunique[j]==y[i]
				s=j
			end
		end
		s=firstindex((yunique.==y[i]),1)
		print(s)
        wyi=transpose(w[s,:])
        f+=log(sum(exp.((w*X[i,:]))))-wyi*X[i,:]

        for r in 1:k
            if r==s
                indic=1.0
            else
                indic=0.0
			end

            for j in 1:d
                g[r,j]+=X[i,j]*((exp.((transpose(w[r,:])*X[i,:]))/sum(exp.((w*X[i,:]))))-indic)
	      	end
	  	end
	end


    return f,g
end
