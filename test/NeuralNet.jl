# We use nHidden as a vector, containing the number of hidden units in each layer

# Function that returns total number of parameters
function NeuralNet_nParams(d,nHidden)

	# Connections from inputs to first hidden layer
	nParams = d*nHidden[1]

	# Connections between hidden layers
	for h in 2:length(nHidden)
		nParams += nHidden[h-1]*nHidden[h]
	end

	# Connections from last hidden layer to output
	nParams += nHidden[end]

end

# Compute squared error and gradient
# for a single training example (x,y)
# (x is assumed to be a column-vector)
function NeuralNet_backprop(bigW,x,y,nHidden)

	if typeof(x)<:Array{Float64,1}
		d=length(x)
	else
		n,d = size(x)
	end
	nLayers = length(nHidden)

	#### Reshape 'bigW' into vectors/matrices
	W1 = reshape(bigW[1:nHidden[1]*d],nHidden[1],d)
	ind = nHidden[1]*d
	Wm = fill([],(nLayers-1))
	for layer in 2:nLayers
		Wm[layer-1] = reshape(bigW[ind+1:ind+nHidden[layer]*nHidden[layer-1]],nHidden[layer],nHidden[layer-1])
		ind += nHidden[layer]*nHidden[layer-1]
	end
	w = bigW[ind+1:end]

	#### Define activation function and its derivative
	h(z) = (tanh.(z))
	dh(z) = (sech.(z)).^2


	#### Forward propagation
	z = fill([],nLayers)
	
	z[1] = W1*x
	for layer in 2:nLayers
		z[layer] = Wm[layer-1]*h(z[layer-1])
	end

	yhat = transpose(w)*h(z[end])

	r = yhat-y
	f = (1/2)r^2

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
function NeuralNet_predict(bigW,Xhat,nHidden)
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
	w = bigW[ind+1:end]

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
