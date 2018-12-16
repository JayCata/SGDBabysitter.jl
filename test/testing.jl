using StatsBase
using Statistics
foo = NeuralNet_predict(wbSGDB,[ones(size(xvalid,1)) xvalid],nHidden)
foor = round.(foo)

bar2 = NeuralNet_predict(wbVan,[ones(size(xvalid,1)) xvalid],nHidden)
barr2 = round.(bar2)
# @show bar = countmap(foor)
