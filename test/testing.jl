using StatsBase
using Statistics

include("NeuralNet.jl")
foo = NeuralNet_predict(wbSGDBb,[ones(size(xvalid,1)) xvalid],nHidden)
fooR = round.(foo)
fooU = sort(unique(foor))

egg = NeuralNet_predict(wbVanb,[ones(size(xvalid,1)) xvalid],nHidden)
eggR = round.(bar2)
eggU = sort(unique(eggR))

bar1 = NeuralNet_predict(wbSGDB,[ones(size(xvalid,1)) xvalid],nHidden)
barr = round.(bar1)
barrU = sort(unique(barr))

bar2 = NeuralNet_predict(wbVan,[ones(size(xvalid,1)) xvalid],nHidden)
barr2 = round.(bar2)
barr2U = sort(unique(barr2))

x = 1
function abc()
    # x = 1
    for i in 1:5
        global x
        x = i
    end
    print(x)
end

#---------------
medyhat= NeuralNet_predict(wbSGDB,[ones(size(xvalid,1)) xvalid],nHidden)
medyhat_u = length(unique(medyhat))

medyhatv = NeuralNet_predict(bestw,[ones(size(xvalid,1)) xvalid],nHidden)
medyhatv_u = length(unique(medyhatv))
