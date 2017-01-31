require 'torch'
require 'nn'
require 'rnn'

torch.setdefaulttensortype('torch.FloatTensor')

require 'gnuplot'

local cmd = torch.CmdLine()
cmd:option('--run', false, 'to draw a letter')
cmd:option('--runProb', false, 'to draw a letter with trajectories of probabilities')
cmd:option('--i', 865, 'instance number')
cmd:option('--l', 70, 'data length')
cmd:option('--t', 7, 'number of trajectories for top competitors')
cmd:option('--d', 'data', 'data path')
cmd:option('--m', 'm.m', 'model path (lstm model and normalization parameters)')
cmd:option('--r', false, 'if you want to pick random instance')
cmd:option('--rmin', 701, 'if r - minimum instance number')
cmd:option('--rmax', 872, 'if r - maximum instance number')
local opt = torch.cmdLineRead or cmd:parse(arg or {})

local preparer = require 'preparer.lua'

local drawer = {}

function draw(dataRaw, index, length, path)
    if dataRaw == nil then
        assert(path ~= nil, 'You have to provide dataRaw or path in order to load it')
        dataRaw = preparer.loadRawData(path, length)
    end

    local x = torch.Tensor(length)
    local y = torch.Tensor(length)

    local x_ffi = torch.data(x)
    local y_ffi = torch.data(y)
    local data_ffi = torch.data(dataRaw)

    for i=1,length do
        x_ffi[i-1] = data_ffi[(2*length+1)*(index-1) + i]
        y_ffi[i-1] = -data_ffi[(2*length+1)*(index-1) + length + i]
    end

    gnuplot.figure(1)
    gnuplot.raw('set size ratio -1')
    gnuplot.raw('set format x ""')
    gnuplot.raw('set format y ""')
    gnuplot.raw('set key outside')
    gnuplot.title("An example of letter " .. string.char(data_ffi[(2*length+1)*(index-1)] + 96))
    gnuplot.plot({{x,y, '+'},
            {'The child is starting', x[{{1,length/3}}],y[{{1,length/3}}], 'with points ps 0.2'},
            {'The child is about to finish', x[{{length*2/3,length}}],y[{{length*2/3,length}}], 'with points ps 0.2'},
            {'The child is in the middle', x[{{length*1/3,2/3*length}}],y[{{length*1/3,2/3*length}}], 'with points ps 0.2'}
        })
end
drawer.draw = draw

function drawWithProbabilities(dataRaw, model, index, length, top, dataPath, modelPath)
    if dataRaw == nil then
        assert(dataPath ~= nil, 'You have to provide dataRaw or path in order to load it')
        dataRaw = preparer.loadRawData(dataPath, length)
    end
    draw(dataRaw, index, length)
    
    if model == nil then
        assert(modelPath ~= nil, 'You have to provide lstm model and normalziation parameters, or path in order to load it')
        model = torch.load(modelPath, 'binary')
    end

    lstm = model.lstm
    normParams = model.normParams
    
    lstm:evaluate()
    lstm:forget()
    
    local data = preparer.differentiateData(dataRaw[{{index},{}}], length)
    preparer.normalize(data, length, normParams)

    local transferredData = data[{{},{2,2*length-1}}]:clone():resize(1, 2, length-1)
    local inputs = nn.SplitTable(3):forward(transferredData)
    local outputs = lstm:forward(inputs)
    tops = {}
    for i=1,top do
        _, index = outputs[#outputs]:squeeze():kthvalue(27 - i)
        tops[i] = index[1]
    end
    
    local x = torch.Tensor(#outputs, top)
    for i=1,#outputs do
        outputs[i]:exp()
        for j=1,top do
            x[{{i},{j}}] = outputs[i][{{1},{tops[j]}}]
        end
    end
    gnuplot.figure(2)
    local ploter = {}
    for i=1,top do
        ploter[i] = {string.char(tops[i] + 96), x[{{},{i}}], '+'}
    end
    gnuplot.movelegend('left','top')
    gnuplot.plot(ploter)
end
drawer.drawWithProbabilities = drawWithProbabilities

if opt.run or opt.runProb then
    if opt.r then
        opt.i = torch.random()%(opt.rmax - opt.rmin) + opt.rmin
        print(opt.i)
    end
    if opt.run then
        draw(nil, opt.i, opt.l, opt.t, opt.d)
    else
        drawWithProbabilities(nil, nil, opt.i, opt.l, opt.t, opt.d, opt.m)
    end
end

return drawer