require 'torch'
require 'nn'
require 'rnn'
require 'optim'

torch.setdefaulttensortype('torch.FloatTensor')

require 'gnuplot'

cmd = torch.CmdLine()
cmd:option('--dataPath', 'data', 'data path')
cmd:option('--trainingSetFraction', 0.7, 'how many of data will be used as a train set')
cmd:option('--rho', 70, 'rho parameter used in BTT in LSTM')
cmd:option('--outputSizeI', 50, 'size of the first layer of LSTM')
cmd:option('--outputSizeII', 50, 'size of the second layer of LSTM')
cmd:option('--MLPsize', 40, 'size of hidden layer of MLPs used to predict final output based on the second LSTM output')
cmd:option('--dropoutProb', 0.5, 'dropout probability used in MLPs')
cmd:option('--uniform', 0.08, 'if greater than zero all parameters of LSTM model will be drawn from uniform distriution - the larger value, the higher variance of initial parameters')
cmd:option('--gradClip', 5, 'if greater than zero gradients will be cliped and absolut value of each of them will be no larger than the value provided')
cmd:option('--seed', 881010, 'seed used to initiate LSTM model')
cmd:option('--batchSize', 20, 'batch size')
cmd:option('--learningRate', 5e-3, 'learning rate')
cmd:option('--vE', 3, 'how often validation is performed')
cmd:option('--patience', 20, 'how many validation without progress make the learning procedure stop')
cmd:option('--mask', 2, 'if greater than one target gradients will be masked to regularize LSTM model - the larger value, the more sparse target gradients')
local opt = cmd:parse(arg or {})
--trick that makes drawer.lua knows that comand line parameters are prepared for trainer.lua
torch.cmdLineRead = {}

print(opt)

local preparer = require 'preparer.lua'
local drawer = require 'drawer.lua'

--we assume that each row of the data represents a single letter and trajectories has a fixed length (70), hence the row is as follows: target (1 value), x positions (70 values), y positions (70 values)
local length = 70
--we also assume that we have just pairs (x,y) of consecutive pen positions (it may be extended by additing pressure, tiltation etc.)
local inputSize = 2

--load and prepare a data
local train, valid, dataRaw, normParams = preparer.loadAndPrepareDatasets(opt.dataPath, length, opt.trainingSetPercent)

--print an example of a letter
drawer.draw(dataRaw, 1, length)

--define model
torch.manualSeed(opt.seed)

local lstm = nn.Sequential()
lstm:add(nn.Sequencer(nn.FastLSTM.maskZero(nn.FastLSTM(inputSize, opt.outputSizeI, opt.rho),1)))
lstm:add(nn.Sequencer(nn.FastLSTM.maskZero(nn.FastLSTM(opt.outputSizeI, opt.outputSizeII, opt.rho),1)))
    letter = nn.Sequential()
    letter:add(nn.MaskZero(nn.Linear(opt.outputSizeII, opt.MLPsize),1))
    letter:add(nn.Dropout(opt.dropoutProb))
    letter:add(nn.ReLU())
    letter:add(nn.MaskZero(nn.Linear(opt.MLPsize, 26),1))
    letter:add(nn.MaskZero(nn.LogSoftMax(),1))
lstm:add(nn.Sequencer(letter))
lstm:remember('both')

if opt.uniform > 0 then
    for k,param in ipairs(lstm:parameters()) do
        param:uniform(-opt.uniform, opt.uniform)
    end
end

--define criterion
local c = nn.SequencerCriterion(nn.ClassNLLCriterion())

--define batch iterator
function batch(data, batchSize, inputSize)
    local spliter = nn.SplitTable(3)
    local startingIndex = 1
    
    local targets = {}
    local target1 = torch.Tensor(batchSize)
    for step=1,(data:size(2) - 1)/inputSize do
        targets[step] = target1
    end

    return function ()
        if startingIndex <= data:size(1) then
            local bs = math.min(batchSize, data:size(1) - startingIndex + 1)

            local transferredData = data[{{startingIndex,startingIndex+bs-1},{2,data:size(2)}}]:clone()
            transferredData:resize(bs, inputSize, (data:size(2) - 1)/inputSize)
            local inputs = spliter:forward(transferredData)

            if bs == batchSize then
                target1:copy(data[{{startingIndex,startingIndex+bs-1},{1}}])
            else
                --we have to prepare new targets table for last batch
                local target1 = data[{{startingIndex,startingIndex+bs-1},{1}}]:clone()
                for step=1,(data:size(2) - 1)/inputSize do
                    targets[step] = target1
                end
            end

            startingIndex = startingIndex + bs
            
            --[[
            if data:size(1) > 400 then
                local leakSize = 7
                local leakyInputs = {}
                local leaks = torch.randperm(length - 1)[{{1,leakSize}}]
                for i=1,#inputs do
                    if leaks:eq(i):sum() == 0 then
                        table.insert(leakyInputs, inputs[i])
                    end
                end
                inputs = leakyInputs
                
                for i=length-leakSize,length - 1 do
                    targets[i] = nil
                end
            end
            --]]

            return inputs, targets
        end
    end
end

--define auxillary stuff
function round(x, digits)
    digits = digits or 2
    return math.floor(10^digits * x + 0.5)/10^digits
end

local remainingGrads = {}
if opt.mask > 1 then
    local lag = length - 1
    local usedGrads = ""
    while true do
        lag = math.min(lag/opt.mask, lag - 1)
        table.insert(remainingGrads, math.floor(lag))
        if lag < 1 then
            break
        end
        usedGrads = usedGrads.. math.floor(lag) .. ", "
    end
    if #usedGrads > 0 then
        print(#remainingGrads.." out of " .. length-1 .. " target gradients will be used in BPTT.")
        print("Counting from the end (zero is the last one) " .. usedGrads .. "and 0 are the only positions where criterion will calculate gradient to fund the flow of BPTT.")
    else
        print("Criterion will calculate gradient only after last one point of trajectory and BPTT through the whole sequence.")
    end
    remainingGrads = torch.Tensor(remainingGrads)
else
    remainingGrads = nil
end

--define learning procedure
function train_epoch(lstm, data, batchSize, inputSize, config, state, remainingGrads)
    lstm:training()
    lstm:forget()
    params, gradParams = lstm:getParameters()

    local gerr = 0
    local gotIt = 0
    local timer = torch.Timer()
    local sum = torch.Tensor(26,26):zero()
    
    local shuffle = torch.randperm(data:size(1))
    data = data:index(1,shuffle:long())

    local it = 0
    for inputs, targets in batch(data, batchSize, inputSize) do
        if it + 1 > data:size(1)/batchSize then
            print("There were "..it..
                " batches, time needed for one batch "..round(timer:time().real/it)..
                "s so "..round(timer:time().real).."s for epoch.")
            print("Train error: "..gerr/data:size(1))
            break
        end
        it = it + 1

        --feval
        local f = function(x)
            if x ~= params then
                params:copy(x)
            end

            lstm:zeroGradParameters()
            local outputs = lstm:forward(inputs)
            
            for i=1,inputs[1]:size(1) do
                sum[{{targets[1][{i}]},{}}]:add(outputs[#outputs][{{i},{}}]:clone():exp())
            end
            
            local _, candidates = outputs[#outputs][{{},{}}]:kthvalue(26)
            gotIt = gotIt + candidates:add(-1, targets[1]:long()):eq(0):sum()

            local err = c:forward(outputs, targets)
            local gradOutputs = c:backward(outputs, targets)
            
            if remainingGrads then
                for i=1,#gradOutputs do
                    if remainingGrads:eq(#gradOutputs - i):sum() == 0 then
                        gradOutputs[i]:zero()
                    end
                end
            end

            lstm:backward(inputs, gradOutputs)
            
            if opt.gradClip > 0 then
                gradParams:clamp(-opt.gradClip,opt.gradClip)
            end

            gerr = gerr + err

            return err, gradParams
        end
        optim.adam(f, params, config, state)
        lstm:forget()
    end
    
    print("Fraction of letters caught: "..round(gotIt/data:size(1),3))
    
    return sum:clone():cdiv(sum:sum(2):expand(26,26):add(1e-6)):cmul(torch.eye(26)):sum()/26, sum, gotIt/data:size(1)
end

--define validation procedure
function validate(lstm, data, batchSize, inputSize)
    lstm:evaluate()

    local gerr = 0
    local timer = torch.Timer()

    local sum = torch.Tensor(26,26):zero()
    local gotIt = 0

    local it = 0
    for inputs, targets in batch(data, batchSize, 2) do
        if it + 1 > data:size(1)/batchSize then
            print("There are "..round(timer:time().real).."s needed for validation.")
            print("Valid error: "..gerr/data:size(1))
            break
        end
        it = it + 1

        lstm:forget()
        local outputs = lstm:forward(inputs)

        for i=1,inputs[1]:size(1) do
            sum[{{targets[1][{i}]},{}}]:add(outputs[length-1][{{i},{}}]:clone():exp())
        end

        local _, candidates = outputs[length-1][{{},{}}]:kthvalue(26)
        gotIt = gotIt + candidates:add(-1, targets[1]:long()):eq(0):sum()
        
        gerr = gerr + c:forward(outputs, targets)
    end

    print("Fraction of letters caught: "..round(gotIt/data:size(1),3))

    return sum:clone():cdiv(sum:sum(2):expand(26,26):add(1e-6)):cmul(torch.eye(26)):sum()/26, sum, gotIt/data:size(1)
end

--train LSTM model
local counts
local trainErrors = {}
local validErrors = {}
local trainCatches = {}
local validCatches = {}

local config = {
    learningRate = opt.learningRate
}
local state = {}

local epoch = 1
while true do
    print(epoch .. '.')
    local score, sum, catch = train_epoch(lstm, train, opt.batchSize, inputSize, config, state, remainingGrads)
    table.insert(trainErrors, 100*score)
    table.insert(trainCatches, 100*catch)
    print("Score: "..score.."\n")

    if epoch % opt.vE == 1 then
        print("Validate:")
        local score, _, catch = validate(lstm, valid, opt.batchSize, inputSize)
        print("Score: "..score.."\n")
        table.insert(validErrors, 100*score)
        table.insert(validCatches, 100*catch)
    end
    
    local validErr = torch.Tensor(validErrors)
    gnuplot.figure(0)
    gnuplot.plot({{'training score', torch.range(1,epoch),torch.Tensor(trainErrors), '+-'},
        {'validation score', torch.range(1,math.ceil(epoch/opt.vE)):mul(opt.vE):add(1-opt.vE),validErr, '+-'},
        {'training catches', torch.range(1,epoch),torch.Tensor(trainCatches), '+-'},
        {'validation catches', torch.range(1,math.ceil(epoch/opt.vE)):mul(opt.vE):add(1-opt.vE),torch.Tensor(validCatches), '+-'}})
    gnuplot.title("Learning curve")
    gnuplot.movelegend('right','bottom')
    gnuplot.grid(true)
    
    if validErr[{{#validErrors}}]:max() >= validErr:max() then
        lstm:clearState()
        torch.save("model"..opt.seed..".m", {lstm = lstm, normParams = normParams}, 'binary')
    end

    if #validErrors > opt.patience and validErr[{{#validErrors-opt.patience+1, #validErrors}}]:max() < validErr:max() then
        print("Any progress in last "..opt.patience.." validations. Learning procedure done.")
        print("Distribution of letters:")
        counts = sum:sum(2):add(0.5):long():reshape(26)
        for i=1,26 do
            print(string.char(i + 96), counts[{i}])
        end
        break
    end

    epoch = epoch + 1
end

--load the mode with the best score on validation set
local model = torch.load("modelAA"..opt.seed..".m", 'binary')

--draw an example of a letter with trajectories of class probabilities
drawWithProbabilities(dataRaw, model, 826, length, 7)