require 'torch';
torch.setdefaulttensortype('torch.FloatTensor')

local preparer = {}

function loadRawData(path, length)
    local storage = torch.FloatStorage('data')
    local dataRaw = torch.FloatTensor(storage):resize(storage:size()/(1 + 2*length), (1 + 2*length))
    dataRaw[{{},{1}}]:add(-dataRaw[{{},{1}}]:min() + 1)
    
    return dataRaw
end
preparer.loadRawData = loadRawData

function differentiateData(dataRaw, length)
    --calcualte differences
    local data = torch.Tensor(dataRaw:size(1), dataRaw:size(2) - 2)
    data[{{},{1}}] = dataRaw[{{},{1}}]
    data[{{},{2,length}}] = dataRaw[{{},{3,length + 1}}] - dataRaw[{{},{2,length}}]
    data[{{},{length + 1,2*length - 1}}] = dataRaw[{{},{length + 3,2*length + 1}}] - dataRaw[{{},{length + 2,2*length}}]

    return data, dataRaw
end
preparer.differentiateData = differentiateData

function loadAndDifferentiateData(path, length)
    local dataRaw = loadRawData(path, length)
    return differentiateData(dataRaw, length)
end

function extractTrainingSetAndNormalizingParams(data, length, trainPercent)
    local trainSize = math.ceil(data:size(1) * trainPercent)
    
    assert(trainPercent > 0, 'train percent has to be positive')
    assert(trainSize+1 < data:size(1), 'train percent is too large')
    
    local train = data[{{1,trainSize},{}}]
    local normParams = {}
    normParams.xmean = train[{{},{2,length}}]:mean()
    normParams.ymean = train[{{},{length + 1,2*length - 1}}]:mean()
    normParams.xstd = train[{{},{2,length}}]:std()
    normParams.ystd = train[{{},{length + 1,2*length - 1}}]:std()

    local valid = data[{{trainSize+1, data:size(1)},{}}]
    
    return train, valid, normParams
end

function normalize(data, length, normParams)
    data[{{},{2,length}}]:add(-normParams.xmean):div(normParams.xstd)
    data[{{},{length + 1,2*length - 1}}]:add(-normParams.ymean):div(normParams.ystd)
end
preparer.normalize = normalize

function loadAndPrepareDatasets(path, length, trainPercent)
    local trainPercent = trainPercent or 0.7

    local data, dataRaw = loadAndDifferentiateData(path, length)
    local train, valid, normParams = extractTrainingSetAndNormalizingParams(data, length, trainPercent)
    normalize(train, length, normParams)
    normalize(valid, length, normParams)
    
    return train, valid, dataRaw, normParams
end
preparer.loadAndPrepareDatasets = loadAndPrepareDatasets

return preparer