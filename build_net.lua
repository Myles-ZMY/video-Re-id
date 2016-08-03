require 'nn'
require 'nngraph'
-- a easy way to build a raw rnn 
local build_net = {}
function build_net.rnn(input_size, rnn_size, output_size)
    local inputs = {}
    inputs[1] = nn.Identity()()
    inputs[2] = nn.Identity()() -- previous h

    local x = inputs[1]
    local pre_h = inputs[2]
    local ix = nn.Linear(input_size, rnn_size)(x)
    local hx = nn.Linear(rnn_size, rnn_size)(pre_h)
    local o = nn.CAddTable()({ix, hx})
    local h = nn.Tanh()(o)  -- this way according to the paper is not so common

    local outputs = {}
    table.insert(outputs, o)
    table.insert(outputs, h)

    return nn.gModule(inputs, outputs)
end

function build_net.cnn()
    local cnn_raw = nn.Sequential()
    cnn_raw:add(nn.SpatialConvolution(5, 16, 5, 5))
    cnn_raw:add(nn.SpatialMaxPooling(2, 2))
    cnn_raw:add(nn.Tanh())
    cnn_raw:add(nn.SpatialConvolution(16, 64, 5, 5))
    cnn_raw:add(nn.SpatialMaxPooling(2, 2))
    cnn_raw:add(nn.Tanh())
    cnn_raw:add(nn.SpatialConvolution(64, 64, 5, 5))
    cnn_raw:add(nn.SpatialMaxPooling(2,2))
    cnn_raw:add(nn.Tanh())
    cnn_raw:add(nn.View(64*16*4))
    cnn_raw:add(nn.Dropout(0.5))
    cnn_raw:add(nn.Linear(64*16*4, 128))

return rnn
end

return build_net





