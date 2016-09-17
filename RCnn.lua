require 'nn'
require 'nngraph'

local RCnn = {}
function RCnn.buildNet(input_size, rnn_size, output_size)

    local inputs = {}
    inputs[1] = nn.Identity()() -- input image frame
    inputs[2] = nn.Identity()() -- previous h
    
    -- cnn model 
    local x = inputs[1]
    local c1 = nn.SpatialConvolution(3, 16, 5, 5)(x)
    local p1 = nn.SpatialMaxPooling(2, 2)(c1)
    local t1 = nn.Tanh()(p1)
    local c2 = nn.SpatialConvolution(16, 64, 5, 5)(t1)
    local p2 = nn.SpatialMaxPooling(2, 2)(c2)
    local t2 = nn.Tanh()(p2)
    local c3 = nn.SpatialConvolution(64, 64, 5, 5)(t2)
    local p3 = nn.SpatialMaxPooling(2, 2)(c3)
    local t3 = nn.Tanh()(p3)
    local fc_input = nn.View(64*12*4)(t3)
    local dropout = nn.Dropout(0.5)(fc_input)
    local fc_output = nn.Linear(64*12*4, input_size)(dropout)
    
    -- rnn model
    local pre_h = inputs[2]
    local ix = nn.Linear(input_size, rnn_size)(fc_output)
    local hx = nn.Linear(rnn_size, rnn_size)(pre_h)
    local o = nn.CAddTable()({ix, hx})
    local h = nn.Tanh()(o)  -- this way according to the paper is not so common

    local outputs = {}
    table.insert(outputs, o)
    table.insert(outputs, h)

    return nn.gModule(inputs, outputs)
end


return RCnn





