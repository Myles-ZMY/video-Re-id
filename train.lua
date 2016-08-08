require 'torch'
require 'nn'
require 'nngraph'
require 'hdf5'
require 'ReIdModel'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a video-based Re-Identification net')
cmd:text()
cmd:text('Options')

cmd:option('-images1', './reid_data/data1.h5', 'path to image data from camera a')
cmd:option('-images2', './reid_data/data2.h5', 'path to image data from camera b')
cmd:option('-rnn_size', 128, 'rnn hidden ecoding size')
cmd:option('-input_size', 128, 'image encoding size')

cmd:option('-max_epoch', 500, 'max number of epoch')
cmd:option('learning_rate', 1e-3, 'learning rate for training')
cmd:option('gpuid', 0, 'which gpu to use, -1=use cpu')
cmd:option('seed', 123, 'random number generator seed')

cmd:text()

local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensorType('torch.FloatTensor')

local data1 = hdf5.open(opt.images1, 'r')
local imgs1 = data1:read('/imgs')
local seq_length1 = data1:read('/seq_length'):all()
local index1 = data1:read('/index'):all()
local data2 = hdf5:open(opt.images2, 'r')
local imgs2 = data2:read('/imgs')
local seq_length2 = data2:read('/seq_length'):all()
local index2 = data2:read('/index'):all()

if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opt.seed)
    cutorch.setDevice(opt.gpuid + 1)
end

local model = {}
local m_params = {}
m_params.img_feat_len = opt.input_size
m_params.rnn_size = opt.rnn_size
m_params.output_size = opt.rnn_size
model.net = nn.ReIdModel(m_params)
model.crit = nn.ReIdCriterion()

if opt.gpuid >= 0 then  -- for gpu
    for k,v in pairs(model) do v:cuda() end
end

local params, grad_params = model.net:getParameters()
model.net:createClone()

function choose_frames(seq_length)
    local x1 = 1
    local x2 = seq_length - 15
    local x = torch.random(x1, x2)
    return x, x+16
end


-- main Loop
local epoch = 1
while epoch <= opt.max_epoch do
    model.net:training()
    local epoch = 0
    for i = 1,200 do
        local x = index1[i]
        local pos = index2[i]
        local j = 1
        while j == i do
            j = torch.random(200)
        end
        local neg = index2[j]
        local l1 = seq_length1[x]
        local l2 = seq_length2[pos]
        local l3 = seq_length2[neg]
        local xs, xe = choose_frames(l1)
        local ps, pe = choose_frames(l2)
        local ns, ne = choose_frames(l3)
        local train_set = {{xs,xe}, {ps,pe}, {ns,ne}}
        local train_imgs = {}
        for k,v in pairs(train_set) do
            local n = v[1]
            local ximgs = {}
            while n <= v[2] do
                local ximg = imgs:partial({n, n}, {1, 3}, {1, 128}, {1, 64})
                n = n + 1
                table.insert(ximgs, ximg)
            end
            table.insert(train_imgs, ximgs)
        end
        local example = train_imgs[1]
        local pos_seq = train_imgs[2]
        local neg_seq = train_imgs[3]
        local label = {}
        table.insert(label, i)
        table.insert(label, j)

        -- a train with postive sequence 
        local pos_feat = model.net:forward({example, pos_seq})
        local loss1 = model.crit:forward(pos_feat, {label[1], label[1]})
        print(string.format('the %d person postive loss in epoch %d is: %f', i, epoch, loss1))
        model.net:zeroGradParameters()
        local dpos_feat = model.crit:backward(pos_feat, {label[1], label[1]})
        model.net:backward({example, pos_seq}, dpos_feat)
        local grad = 0
        grad = grad + grad_params

        -- a train with negative sequence
        local neg_feat = model.net:forward({example, neg_seq})
        local loss2 = model.crit:forward(neg_feat, {label[1], label[2]})
        print(string.format('the %d person negative loss in epoch %d is: %f', i, epoch, loss2))
        model.net:zeroGradParameters()
        local dneg_feat = model.crit:backward(neg_feat, {label[1], label[2]})
        model.net:backward({example, neg_feat}, dneg_feat)
        grad = (grad + grad_params)/2

        params:add(-opt.learning_rate, grad)
    end
    epoch = epoch + 1
end


        

