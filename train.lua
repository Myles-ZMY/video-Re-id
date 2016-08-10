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
cmd:option('-learning_rate', 0.003, 'learning rate for training')
cmd:option('-gpuid', 0, 'which gpu to use, -1=use cpu')
cmd:option('-seed', 123, 'random number generator seed')
cmd:option('-checkpath', '', 'checkpoint path for save model parameters empty  =  this folder')

cmd:text()

local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

local data1 = hdf5.open(opt.images1, 'r')
local imgs1 = data1:read('/imgs')
local imgs1_size = imgs1:dataspaceSize()
print('load' .. imgs1_size[1] .. 'images from camera a')
local seq_length1 = data1:read('/seq_length'):all()
local index1 = data1:read('/index'):all()
local number1 = index1:size(1)
print(string.format('load %d persons from camera a', number1))
local data2 = hdf5.open(opt.images2, 'r')
local imgs2 = data2:read('/imgs')
local imgs2_size = imgs2:dataspaceSize()
print('load' .. imgs2_size[1] .. 'images from camera b')
local seq_length2 = data2:read('/seq_length'):all()
local index2 = data2:read('/index'):all()
local number2 = index2:size(1)
print(string.format('load %d persons form camera b', number2))

if opt.gpuid >= 0 then
    print('use gpu and cuda')
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
model.net:createClones()

function choose_frames(index, seq_length)
    local x1 = index
    local x2 = index + seq_length - 16
    local x = torch.random(x1, x2)
    return x, x+15
end


-- main Loop
local epoch = 1
while epoch <= opt.max_epoch do
    model.net:training()
    for i = 1,200 do
        local x = index1[i]
        local pos = index2[i]
        local j = torch.random(200)
        while j == i or seq_length2[j] < 16 do
            j = torch.random(200)
        end
        local neg = index2[j]
        local l1 = seq_length1[i]
        local l2 = seq_length2[i]
        local l3 = seq_length2[j]
        if l1 >16 and l2 > 16 then
            local xs, xe = choose_frames(x, l1)
            local ps, pe = choose_frames(pos, l2)
            local ns, ne = choose_frames(neg, l3)
            local train_imgs = {{}, {}, {}}
            for k = xs, xe do
                local ximg = imgs1:partial({k, k}, {1, 3}, {1, 128}, {1, 64})
                ximg = ximg:cuda()
                table.insert(train_imgs[1], ximg)
            end

            for k = ps, pe do
                local ximg = imgs2:partial({k, k}, {1, 3}, {1, 128}, {1, 64})
                ximg = ximg:cuda() 
                table.insert(train_imgs[2], ximg)
            end
            
            for k = ns, ne do
                local ximg = imgs2:partial({k, k}, {1, 3}, {1, 128}, {1, 64})
                ximg = ximg:cuda()
                table.insert(train_imgs[3], ximg)
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
            print(string.format('the %d person negative loss to person %d in epoch %d is: %f', i,j, epoch, loss2))
            model.net:zeroGradParameters()
            local dneg_feat = model.crit:backward(neg_feat, {label[1], label[2]})
            model.net:backward({example, neg_feat}, dneg_feat)
            grad = (grad + grad_params)/2

            params:add(-opt.learning_rate, grad)
        end
    end
        print('finish the training of epoch' .. epoch)
        if epoch % 50 == 0 then
            local checkpoint = model.net
            local checkpoint_path = path.join(opt.checkpath, 'trained_model')
            torch.save(checkpoint_path .. '.t7', checkpoint)
            print('wrote model parameters to' .. checkpoint_path .. '.t7')
        end
        epoch = epoch + 1
    
end

        

