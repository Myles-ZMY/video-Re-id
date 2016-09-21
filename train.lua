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

cmd:option('-image_data', './reid_data/data_train.h5', 'image data path for training')
cmd:option('-seg_index', './reid_data/seg_index.h5', 'segmentation index hdf5 file')
cmd:option('-rnn_size', 128, 'rnn hidden ecoding size')
cmd:option('-input_size', 128, 'image encoding size')
cmd:option('-max_seq_len', 16, 'the max length of choosed frames')

cmd:option('-max_epoch', 500, 'max number of epoch')
cmd:option('-learning_rate', 0.001, 'learning rate for training')
cmd:option('-gpuid', 1, 'which gpu to use, -1=use cpu')
cmd:option('-seed', 123, 'random number generator seed')
cmd:option('-checkpath', '', 'checkpoint path for save model parameters empty  =  this folder')
cmd:option('-start_from', '', 'start training from a extent model,empty = start from scratch')
cmd:option('-model_name', 'trained_model', 'name for saved model')

cmd:text()

local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

local train_data = hdf5.open(opt.image_data, 'r')
local imgs1 = train_data:read('/imgs1')
local imgs1_size = imgs1:dataspaceSize()
print('load' .. imgs1_size[1] .. 'images from camera a')
local seq_length1 = train_data:read('/seq_length1'):all()
local index1 = train_data:read('/index1'):all()
local number1 = index1:size(1)
assert(seq_length1:size(1) == index1:size(1), 'the amount of people do not match in camera a')
print(string.format('load %d persons from camera a', number1))

local imgs2 = train_data:read('/imgs2')
local imgs2_size = imgs2:dataspaceSize()
print('load' .. imgs2_size[1] .. 'images from camera b')
local seq_length2 = train_data:read('/seq_length2'):all()
local index2 = train_data:read('/index2'):all()
local number2 = index2:size(1)
assert(seq_length2:size(1) == index2:size(1), 'the amount of people do not match in camera b')
print(string.format('load %d persons form camera b', number2))

assert(number1 == number2, 'the amounts of people in both cameras should be equal')
local n_p = number1 -- the amount of persons
print(string.format('there are %d persons in training data', n_p))
local no = train_data:read('/number'):all()

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
m_params.class_n = n_p
if opt.start_from == '' then
    model.net = nn.ReIdModel(m_params)
    model.crit = nn.ReIdCriterion()
else
    model.net = torch.load(opt.start_from)
    model.crit = nn.ReIdCriterion()
end


if opt.gpuid >= 0 then  -- for gpu
    for k,v in pairs(model) do v:cuda() end
end

local params, grad_params = model.net:getParameters()
model.net:createClones(opt.max_seq_len)

-- function to random choose part of sequence to train
function choose_frames(index, seq_length, img, gpu)
    local x1 = index
    local x2 = index + seq_length - 16
    local x_start = 0
    local x_end = 0
    local frames = {}
    if seq_length <= 16 then
        x_start = x1
        x_end = x_start + seq_length - 1
    else
        x_start = torch.random(x1, x2)
        x_end = x_start + 15
    end

   for i = x_start, x_end do
       local img_c = img:partial({i, i}, {1, 3}, {1, 128}, {1, 64})
       if gpu then img_c = img_c:cuda() end
       table.insert(frames, img_c)
   end

   return frames  
end


-- main Loop
local epoch = 1
local gpu = false
if opt.gpuid >= 0 then gpu = true end
while epoch <= opt.max_epoch do
    model.net:training()
    for i = 1, n_p  do
        local x = index1[i]  --start postion for ith person's image in image data
        local pos = index2[i] --postive pair image's start postion in data
        local j = torch.random(n_p)
        while j == i do
            j = torch.random(n_p)
        end
        local neg = index2[j] --negtive pair image's start postion in data
        local l1 = seq_length1[i]
        local l2 = seq_length2[i]
        local l3 = seq_length2[j]
        local example = choose_frames(x, l1, imgs1, gpu)
        local pos_seq = choose_frames(pos, l2, imgs2, gpu)
        local neg_seq = choose_frames(neg, l3, imgs2, gpu)
        local label = {}
        table.insert(label, i)
        table.insert(label, j)

    -- a train with postive sequence
        grad_params:zero()
        local pos_feat = model.net:forward({example, pos_seq})
        local loss1 = model.crit:forward(pos_feat, {label[1], label[1]})
        print(string.format('the %d person postive loss to  person %d in epoch %d is: %f', i,i, epoch, loss1))
        print(model.crit.l1, model.crit.l2, model.crit.s_loss)
        local dpos_feat = model.crit:backward(pos_feat, {label[1], label[1]})
        model.net:backward({example, pos_seq}, dpos_feat)
        params:add(-opt.learning_rate, grad_params)
        --local grad = grad_params
        --print(torch.max(grad))
        --print(torch.min(grad))

    -- a train with negative sequence
        grad_params:zero()
        local neg_feat = model.net:forward({example, neg_seq})
        local loss2 = model.crit:forward(neg_feat, {label[1], label[2]})
        print(string.format('the %d person negative loss to person %d in epoch %d is: %f', i,j, epoch, loss2))
        print(model.crit.l1, model.crit.l2, model.crit.s_loss)
        local dneg_feat = model.crit:backward(neg_feat, {label[1], label[2]})
        model.net:backward({example, neg_feat}, dneg_feat)
        --print(torch.max(grad))
        --print(torch.min(grad))

        params:add(-opt.learning_rate, grad_params)
        --print(torch.max(params))
        --print(torch.min(params))
    end
        print('finish the training of epoch' .. epoch)
        if epoch % 50 == 0 then
            local checkpoint = model.net
            local checkpoint_path = path.join(opt.checkpath, opt.model_name)
            torch.save(checkpoint_path .. '.t7', checkpoint)
            print('wrote model parameters to ' .. checkpoint_path .. '.t7')
        end
        epoch = epoch + 1
    
end

        

