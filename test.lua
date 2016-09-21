require 'torch'
require 'nn'
require 'nngraph'
require 'ReIdModel'
require 'hdf5'
cmd = torch.CmdLine()
cmd:text()
cmd:text('test the model for reid')
cmd:text()
cmd:text('Options')

cmd:option('-model', 'trained_model.t7', 'path to the model to test/evaluate')
cmd:option('-input_images', './reid_data/data_test.h5', 'path to probe images to test')
cmd:option('-maxR', 20, 'max CMC rank value')
cmd:option('-gpuid', 1, 'which gpu to use. -1 = use cpu')
cmd:option('-seed', 123, 'random seed')
cmd:option('-gallery_length', 16, 'gallery sequence length for test')
cmd:text()

local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

if opt.gpuid >= 0 then
    require 'cutorch'
    require 'cunn'
    cutorch.manualSeed(opt.seed)
    cutorch.setDevice(opt.gpuid + 1)
end

local reid_net = torch.load(opt.model)
len = opt.gallery_length
reid_net:createClones(len)
reid_net:evaluate()

print('load test images data')
local data = hdf5.open(opt.input_images, 'r')
local imgs_p = data:read('/imgs1')
local index_p = data:read('/index1'):all()
local length_p = data:read('/seq_length1'):all()
local imgs_g = data:read('/imgs2')
local index_g = data:read('/index2'):all()
local length_g = data:read('/seq_length2'):all()
local n1 = torch.sum(length_p)
local n2 = torch.sum(length_g)
local num1 = index_p:size(1)
local num2 = index_g:size(1)
assert(num1 == num2, 'amount of people do not match')
local num = num1
print(string.format('load %d people and %d images from camera a, load %d images from camera b', num, n1, n2))

local probe = {}
local gallery = {}
for i = 1, num do
    local imgs_set_p = {}
    local imgs_set_g = {}
    if length_p[i] > len then
        local x1 = index_p[i]
        for j = x1, x1 + len - 1 do
            local ximg = imgs_p:partial({j, j}, {1, 3}, {1, 128}, {1, 64})
            table.insert(imgs_set_p, ximg)
        end
    else
        local x1 = index_p[i]
        for j = x1, x1+length_p[i]-1 do
            local ximg = imgs_p:partial({j, j}, {1, 3}, {1, 128}, {1, 64})
            table.insert(imgs_set_p, ximg)
        end
    end

    if length_g[i] > len then
        local x1 = index_g[i] + length_g[i] - len
        for j = x1, x1 + len - 1 do
            local ximg = imgs_g:partial({j, j}, {1, 3}, {1, 128}, {1, 64})
            table.insert(imgs_set_g, ximg)
        end
    else
        local x1 = index_g[i]
        for j = x1, x1+length_g[i]-1 do
            local ximg = imgs_g:partial({j, j}, {1, 3}, {1, 128}, {1, 64})
            table.insert(imgs_set_g, ximg)
        end
    end
    
    table.insert(probe, imgs_set_p)
    table.insert(gallery, imgs_set_g)
end

--convert images to gpu
if opt.gpuid >= 0 then
    for i = 1, num do
        for j = 1, #probe[i] do
            probe[i][j] = probe[i][j]:cuda()
        end
        
        for j = 1, #gallery[i] do
            gallery[i][j] = gallery[i][j]:cuda()
        end

    end
end


local probe_feat = {}
local gall_feat = {}

for i = 1, num do
    local feat = reid_net:forward({probe[i], gallery[i]})
    local feat_p = feat[1][1]
    local feat_g = feat[1][2]
    table.insert(probe_feat, feat_p)
    table.insert(gall_feat, feat_g)
end

local R = opt.maxR
for r = 1, R do
    local acc = 0
    local count = 0
    for i = 1, num do
        local p_feat = probe_feat[i]
        local dist = torch.zeros(num)
        dist:typeAs(p_feat)
        for j = 1, num do
            local g_feat = gall_feat[j]
            local d = torch.pow((torch.sum(torch.pow((p_feat - g_feat), 2))), 0.5)
            dist[j] = d
        end
        local dist, sort = torch.sort(dist)
        for s = 1,r do
            if sort[s] == i then
                count = count + 1
                break
            end
        end
    end
    
    print(string.format('there are %d persons ared corrected ideificated in test', count))
    acc = count/num
    
    print(string.format('the accuracy when R = %d is %f', r, acc))
end





