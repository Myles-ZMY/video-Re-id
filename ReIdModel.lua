require 'nn'
local RCnn = require 'RCnn'

local layer, parent = torch.class('nn.ReIdModel', 'nn.Module')

function layer:__init(opt)
    parent.__init(self)
    
    self.input_size = opt.img_feat_len
    self.rnn_size = opt.rnn_size
    self.output_size = opt.output_size
    self.rcnn = RCnn.buildNet(self.input_size, self.rnn_size, self.output_size)
    self.linear = nn.Linear(self.output_size, opt.class_n, false)
    self.soft1 = nn.LogSoftMax()
    self.soft2 = nn.LogSoftMax()
    self:_createInitState()
end

function layer:_createInitState()

    if not self.InitState then self.InitState = {} end
    for i = 1,2 do
        self.InitState[i] = torch.zeros(self.rnn_size)
    end

end

function layer:createClones(max_len)
    
    self.rcnns = {}
    print('copy models for video sequence')
    self.rcnns[1] = {self.rcnn}
    self.rcnns[2] = {}
    for t = 2, max_len do
        self.rcnns[1][t] = self.rcnn:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
    for t = 1, max_len do
        self.rcnns[2][t] = self.rcnn:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end

    self.linears = {self.linear}
    self.linears[2] = self.linear:clone('weight', 'gradWeight')

end

function layer:training()
    for k,v in pairs(self.rcnns) do
        for i,j in pairs(v) do
            j:training()
        end
    end

    for k,v in pairs(self.linears) do
        v:training()
    end

end

function layer:evaluate()
    for k,v in pairs(self.rcnns) do
        for i,j in pairs(v) do
            j:evaluate()
        end
    end

    for k,v in pairs(self.linears) do
        v:training()
    end

end

function layer:parameters()
    local p1, g1 = self.rcnn:parameters()
    local p2, g2 = self.linear:parameters()

    local params = {}
    for k,v in pairs(p1) do table.insert(params, v) end
    for k,v in pairs(p2) do table.insert(params, v) end

    local grad_params = {}
    for k,v in pairs(g1) do table.insert(grad_params, v) end
    for k,v in pairs(g2) do table.insert(grad_params, v) end

    return params, grad_params
end

function layer:updateOutput(input)
    local seq = {}
    self.state = {{}, {}} 
    self:_createInitState()
    if not self.rcnns then self:createClones() end

    for i = 1,2 do
        seq[i] = input[i]
        self.state[i][0] = self.InitState[i]:cuda() 
    end
        
    local len1 = #input[1]
    local len2 = #input[2]
    local ot1 = {}
    local ot2 = {}
    for t = 1, len1 do    
        local img = seq[1][t]
        local out = self.rcnns[1][t]:forward({img, self.state[1][t-1]})
        self.state[1][t] = out[2] -- for next fram
        table.insert(ot1, out[1]) -- temporal output
    end

    for t = 1, len2 do
        local img = seq[2][t]
        local out = self.rcnns[2][t]:forward({img, self.state[2][t-1]})
        self.state[2][t] = out[2]
        table.insert(ot2, out[1])
    end
    local sf1 = 0
    local sf2 = 0

    for k = 1,#ot1 do
        sf1 = sf1 + ot1[k]
    end
    sf1 = sf1/#ot1

    for k = 1,#ot2 do
        sf2 = sf2 + ot2[k]
    end
    sf2 = sf2/#ot2

    local seq_feat = {}
    table.insert(seq_feat, sf1)
    table.insert(seq_feat, sf2)
    self.ident1 = self.linears[1]:forward(sf1)
    self.ident2 = self.linears[2]:forward(sf2)
    local logp1 = self.soft1:forward(self.ident1)
    local logp2 = self.soft2:forward(self.ident2)
    local logp = {}
    table.insert(logp, logp1)
    table.insert(logp, logp2)

    self.output = {}
    table.insert(self.output, seq_feat)
    table.insert(self.output, logp)
    return self.output
end


function layer:updateGradInput(input, gradOutput)
    local len1 = #input[1]
    local len2 = #input[2]
    local dstate1 = {[len1] = self.InitState[1]:cuda()}
    local dstate2 = {[len2] = self.InitState[2]:cuda()}
    local dimgs1 = {}
    local dimgs2 = {}
    --bp for each sequence
    local df1 = self.soft1:backward(self.ident1, gradOutput[2][1])
    local df2 = self.soft2:backward(self.ident2, gradOutput[2][2])
    df1 = self.linears[1]:backward(self.output[1][1], df1)
    df2 = self.linears[2]:backward(self.output[1][2], df2)
    df1 = (df1 + gradOutput[1][1])/len1
    df2 = (df2 + gradOutput[1][2])/len2
    for t = len1, 1, -1 do
       local dout = {}
       table.insert(dout, df1)
       table.insert(dout, dstate1[t])
       local dinputs = self.rcnns[1][t]:backward({input[1][t], self.state[1][t-1]}, dout)
       dimgs1[t] = dinputs[1]
       dstate1[t-1] = dinputs[2]
   end

   for t = len2, 1, -1 do
       local dout = {}
       table.insert(dout, df2)
       table.insert(dout, dstate2[t])
       local dinputs = self.rcnns[2][t]:backward({input[2][t], self.state[2][t-1]}, dout)
       dimgs2[t] = dinputs[1]
       dstate2[t-1] = dinputs[2]
   end

   local dimgs = {}
   table.insert(dimgs, dimgs1)
   table.insert(dimgs, dimgs2)
   
   self.gradInput = dimgs
   return self.gradInput

end


-- model criterion/loss function 

local crit, parent = torch.class('nn.ReIdCriterion', 'nn.Criterion')

function crit:__init()
    parent.__init(self)

end

function crit:updateOutput(input, label)
    local f1 = input[1][1]
    local f2 = input[1][2]
    local logp1  = input[2][1]
    local logp2  = input[2][2]
    local label1 = label[1]
    local label2 = label[2]
    local identity = true
    if label1 ~= label2 then identity = false end
    assert(f1:size(1) == f2:size(1), 'feature dimension not match')
    self.l1 = -logp1[label1] -- identity loss
    self.l2 = -logp2[label2]
    local d = torch.pow(torch.sum(torch.pow(f1-f2, 2)), 0.5)
    -- calculate the Siamese Network loss
    if identity then
        self.s_loss = 0.5 * torch.pow(d, 2)
    else
        self.z = 2 - d
        self.s_loss = 0.5 * torch.pow(math.max(self.z, 0), 2)
    end
   
    self.output = self.l1 + self.l2 + self.s_loss
    return self.output
end

function crit:updateGradInput(input, label)
    local f1 = input[1][1]
    local f2 = input[1][2]
    local dim = f1:size(1)
    local df1 = torch.Tensor(dim):typeAs(f1)
    local df2 = torch.Tensor(dim):typeAs(f2)
    local logp1 = input[2][1]
    local logp2 = input[2][2]
    local label1 = label[1]
    local label2 = label[2]
    local identity = true
    if label1 ~= label2 then identity = false end
   
    local n = logp1:size(1)
    local dlogp1 = torch.zeros(n)
    local dlogp2 = torch.zeros(n)
    dlogp1[label1] = -1
    dlogp2[label2] = -1
    dlogp1 = dlogp1:cuda()
    dlogp2 = dlogp2:cuda()
    
    df1:zero()
    df2:zero()
    if identity then
        local delta_f = f1 - f2
        df1 = delta_f
        df2 = -delta_f
    else
        if self.z > 0 then
            local sum = torch.sum(torch.pow(f1-f2,2))
            local d = torch.pow(sum,0.5)
            local delta_f = (1 - 2/d) * (f1 - f2)
            df1 = delta_f
            df2 = -delta_f
        end
    end

    self.gradInput = {}
    local dlogp = {}
    table.insert(dlogp, dlogp1)
    table.insert(dlogp, dlogp2)

    local df = {}
    table.insert(df, df1)
    table.insert(df, df2)

    table.insert(self.gradInput, df)
    table.insert(self.gradInput, dlogp)
    return self.gradInput
end



            

    


    




