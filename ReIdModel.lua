require 'nn'
local RCnn = require 'RCnn'

local layer, parent = torch.class('nn.ReIdModel', 'nn.Module')

function layer:__init(opt)
    parent.__init(self)
    
    self.input_size = opt.img_feat_len
    self.rnn_size = opt.rnn_size
    self.output_size = opt.output_size
    self.rcnn = RCnn.buildNet(self.input_size, self.rnn_size, self.output_size)
    self.linear = nn.Linear(self.output_size, 200)
    self._createInitState()
end

function layer:_createInitState()

    if not self.InitState then self.InitState = {} end
    for i = 1,2 do
        self.InitState[t] = torch.zeros(self.rnn_size)
    end

end

function layer:createClones()
    
    self.rcnns = {}
    print('copy models for video sequence')
    self.rcnns[1] = {self.rcnn}
    self.rcnns[2] = {self.rcnn}
    for t = 2,16 do
        self.rcnns[1][t] = self.rcnn:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
    for t = 2,16 do
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
end

function layer:evaluate()
    for k,v in pairs(self.rcnns) do
        for i,j in pairs(v) do
            j:evaluate()
        end
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
    self.state = {} 
    self._createInitState()
    if not self.rcnns then self:createClones() end

    for i = 1,2 do
        seq[i] = input[i]
        self.state[i][0] = self.InitState[i] 
    end
        
    local ot1 = {}
    local ot2 = {}
    for i = 1,2 do    
        for t = 1,16 do
            local img = seq[i][t]
            local out = self.rcnns1[t]:forward({img, self.state[i][t-1]})
            self.state[i][t] = out[2] -- for next fram
            table.insert(ot1, out[1]) -- temporal output
        end
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

    local ident1 = self.linears[1]:forward(sf1)
    local ident2 = self.linears[2]:forward(sf2)
    local ident = {}
    table.insert(ident, ident1)
    table.insert(ident, ident2)

    self.output = {}
    table.insert(self.output, seq_feat)
    table.insert(self.output, ident)
    return self.output
end


function layer:updateGradInput(input, gradOutput)
    local dstate1 = {[16] = self.InitState[1]}
    local dstate2 = {[16] = self.InitState[2]}
    local dimgs1 = {}
    local dimgs2 = {}
    --bp for each sequence
    for t = 16, 1, -1 do
       local dout = {}
       local df = self.linears[1]:backward(self.output[2][1], gradOutput[2][1])
       df = (df + gradOutput[1][1])/16
       table.insert(dout, df)
       table.insert(dout, dstate1[t])
       local dinputs = self.rcnns[1][t]:backward({input[1][t], self.state[1][t-1]}, dout)
       dimgs1[t] = dinput[1]
       dstate1[t-1] = dinputs[2]
   end

   for t = 16, 1, -1 do
       local dout = {}
       local df = self.linears[2]:backward(self.output[2][2], gradOutput[2][2])
       df = (df + gradOutput[1][2])/16
       table.insert(dout, df)
       table.insert(dout, dstate2[t])
       local dinputs = self.rcnns[2][t]:backward({input[2][t], self.state[2][t-1]}, dout)
       dimgs2[t] = dinput[2]
       dstate[t-1] = dinputs[2]
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
    local ident1 = input[2][1]
    local ident2 = input[2][2]
    local label1 = label[1]
    local label2 = label[2]
    local identity = true
    if label1 ~= label2 then identity = false end

    assert(f1:size() == f2:size(), 'feature dimension not match')
    self.soft1 = nn.LogSoftMax()
    self.soft2 = nn.LogSoftMax()
    local soft1 = self.soft1:forward(ident1)
    local soft2 = self.soft2:forward(ident2)
    local l1 = -soft1[label1] -- identity loss
    local l2 = -soft2[label2]
    local s_loss
    -- calculate the Siamese Network loss
    if identity then
        s_loss = torch.sum(torch.pow(f1-f2, 2)) / 2
    else
        self.z = 2 - torch.sum(torch.pow(f1-f2, 2))
        s_loss = math.max(self.z, 0) / 2
    end
   
    self.output = l1 + l2 + s_loss
    return self.output
end

function crit:updateOutput(input, label)
    local df1
    local df2
    local f1 = input[1][1]
    local f2 = input[1][2]
    local ident1 = input[2][1]
    local ident2 = input[2][2]
    local label1 = label[1]
    local label2 = label[2]
    local identity = true
    if label1 ~= label2 then identity = false end

    df1:resizeAs(input[1][1]):zero()
    df2:resizeAs(input[1][2]):zero()
    local dsoft1 = torch.zeros(200)
    local dsoft2 = torch.zeros(200)
    dsfot1[label1] = -1
    dsoft2[label2] = -1
    local dident1 = self.soft1:backward(ident1, dsoft1)
    local dident2 = self.soft2:backward(ident2, dsoft2)

    if identity then 
        df1 = df1 + (f1 - f2)
        df2 = df2 + (f2 - f1)
    else
        if self.z < 2 then
            local sum = torch.sum(torch.pow(f1-f2,2))
            local d = torch.pow(sum,0.5)
            local delta_f = (1 - 2/d) * (f1 - f2)
            df1 = df1 + delta_f
            df2 = df2 - delta_f
        end
    end

    self.gradInput = {}
    local dident = {}
    table.insert(dident, dident1)
    table.insert(dident, didnet2)

    local df = {}
    table.insert(df, df1)
    table.insert(df, df2)

    table.insert(self.gradInput, df)
    table.insert(sefl.gradInput, dident)
    return self.gradInput
end



            

    


    




