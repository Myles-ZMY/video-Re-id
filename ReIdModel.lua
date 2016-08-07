require 'nn'
local RCnn = require 'RCnn'

local layer, parent = torch.class('nn.ReIdModel', 'nn.Module')

function layer:__init(opt)
    parent.__init(self)
    
    self.input_size = opt.img_feat_len
    self.rnn_size = opt.rnn_size
    self.output_size = opt.output_size

end

function layer:_createInitState()

    if not self.InitState then self.InitState = {} end
    for i = 1,2 do
        self.InitState[t] = torch.zeros(self.rnn_size)
    end

end

function layer:createClones(len)
    
    self.rcnns = {}
    print('copy models for video sequence')
    self.rcnns[1] = {self.rcnn}
    self.rcnns[2] = {self.rcnn}
    for t = 2,len[1] do
        self.rcnns[1][t] = self.rcnn:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end
    for t = 2,len[2] do
        self.rcnns[2][t] = self.rcnn:clone('weight', 'bias', 'gradWeight', 'gradBias')
    end

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


function layer:updateOutput(input)
    local seq = {}
    local len = {}
    self.state = {} 
    self._createInitState()

    for i = 1,2 do
        seq[i] = input[i]
        len[i] = #input[i]
        self.state[i][0] = self.InitState[i] 
    end
        
    self.createClones(len)
    local ot1 = {}
    local ot2 = {}
    for i = 1,2 do    
        for t = 1,len1 do
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

    local output = {}
    table.insert(output, sf1)
    table.insert(output, sf2)

    return output
end

function layer:updateGradInput(input, gradOutput)
    local n1 = input[1]
    local n2 = input[2]
    local dstate1 = {[n1] = self.InitState[1]}
    local dstate2 = {[n2] = self.InitState[2]}
    local dimgs1 = {}
    local dimgs2 = {}
    --bp for each sequence
    for t = n1, 1, -1 do
       local dout = {}
       table.insert(dout,gradOutput[1]/n1)
       table.insert(dout,dstate1[t])
       local dinputs = self.rcnns[1][t]:backward({input[1][t], self.state[1][t-1]}, dout)
       dimgs1[t] = dinput[1]
       dstate1[t-1] = dinputs[2]
   end

   for t = n2, 1, -1 do
       local dout = {}
       table.insert(dout, gradOutput[2]/n2)
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

function crit:updateOutput(input)
    local f1 = input[1]
    local f2 = input[2]
    local label1 = input[3][1]
    local label2 = input[2][2]
    local identity = input[4]
    local isize = #f1

    assert(f1 == f2, 'feature dimension not match')

    self.tran1 = nn.Linear(isize, 200, false)
    self.tran2 = tran1:clones('weight', 'gradWeigth')
    local soft1 = nn.LogSoftMax(self.tran1:forward(f1))
    local soft2 = nn.LogSoftMax(self.tran2:forward(f2))
    local l1 = -soft1[label1] -- identity loss
    local l2 = -soft2[label2]
    local s_loss
    -- calculate the Siamese Network loss
    if identity then
        s_loss = torch.sum(torch.pow(f1-f2, 2)) / 2
    else
        self.z = 2 - torch.sum(torch.pow(f1-f2, 2))
        s_loss = math.max(self.z,0) / 2
    end
   
    self.output = l1 + l2 + s_loss
    return self.output
end

function crit:updateOutput(input, gradOutput)
    local gradInput1
    local gradInput2
    local label1 = input[3][1]
    local label2 = input[3][2]
    local identity = input[4]
    gradInput1:resizeAs(input[1]):zero()
    gradInput2:resizeAs(input[2]):zero()
    local dsoft1 = torch.zeros(200)
    local dsoft2 = torch.zeros(200)
    dsfot1[label1] = -1
    dsoft2[label2] = -1
    local df1 = self.tran1:backward(f1, dsoft1)
    local df2 = self.tran2:backward(f2, dsoft2)
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
    table.insert(self.gradInput, df1)
    table.insert(sefl.gradInput, df2)
    return self.gradInput
end



            

    


    




