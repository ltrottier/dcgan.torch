
local nn = require 'nn'
local utils = paths.dofile'utils.lua'

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   assert(opt and opt.depth)
   assert(opt and opt.widen_factor)

   local depth = opt.depth

   local blocks = {}

   local function wide_basic(nInputPlane, nOutputPlane, stride)
      local conv_params = {
         {3,3,stride,stride,1,1},
         {3,3,1,1,1,1},
      }
      local nBottleneckPlane = nOutputPlane

      local block = nn.Sequential()
      local convs = nn.Sequential()

      for i,v in ipairs(conv_params) do
         if i == 1 then
            local module = nInputPlane == nOutputPlane and convs or block
            module:add(SBatchNorm(nInputPlane)):add(ReLU(true))
            convs:add(Convolution(nInputPlane,nBottleneckPlane,table.unpack(v)))
         else
            convs:add(SBatchNorm(nBottleneckPlane)):add(ReLU(true))
            convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
         end
      end

      local shortcut = nInputPlane == nOutputPlane and
         nn.Identity() or
         Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)

      return block
         :add(nn.ConcatTable()
            :add(convs)
            :add(shortcut))
         :add(nn.CAddTable(true))
   end

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride)
      local s = nn.Sequential()

      s:add(block(nInputPlane, nOutputPlane, stride))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   local model = nn.Sequential()
   do
      assert((depth - 2) % 8 == 0, 'depth should be 8n+2')
      local n = (depth - 2) / 8

      local k = opt.widen_factor
      local nStages = torch.Tensor{16, 16*k, 32*k, 64*k, 128*k}

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 64x64)
      model:add(layer(wide_basic, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 64x64)
      model:add(layer(wide_basic, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 32x32)
      model:add(layer(wide_basic, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 16x16)
      model:add(layer(wide_basic, nStages[4], nStages[5], n, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(nStages[5]))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[5]):setNumInputDims(3))
      model:add(nn.Linear(nStages[5], 1))
      model:add(nn.Sigmoid())
   end

   utils.DisableBias(model)
   utils.MSRinit(model)
   utils.FCinit(model)

   return model
end

return createModel
