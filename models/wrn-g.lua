
local nn = require 'nn'
local utils = paths.dofile'utils.lua'

local Convolution = nn.SpatialConvolution
local FullConvolution = nn.SpatialFullConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   assert(opt and opt.depth)
   assert(opt and opt.widen_factor)

   local depth = opt.depth

   local blocks = {}

   local function wide_block(nInputPlane, nOutputPlane, stride)
      local block = nn.Sequential()
      if stride == 2 then
        block:add(SBatchNorm(nInputPlane))
        block:add(ReLU(true))
        block:add(FullConvolution(nInputPlane,nOutputPlane, 4,4, 2,2, 1,1))
      else
        local convs = nn.Sequential()
        convs:add(SBatchNorm(nInputPlane))
        convs:add(ReLU(true))
        convs:add(Convolution(nInputPlane,nOutputPlane, 3,3, 1,1, 1,1))
        convs:add(SBatchNorm(nOutputPlane))
        convs:add(ReLU(true))
        convs:add(Convolution(nOutputPlane,nOutputPlane, 3,3, 1,1, 1,1))

        local shortcut = nInputPlane == nOutputPlane
                         and nn.Identity()
                         or Convolution(nInputPlane,nOutputPlane, 1,1, 1,1)

        block:add(nn.ConcatTable()
                    :add(convs)
                    :add(shortcut))
             :add(nn.CAddTable(true))
      end

      return block
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
      local nz = opt.nz
      local k = opt.widen_factor
      local nStages = torch.Tensor{128*k, 64*k, 32*k, 16*k, 3}

      model:add(FullConvolution(nz, nStages[1], 4,4)) -- one conv at the beginning (spatial size: 4x4)
      model:add(layer(wide_block, nStages[1], nStages[2], n, 2)) -- Stage 1 (spatial size: 8x8)
      model:add(layer(wide_block, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(wide_block, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 32x32)
      model:add(layer(wide_block, nStages[4], nStages[5], n, 2)) -- Stage 3 (spatial size: 64x64)
      --model:add(SBatchNorm(nStages[5]))
      --model:add(ReLU(true))
      --model:add(Convolution(nStages[5], 3, 1,1, 1,1)) -- one conv at the end (spatial size: 64x64)
      --model:add(SBatchNorm(3))
      model:add(nn.Tanh())
   end

   utils.DisableBias(model)
   utils.MSRinit(model)
   utils.FCinit(model)

   return model
end

return createModel
