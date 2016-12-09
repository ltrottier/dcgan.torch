
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
      local convs = nn.Sequential()

      -- residual mapping
      local module
      if nInputPlane == nOutputPlane then
        module = convs
      else
        module = block
      end
      module:add(SBatchNorm(nInputPlane))
      module:add(ReLU(true))
      if stride == 2 then
        convs:add(FullConvolution(nInputPlane,nOutputPlane, 3,3, 2,2, 1,1, 1,1))
      else
        convs:add(Convolution(nInputPlane,nOutputPlane, 3,3, 1,1, 1,1))
      end

      convs:add(SBatchNorm(nOutputPlane))
      convs:add(ReLU(true))
      convs:add(Convolution(nOutputPlane,nOutputPlane, 3,3, 1,1, 1,1))

      -- skip connection
      local shortcut
      if nInputPlane == nOutputPlane then
         shortcut = nn.Identity()
      else
        if stride == 2 then
          shortcut = FullConvolution(nInputPlane,nOutputPlane, 2,2, 2,2)
        else
          shortcut = Convolution(nInputPlane,nOutputPlane, 1,1, 1,1)
        end
      end

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
      local nz = opt.nz
      local k = opt.widen_factor
      local nStages = torch.Tensor{128*k, 64*k, 32*k, 16*k, 16}

      model:add(FullConvolution(nz,nStages[1], 8,8)) -- one conv at the beginning (spatial size: 8x8)

      model:add(layer(wide_block, nStages[1], nStages[2], n, 2)) -- Stage 1 (spatial size: 16x16)
      model:add(layer(wide_block, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 32x32)
      model:add(layer(wide_block, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 64x64)
      model:add(layer(wide_block, nStages[4], nStages[5], n, 1)) -- Stage 3 (spatial size: 64x64)
      model:add(SBatchNorm(nStages[5]))
      model:add(ReLU(true))
      model:add(Convolution(nStages[5], 3, 1,1, 1,1)) -- one conv at the end (spatial size: 64x64)
      model:add(SBatchNorm(nStages[5]))
      model:add(nn.Tanh())
   end

   utils.DisableBias(model)
   utils.MSRinit(model)
   utils.FCinit(model)

   return model
end

return createModel
