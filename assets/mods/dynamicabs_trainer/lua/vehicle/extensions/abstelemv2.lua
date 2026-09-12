-- abstelemv2.lua: per-wheel brake application hook for Dynamic ABS
--
-- Hooks into the native setWheelBrakeUpdate callback to flawlessly support
-- thermal brake models and race brakes without manual brakeTorque clobbering.

local M = {}

local wheelData = {}
local wheelNameToIndex = {}
local initialized = false
local perWheelMode = false
local brakeCmd = {}

-- Called natively by the physics engine every single step.
-- wd.brakeTorque is guaranteed to be perfectly updated by thermals here.
local function updateBrakeABS(wd, brake, invAirspeed, airspeed, airspeedCutOff, dt)
  local cmd = brake
  if perWheelMode then
    local idx = wheelNameToIndex[wd.name]
    if idx and brakeCmd[idx] then
      cmd = brakeCmd[idx]
    end
  end
  
  local brakeInputSplit = wd.brakeInputSplit or 1
  local brakeSplitCoef = wd.brakeSplitCoef or 1
  return wd.brakeTorque * (math.min(cmd, brakeInputSplit) + math.max(cmd - brakeInputSplit, 0) * brakeSplitCoef)
end


local function tryInitWheels()
  wheelData = {}
  wheelNameToIndex = {}
  pcall(function()
    if wheels and wheels.wheelRotators and wheels.wheelRotatorCount and wheels.wheelRotatorCount > 0 then
      for i = 0, wheels.wheelRotatorCount - 1 do
        local w = wheels.wheelRotators[i]
        if w then
          local wName = w.name or ("rot_" .. i)
          table.insert(wheelData, { name = wName, ref = w })
          wheelNameToIndex[wName] = #wheelData
        end
      end
    end
  end)

  if #wheelData > 0 then
    initialized = true
    for _, wd in ipairs(wheelData) do
      -- Register our native brake callback
      if wheels and wheels.setWheelBrakeUpdate then
        wheels.setWheelBrakeUpdate(wd.name, updateBrakeABS, updateBrakeABS)
      end
      -- Disable legacy simple ABS
      pcall(function()
        if wd.ref.absSlipRatioTarget ~= nil then wd.ref.absSlipRatioTarget = 0 end
        if wd.ref.absEnabled ~= nil then wd.ref.absEnabled = false end
        if wd.ref.absActive ~= nil then wd.ref.absActive = false end
      end)
    end
  end
end


local function setBrakes(cmdTable)
  if not initialized then tryInitWheels() end

  brakeCmd = cmdTable or {}
  local maxBrake = 0
  for i, cmd in pairs(brakeCmd) do
    brakeCmd[i] = math.max(0, math.min(1, cmd or 0))
    if brakeCmd[i] > maxBrake then maxBrake = brakeCmd[i] end
  end

  perWheelMode = true
  electrics.values.brake = maxBrake
end


local function releaseBrakes()
  perWheelMode = false
  brakeCmd = {}
end


local function onReset()
  initialized = false
  perWheelMode = false
  brakeCmd = {}
  wheelData = {}
  wheelNameToIndex = {}
end


M.onReset           = onReset
M.setBrakes         = setBrakes
M.releaseBrakes     = releaseBrakes

return M
