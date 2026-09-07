local M = {}
M.type = "auxiliary"

-- Deployment controller for the co-sim policy contract.
-- 100 Hz, 35x64 obs, front/rear release in [0,1]. Separate from MTB-ML-ABS.lua.

local GRAV = 9.81
local OBS_DIM = 35
local N_STACK = 64
local FLAT_DIM = OBS_DIM * N_STACK   -- 2240
local ACT_DIM = 2
local TICK_RATE_HZ = 100
local TICK_STEP = 1 / TICK_RATE_HZ
local ENGAGE_BRAKE_THRESH = 0.9
local ENGAGE_SPEED_MS = 8.0
local DISENGAGE_SPEED_MS = 0.5

local weightsModule = "controller/mtb_ml_weights"
local W = nil
local weightsOk = false
local haveTelem = false
local warnedMissing = false
local warnedNoData = false

-- Trace buffers are preallocated numbers only. Formatting or table growth per
-- tick allocates, and allocation on the physics thread costs real braking: a
-- string.format per tick measured 0.135 g worse. Strings are built on disengage.
local TRACE_N = 37
local TRACE_MAX = 500
local traceBuf = {}
local traceCount = 0
local traceSeq = 0
local traceOn = false

local rawObs = {}
local stack = {}
local flat = {}
local normObs = {}
local actBuf = {}
local cmd = {1, 1, 1, 1} -- FR, FL, RR, RL brake fractions
local prevReleaseFront = 0
local prevReleaseRear = 0
local prevWs = nil
local prevPitch, prevRoll = 0, 0

local wheelToBrakeMap = {3, 4, 1, 2}
-- Read live rotators instead of GFX-rate tel_ws_* (stale under speed factor).
local liveWs = nil
local origBrakeTorque = {}
local wheelCount = 0

local active = false
local driverBrakeHeld = false
local lastWrittenBrake = -1
local timeAccum = 0
local mlabsTicks = 0
local tickMsSum = 0
local tickMsCount = 0

local function finite(x)
  return type(x) == "number" and x == x and x > -math.huge and x < math.huge
end

local function clamp(x, lo, hi)
  if not finite(x) then return 0 end
  if x < lo then return lo end
  if x > hi then return hi end
  return x
end

local function fastTanh(x)
  if x > 10 then return 1 end
  if x < -10 then return -1 end
  local e2x = math.exp(2 * x)
  return (e2x - 1) / (e2x + 1)
end

local function parseFloats(s)
  local out = {}
  local n = 0
  for token in string.gmatch(s, "%S+") do
    local value = tonumber(token)
    if value == nil then return nil, n end
    n = n + 1
    out[n] = value
  end
  return out, n
end

local function parseMatrix(strings, rows, cols)
  local matrix = {}
  local row, col, total = 1, 0, 0
  matrix[1] = {}
  for _, text in ipairs(strings) do
    for token in string.gmatch(text, "%S+") do
      local value = tonumber(token)
      if value == nil then return nil, total end
      if col >= cols then
        row = row + 1
        col = 0
        if row > rows then return nil, total + 1 end
        matrix[row] = {}
      end
      col = col + 1
      total = total + 1
      matrix[row][col] = value
    end
  end
  return matrix, total
end

local function failWeights(message)
  print("[MTB-ML-ABS-CoSim] !!! FATAL: " .. message
    .. " -> controller disabled; stock brakes retained")
  W = nil
  weightsOk = false
  return false
end

local function loadWeights()
  local ok, mod = pcall(require, weightsModule)
  if not ok or type(mod) ~= "table" then
    if not warnedMissing then
      warnedMissing = true
      print("[MTB-ML-ABS-CoSim] !!! FATAL: weights module '" .. weightsModule
        .. "' missing/invalid: " .. tostring(mod))
    end
    W = nil
    weightsOk = false
    return false
  end
  if mod.interface ~= "cosim_axle_release_v2" then
    return failWeights("weights interface is " .. tostring(mod.interface)
      .. ", expected cosim_axle_release_v2")
  end
  if mod.obs_dim ~= FLAT_DIM or mod.act_dim ~= ACT_DIM then
    return failWeights("weights dimensions are " .. tostring(mod.obs_dim) .. "->"
      .. tostring(mod.act_dim) .. ", expected 2240->2")
  end
  if mod.control_hz ~= TICK_RATE_HZ then
    return failWeights("weights control_hz is " .. tostring(mod.control_hz)
      .. ", expected 100")
  end
  if mod.head ~= "ppo_tanh_release01" then
    return failWeights("weights head is " .. tostring(mod.head)
      .. ", expected ppo_tanh_release01")
  end
  if type(mod.obs_mean_s) ~= "string" or type(mod.obs_var_s) ~= "string" then
    return failWeights("normalization strings missing")
  end
  local mean, nMean = parseFloats(mod.obs_mean_s)
  local var, nVar = parseFloats(mod.obs_var_s)
  if mean == nil or var == nil or nMean ~= FLAT_DIM or nVar ~= FLAT_DIM then
    return failWeights("normalization vector length mismatch")
  end
  if type(mod.layers) ~= "table" or #mod.layers == 0 then
    return failWeights("layers missing")
  end

  local parsed = {
    obs_mean = mean,
    obs_var = var,
    clip_obs = tonumber(mod.clip_obs) or 10,
    latent_bound = tonumber(mod.latent_bound) or 0,  -- b*tanh(mu/b) before head; 0 = none
    eps = tonumber(mod.eps) or 1e-8,
    layers = {},
  }
  local inDim = FLAT_DIM
  actBuf = {}
  for li, source in ipairs(mod.layers) do
    local rows, cols = source.rows, source.cols
    if type(rows) ~= "number" or type(cols) ~= "number" or cols ~= inDim then
      return failWeights("layer " .. li .. " dimension mismatch")
    end
    local bias, nBias = parseFloats(source.b_s or "")
    if bias == nil or nBias ~= rows then
      return failWeights("layer " .. li .. " bias length mismatch")
    end
    local strings
    if type(source.W_s) == "string" then
      strings = {source.W_s}
    elseif type(source.W_chunks) == "table" then
      strings = source.W_chunks
    else
      return failWeights("layer " .. li .. " weight string missing")
    end
    local matrix, nWeights = parseMatrix(strings, rows, cols)
    if matrix == nil or nWeights ~= rows * cols then
      return failWeights("layer " .. li .. " weight length mismatch")
    end
    parsed.layers[li] = {
      W = matrix, b = bias, rows = rows, cols = cols,
      relu = li < #mod.layers,
    }
    local buffer = {}
    for j = 1, rows do buffer[j] = 0 end
    actBuf[li] = buffer
    inDim = rows
  end
  if inDim ~= ACT_DIM then
    return failWeights("final layer output is " .. inDim .. ", expected 2")
  end
  W = parsed
  weightsOk = true
  print("[MTB-ML-ABS-CoSim] weights loaded: 35x64 obs -> 2 axle releases at 100 Hz")
  return true
end

local function resetStack()
  for f = 1, N_STACK do
    local row = stack[f]
    for k = 1, OBS_DIM do row[k] = 0 end
  end
  prevWs = nil
  prevPitch, prevRoll = 0, 0
end

-- oldest FIRST (row 1), newest LAST (row N_STACK); only past frames ever enter
local function pushStack()
  for f = 1, N_STACK - 1 do
    local dst, src = stack[f], stack[f + 1]
    for k = 1, OBS_DIM do dst[k] = src[k] end
  end
  local newest = stack[N_STACK]
  for k = 1, OBS_DIM do newest[k] = rawObs[k] end
  local idx = 0
  for f = 1, N_STACK do
    local row = stack[f]
    for k = 1, OBS_DIM do
      idx = idx + 1
      flat[idx] = row[k]
    end
  end
end

-- same clip bounds as the training env (FRAME_LOW / FRAME_HIGH)
local FRAME_LO = {0,0,0,0, -50,-50,-10, 0,0,0,0, -50,-50, 0,-2,-2, -math.pi,-math.pi, 0,0, -50,-10,-10, -200,-200,-200,-200, 0,0,0,0, 0,0,0,0}
local FRAME_HI = {100,100,100,100, 50,50,10, 1,1,1,1, 50,50, 15000,12,2, math.pi,math.pi, 1,1, 50,10,10, 200,200,200,200, 1,1,1,1, 5000,5000,5000,5000}


-- Resolve rotators in abstelemetry's logical order (1=RR 2=RL 3=FR 4=FL) by
-- name: rotator order differs per car, so index alone is not the mapping.
local function buildLiveWsMap()
  liveWs = nil
  if not (wheels and wheels.wheelRotators and wheels.wheelRotatorCount) then return end
  local ev = (electrics and electrics.values) or {}
  local map = {}
  for i = 1, 4 do
    local want = ev['tel_wname_' .. i]
    if type(want) ~= 'string' then return end
    for j = 0, wheels.wheelRotatorCount - 1 do
      local r = wheels.wheelRotators[j]
      if r and r.name == want then map[i] = r break end
    end
    if not map[i] then return end
  end
  liveWs = map
end
-- MachineTrainerBoy 27-value frame + 4 slips vs the stock ABS speed estimate.
-- Same sources as ABSCoSimEnv._frame(): abstelemetry's per-tick window
-- electrics (past WIN_N ticks) + wheel speeds. Rates use the PREVIOUS tick.
local function buildObservation(data)
  local function get(key)
    local value = data[key]
    return finite(value) and value or 0
  end
  local ev = (electrics and electrics.values) or {}
  local function e(key)
    local value = ev[key]
    return finite(value) and value or 0
  end
  -- abstelemetry wheel mapping: 3=FR, 4=FL, 1=RR, 2=RL.
  local fr, fl, rr, rl
  if liveWs then
    fr = math.abs(liveWs[3].wheelSpeed or 0)
    fl = math.abs(liveWs[4].wheelSpeed or 0)
    rr = math.abs(liveWs[1].wheelSpeed or 0)
    rl = math.abs(liveWs[2].wheelSpeed or 0)
  else
    fr = math.abs(get("ws3"))
    fl = math.abs(get("ws4"))
    rr = math.abs(get("ws1"))
    rl = math.abs(get("ws2"))
  end
  local pitch = e("tel_att_pitch")
  local roll = e("tel_att_roll")
  local waFr, waFl, waRr, waRl, pitchRate, rollRate = 0, 0, 0, 0, 0, 0
  if prevWs then
    waFr = (fr - prevWs[1]) / TICK_STEP
    waFl = (fl - prevWs[2]) / TICK_STEP
    waRr = (rr - prevWs[3]) / TICK_STEP
    waRl = (rl - prevWs[4]) / TICK_STEP
    pitchRate = (pitch - prevPitch) / TICK_STEP
    rollRate = (roll - prevRoll) / TICK_STEP
  else
    prevWs = {0, 0, 0, 0}
  end
  prevWs[1], prevWs[2], prevWs[3], prevWs[4] = fr, fl, rr, rl
  prevPitch, prevRoll = pitch, roll

  rawObs[1], rawObs[2], rawObs[3], rawObs[4] = fr, fl, rr, rl
  rawObs[5] = e("tel_win_gy_avg")
  rawObs[6] = e("tel_gx_inst")
  rawObs[7] = e("tel_win_yaw_avg")
  rawObs[8], rawObs[9], rawObs[10], rawObs[11] = cmd[1], cmd[2], cmd[3], cmd[4]
  rawObs[12] = e("tel_win_gy_min")
  rawObs[13] = e("tel_win_gy_max")
  rawObs[14] = e("tel_rpm")
  rawObs[15] = e("tel_gear")
  rawObs[16] = e("tel_steer")
  rawObs[17] = pitch
  rawObs[18] = roll
  rawObs[19] = e("tel_brake_in")
  rawObs[20] = e("tel_throttle_in")
  rawObs[21] = e("tel_gz_inst")
  rawObs[22] = pitchRate
  rawObs[23] = rollRate
  rawObs[24], rawObs[25], rawObs[26], rawObs[27] = waFr, waFl, waRr, waRl
  local absRef = math.max(e("tel_abs_speed"), 0.5)
  rawObs[28] = clamp(1 - fr / absRef, 0, 1)
  rawObs[29] = clamp(1 - fl / absRef, 0, 1)
  rawObs[30] = clamp(1 - rr / absRef, 0, 1)
  rawObs[31] = clamp(1 - rl / absRef, 0, 1)
  rawObs[32] = e("tel_brk_applied_fr")
  rawObs[33] = e("tel_brk_applied_fl")
  rawObs[34] = e("tel_brk_applied_rr")
  rawObs[35] = e("tel_brk_applied_rl")
  for k = 1, OBS_DIM do rawObs[k] = clamp(rawObs[k], FRAME_LO[k], FRAME_HI[k]) end
  pushStack()
end

-- Zero-variance channels get emitted as 0. Dividing by sqrt(1e-8) would amplify
-- any car-side deviation by 9535x, saturating the clip and corrupting the stack.
local DEGENERATE_VAR = 1e-7

local function forward()
  for i = 1, FLAT_DIM do
    if W.obs_var[i] <= DEGENERATE_VAR then
      normObs[i] = 0
    else
      local denom = math.sqrt(math.max(W.obs_var[i] + W.eps, W.eps))
      normObs[i] = clamp((flat[i] - W.obs_mean[i]) / denom,
                         -W.clip_obs, W.clip_obs)
    end
  end
  local inputValues = normObs
  for li, layer in ipairs(W.layers) do
    local output = actBuf[li]
    for row = 1, layer.rows do
      local sum = layer.b[row]
      local weights = layer.W[row]
      for col = 1, layer.cols do
        sum = sum + weights[col] * inputValues[col]
      end
      output[row] = layer.relu and math.max(0, sum) or sum
    end
    inputValues = output
  end
  local mu = actBuf[#W.layers]
  local bound = W.latent_bound
  if bound > 0 then
    mu[1] = bound * fastTanh(mu[1] / bound)
    mu[2] = bound * fastTanh(mu[2] / bound)
  end
  local releaseFront = clamp(0.5 * (fastTanh(mu[1]) + 1), 0, 1)
  local releaseRear = clamp(0.5 * (fastTanh(mu[2]) + 1), 0, 1)
  local brakeFront = math.max(0.01, 1 - releaseFront)
  local brakeRear = math.max(0.01, 1 - releaseRear)

  if traceOn and traceCount < TRACE_MAX then
    local o = traceCount * TRACE_N
    for j = 1, 35 do traceBuf[o + j] = rawObs[j] end
    traceBuf[o + 36] = releaseFront; traceBuf[o + 37] = releaseRear
    traceCount = traceCount + 1
  end
  cmd[1], cmd[2], cmd[3], cmd[4] = brakeFront, brakeFront, brakeRear, brakeRear
  prevReleaseFront, prevReleaseRear = releaseFront, releaseRear
end

local function applyBrakes()
  local maxBrake = math.max(cmd[1], cmd[2], cmd[3], cmd[4])
  if input then
    input.brake = maxBrake
    lastWrittenBrake = maxBrake
  end
  if haveTelem and extensions and extensions.abstelemetry
     and extensions.abstelemetry.setBrakes then
    extensions.abstelemetry.setBrakes(cmd[1], cmd[2], cmd[3], cmd[4])
    return
  end
  if wheelCount < 4 then return end
  if electrics and electrics.values then electrics.values.brake = maxBrake end
  for i = 1, 4 do
    local wheel = wheels.wheelRotators[i - 1]
    local command = cmd[wheelToBrakeMap[i]] or 0
    local original = origBrakeTorque[i] or 0
    if maxBrake > 0.001 then
      wheel.brakeTorque = original * command / maxBrake
    else
      wheel.brakeTorque = 0
    end
  end
end

local function restoreBrakes()
  if haveTelem and extensions and extensions.abstelemetry
     and extensions.abstelemetry.releaseBrakes then
    extensions.abstelemetry.releaseBrakes()
    return
  end
  for i = 1, wheelCount do
    if origBrakeTorque[i] then
      wheels.wheelRotators[i - 1].brakeTorque = origBrakeTorque[i]
    end
  end
end

local function onEngage()
  active = true
  prevReleaseFront, prevReleaseRear = 0, 0
  cmd[1], cmd[2], cmd[3], cmd[4] = 1, 1, 1, 1
  resetStack()
  if haveTelem and extensions.abstelemetry.resetAccum then
    pcall(extensions.abstelemetry.resetAccum)
  end
end

local function onDisengage()
  active = false
  restoreBrakes()
  lastWrittenBrake = -1
  driverBrakeHeld = false
end

local function runTick()
  local started = os.clock()
  local data = nil
  if haveTelem and extensions and extensions.abstelemetry
     and extensions.abstelemetry.buildSensorData then
    local ok, result = pcall(extensions.abstelemetry.buildSensorData)
    if ok and type(result) == "table" then data = result end
  end
  if data == nil then
    if not warnedNoData then
      warnedNoData = true
      print("[MTB-ML-ABS-CoSim] !!! sensor producer unavailable; stock brakes retained")
    end
    return
  end
  buildObservation(data)
  forward()
  applyBrakes()
  mlabsTicks = mlabsTicks + 1
  tickMsSum = tickMsSum + (os.clock() - started) * 1000
  tickMsCount = tickMsCount + 1
end

local function init(jbeamData)
  weightsModule = "controller/" .. ((jbeamData and jbeamData.weights)
    or "mtb_ml_weights")
  active = false
  driverBrakeHeld = false
  lastWrittenBrake = -1
  timeAccum = 0
  mlabsTicks = 0
  tickMsSum, tickMsCount = 0, 0
  warnedMissing, warnedNoData = false, false
  prevReleaseFront, prevReleaseRear = 0, 0
  for i = 1, OBS_DIM do rawObs[i] = 0 end
  stack = {}
  for f = 1, N_STACK do
    local row = {}
    for k = 1, OBS_DIM do row[k] = 0 end
    stack[f] = row
  end
  for i = 1, FLAT_DIM do flat[i], normObs[i] = 0, 0 end
  prevWs = nil

  origBrakeTorque = {}
  wheelCount = wheels and wheels.wheelRotatorCount or 0
  for i = 0, wheelCount - 1 do
    origBrakeTorque[i + 1] = wheels.wheelRotators[i].brakeTorque or 0
  end

  local okT = pcall(function() extensions.load("abstelemetry") end)
  haveTelem = okT and extensions and extensions.abstelemetry ~= nil
  loadWeights()

  if electrics and electrics.values then
    electrics.values.mlabs_active = 0
    electrics.values.mlabs_ticks = 0
    electrics.values.mlabs_tick_ms = 0
    electrics.values.mlabs_loaded = weightsOk and 1 or 0
    electrics.values.mlabs_interface = "cosim_axle_release_v2"
  end
end

local function update(dtPhys)
  if not dtPhys or dtPhys <= 0 then return end
  local speed = 0
  local velocity = obj:getVelocity()
  if velocity then
    speed = math.sqrt((velocity.x or 0) ^ 2 + (velocity.y or 0) ^ 2
      + (velocity.z or 0) ^ 2)
  end
  local rawBrake = (input and input.brake) or 0
  local driverBrake
  if active and lastWrittenBrake >= 0
     and math.abs(rawBrake - lastWrittenBrake) < 1e-6 then
    driverBrake = driverBrakeHeld and 1 or 0
  else
    driverBrake = rawBrake
  end

  if not active then
    if driverBrake > ENGAGE_BRAKE_THRESH and speed > ENGAGE_SPEED_MS then
      onEngage()
      buildLiveWsMap()
      traceOn = true
      traceCount = 0
      driverBrakeHeld = true
      lastWrittenBrake = -1
      -- Co-sim applies the first action at the handoff packet, not one control
      -- period later. Seed one full period so deployment also acts immediately
      -- on the engagement physics update.
      timeAccum = TICK_STEP
    end
  else
    if not (lastWrittenBrake >= 0
            and math.abs(rawBrake - lastWrittenBrake) < 1e-6) then
      driverBrakeHeld = rawBrake >= 0.05
    end
    if not driverBrakeHeld or speed < DISENGAGE_SPEED_MS then
      if traceOn and traceCount > 0 then
        traceSeq = traceSeq + 1
        local fh = io.open("mlabs_trace_" .. traceSeq .. ".csv", "w")
        if fh then
          fh:write("tick,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31,c32,c33,c34,c35,relF,relR\n")
          local parts = {}
          for k = 0, traceCount - 1 do
            local o = k * TRACE_N
            parts[1] = tostring(k + 1)
            for j = 1, TRACE_N do
              parts[j + 1] = string.format("%.6g", traceBuf[o + j])
            end
            fh:write(table.concat(parts, ",") .. "\n")
          end
          fh:close()
        end
        traceOn = false
      end
      onDisengage()
    end
  end

  if active and weightsOk then
    timeAccum = timeAccum + dtPhys
    while timeAccum >= TICK_STEP do
      runTick()
      timeAccum = timeAccum - TICK_STEP
    end
    applyBrakes()
  end
  if electrics and electrics.values then
    electrics.values.mlabs_active = active and 1 or 0
    electrics.values.mlabs_ticks = mlabsTicks
    electrics.values.mlabs_loaded = weightsOk and 1 or 0
    if tickMsCount > 0 then
      electrics.values.mlabs_tick_ms = tickMsSum / tickMsCount
    end
  end
end

local function reset(jbeamData)
  init(jbeamData)
end

M.init = init
M.update = update
M.reset = reset

return M
