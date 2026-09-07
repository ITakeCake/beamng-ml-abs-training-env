local M = {}
M.type = "auxiliary"

-- =====================================================================
-- MTB-ML-ABS.lua, ML-policy ABS controller for the etk800 wagon
-- =====================================================================
-- Observation, action, and normalization contract mirrors abs_env.py.
-- Physics-rate update self-subdivides to 200 Hz control ticks.
-- Actuation via abstelemetry.setBrakes(fr,fl,rr,rl).
-- =====================================================================

-- ---- timing ---------------------------------------------------------
local TICK_RATE_HZ = 200
local TICK_STEP    = 1 / TICK_RATE_HZ      -- 0.005 s, == abs_env DETERM_HZ dt
local timeAccum    = 0

-- ---- obs / stack / model dims --------------------------------------
local OBS_DIM   = 27
local N_STACK   = 16
local FLAT_DIM  = OBS_DIM * N_STACK        -- 432
local ACT_DIM   = 4

-- ---- wheel mapping --------------------------------------------------
-- logical wheel i (RR,RL,FR,FL) -> brake command slot (FR,FL,RR,RL)
local wheelToBrakeMap = {3, 4, 1, 2}
local origBrakeTorque = {}                 -- [1..4] by wheelRotator index
local wheelCount      = 0

-- ---- engage state ---------------------------------------------------
local ENGAGE_BRAKE_THRESH = 0.9            -- driver brake input must exceed this
local ENGAGE_SPEED_MS     = 8.0            -- and airspeed must exceed this (m/s)
local DISENGAGE_SPEED_MS  = 0.5            -- below this -> disengage (stopped)
local active        = false
local mlabsTicks    = 0                    -- total 200Hz ticks while active
-- Driver-pedal latch: controller overwrites input.brake with maxBrake.
-- lastWrittenBrake tracks what we wrote vs genuine driver changes.
local lastWrittenBrake = -1
local driverBrakeHeld  = false             -- latched driver intent while active

-- ---- engage warmup (S1 fix: kill the accel->brake transition poll-window spike)
-- Warmup: first WARMUP_TICKS ticks discard the accel-to-brake transition window.
-- Applies gentle WARMUP_BRAKE instead of full torque during the flush.
local WARMUP_TICKS  = 2
local WARMUP_BRAKE  = 0.05
local pendingWarmup = 0

-- ---- weights module (loaded lazily, fails loud if missing) ----------
-- Weights module is configurable per jbeam part via jbeamData.weights.
local weightsModule = "controller/mtb_ml_weights"
local W = nil                              -- the weights table
local weightsOk = false
local warnedMissing = false
local warnedNoData  = false                -- one-shot warn if buildSensorData() is missing

-- ---- kinematic-g accumulators (physics-rate, consumed each 200Hz tick)
local kPrevSpeed   = -1                    -- speed at the previous physics step
local kSpeedStart  = -1                    -- speed at the first substep of this tick window
local kDtSum       = 0                     -- summed physics dt in this tick window
local kGyRollSize  = 6                     -- rolling-mean window for gy_min/gy_max (abstelemetry.lua:43 pollGyRollingSize=6)
local kGyRollBuf   = {}
local kGyRollIdx   = 0
local kGyFrames    = 0
local kGySmoothMin = 999999                -- sentinel init per abstelemetry.lua:44-45 (NOT 0: braking decels are positive,
local kGySmoothMax = -999999               -- a 0 floor would pin gy_min to 0 forever and feed the policy a wrong channel)

-- ---- per-tick obs derivative state (rates / wheel accels) ----------
local hasPrevState = false
local prevWsFR, prevWsFL, prevWsRR, prevWsRL = 0, 0, 0, 0
local prevPitch, prevRoll = 0, 0

-- ---- prev_brakes (commanded fraction from previous tick) -----------
-- Per spec: 0.0 on the FIRST tick of an episode (NOT 0.01).
local prevBrakeFR, prevBrakeFL, prevBrakeRR, prevBrakeRL = 0, 0, 0, 0

-- ---- preallocated buffers (NO per-tick allocation) -----------------
local rawObs   = {}                        -- [1..27] current raw obs
local stack    = {}                        -- ring: stack[f][k], f=1..16 (1=oldest)
local flat     = {}                        -- [1..432] flattened oldest-first
local normFlat = {}                        -- [1..432] normalized
local actBuf   = {}                        -- preallocated per-layer activations
local cmd      = {0, 0, 0, 0}              -- brake command slots (FR,FL,RR,RL)

-- ---- DEBUG INSTRUMENTATION (OFF by default) ------------------------
-- Debug: publishes raw obs and cmds to electrics when mlabs_debug = 1.
M.debugObs = false
local DBG_OBS_KEYS = {}                     -- ["mlabs_o0".."mlabs_o26"]
local DBG_CMD_KEYS = {}                     -- ["mlabs_c0".."mlabs_c3"]
for k = 1, OBS_DIM do DBG_OBS_KEYS[k] = "mlabs_o" .. (k - 1) end
for k = 1, ACT_DIM do DBG_CMD_KEYS[k] = "mlabs_c" .. (k - 1) end

-- ---- EXT-MODE (in-car training mailbox; OFF by default) -------------
-- Ext-mode: Python mailboxes brake actions, skipping the in-Lua forward pass.
-- Makes the trained loop identical to the deployed loop.
M.extMode      = false
M.deployLatencyTicks = 0                    -- one-tick output buffer at deploy if probe measures k=1 (spec §2)
local extCmd        = {WARMUP_BRAKE, WARMUP_BRAKE, WARMUP_BRAKE, WARMUP_BRAKE}  -- mailboxed action (FR,FL,RR,RL)
local extSeqPending = 0                      -- last seq handed in by setExtCmd
local extSeqApplied = 0                      -- seq echoed back after a tick applies it
local extTickSeq    = 0                      -- monotonic real-tick count (post-warmup ticks only)

-- ---- obs clip bounds (abs_env.py:251-272) --------------------------
local PI = math.pi
local OBS_LOW = {
  0.0, 0.0, 0.0, 0.0,
  -50.0, -50.0, -10.0,
  0.0, 0.0, 0.0, 0.0,
  -50.0, -50.0,
  0.0, -2.0, -720.0,   -- (PPO_V2) steering now measured wheel angle in DEG (electrics.steering)
  -PI, -PI,
  0.0, 0.0,
  -50.0, -10.0, -10.0,
  -200.0, -200.0, -200.0, -200.0,
}
local OBS_HIGH = {
  100.0, 100.0, 100.0, 100.0,
  50.0, 50.0, 10.0,
  1.0, 1.0, 1.0, 1.0,
  50.0, 50.0,
  15000.0, 12.0, 720.0,   -- (PPO_V2) steering deg
  PI, PI,
  1.0, 1.0,
  50.0, 10.0, 10.0,
  200.0, 200.0, 200.0, 200.0,
}

-- ---- normalization constants (train.py / vec_normalize.py) ---------
local NORM_EPS  = 1e-8
local CLIP_OBS  = 10.0

-- ---- budget timer ---------------------------------------------------
local tickMsSum   = 0
local tickMsCount = 0


-- =====================================================================
-- Helpers
-- =====================================================================

local function clampNum(x, lo, hi)
  if x ~= x then return 0 end            -- NaN -> 0 (mirror np.nan_to_num)
  if x == math.huge then x = 1e6 end
  if x == -math.huge then x = -1e6 end
  if x < lo then return lo end
  if x > hi then return hi end
  return x
end

local function fastTanh(x)
  -- LuaJIT has no math.tanh in all builds; use a stable explicit form.
  if x > 20 then return 1.0 end
  if x < -20 then return -1.0 end
  local e2 = math.exp(2 * x)
  return (e2 - 1) / (e2 + 1)
end

-- =====================================================================
-- EXT-MODE setters (called from Python via queue_lua_command). Not hot-path.
-- =====================================================================
-- Clamp one brake float to the SAME range abs_env.step() maps actions into
-- ([0.01, 1.0]); NaN -> WARMUP_BRAKE (gentle, never a fabricated lock).
local function clampBrake(x)
  if type(x) ~= "number" or x ~= x then return WARMUP_BRAKE end
  if x < 0.01 then return 0.01 elseif x > 1.0 then return 1.0 end
  return x
end

-- Enable/disable in-car training ext mode. Forces debugObs on (the obs-parity
-- publish path the env reads). Does NOT touch active/engage state.
function M.setExtMode(on)
  M.extMode = (on == true) or (on == 1)
  if M.extMode then
    M.debugObs = true                       -- env reads mlabs_o*/mlabs_c*; keep publishing
    electrics.values.mlabs_debug = 1
  end
  electrics.values.mlabs_extmode = M.extMode and 1 or 0
end

-- Mailbox a 4-float brake action (FR,FL,RR,RL) + a monotonic seq. The action
-- is clamped to [0.01,1.0] HERE so the stored extCmd is always deploy-legal;
-- runTick consumes it once and echoes seq into mlabs_seq.
function M.setExtCmd(fr, fl, rr, rl, seq)
  extCmd[1] = clampBrake(fr)
  extCmd[2] = clampBrake(fl)
  extCmd[3] = clampBrake(rr)
  extCmd[4] = clampBrake(rl)
  if type(seq) == "number" then extSeqPending = seq end
end

-- =====================================================================
-- Hoisted sensor readers. No closures, no per-tick allocation.
-- =====================================================================
local mathSqrt  = math.sqrt
local mathAtan2 = math.atan2

-- pcall target: returns the honest ESC yaw rate (rad/s).
local function readYawRate()
  return obj:getYawAngularVelocity() or 0
end

-- pcall target: returns pitch, roll (rad) from the direction + up vectors.
-- Returns two values; pcall forwards them on success.
local function readPitchRoll()
  local d = obj:getDirectionVector()
  if not d then return 0, 0 end
  local pitch = mathAtan2(d.z, mathSqrt(d.x * d.x + d.y * d.y))
  local roll = 0
  local u = obj:getDirectionVectorUp()
  if u then
    local rightZ = d.x * u.y - d.y * u.x
    roll = mathAtan2(rightZ, u.z)
  end
  return pitch, roll
end


-- =====================================================================
-- Weights module loading + buffer preallocation
-- =====================================================================
-- STRING-BLOB format: weight payloads in long strings, parsed once at init.
-- Preallocated Lua tables, verified counts match rows*cols.

-- Parse a string of space-separated floats into a preallocated number table.
-- Returns (table, count). count lets the caller verify the expected length.
local function parseFloats(s)
  local t = {}
  local n = 0
  for tok in string.gmatch(s, "%S+") do
    local v = tonumber(tok)
    if v == nil then return nil, n end   -- non-numeric token -> fail loud upstream
    n = n + 1
    t[n] = v
  end
  return t, n
end

-- Parse a row-major flat float list (one string, or concatenated chunks) into a
-- 2D table W[r][c] with r in 1..rows, c in 1..cols. Returns (W2d, totalCount).
-- Reads sequentially so chunk boundaries (split on whole rows) are transparent.
local function parseMatrix(strList, rows, cols)
  local W2d = {}
  local r = 1
  local c = 0
  local total = 0
  W2d[1] = {}
  for _, s in ipairs(strList) do
    for tok in string.gmatch(s, "%S+") do
      local v = tonumber(tok)
      if v == nil then return nil, total end
      if c >= cols then
        r = r + 1
        c = 0
        if r > rows then return nil, total + 1 end   -- overflow: too many floats
        W2d[r] = {}
      end
      c = c + 1
      total = total + 1
      W2d[r][c] = v
    end
  end
  return W2d, total
end

local function loadWeights()
  -- BeamNG vehicle-lua require uses forward-slash subdir paths (see
  -- engine controller.lua:460-499: require("controller/" .. fileName)).
  local ok, mod = pcall(require, weightsModule)
  if not ok or type(mod) ~= "table" then
    weightsOk = false
    W = nil
    if not warnedMissing then
      warnedMissing = true
      print("[MTB-ML-ABS] !!! FATAL: weights module '" .. weightsModule .. "' "
        .. "is MISSING or invalid -> ML ABS DISABLED, brakes NOT written. err="
        .. tostring(mod))
    end
    return false
  end

  -- Build the parsed table we hand to the forward pass.
  local parsed = { layers = {} }

  -- --- obs_mean / obs_var (string -> number table, length-checked) ---
  if type(mod.obs_mean_s) ~= "string" then
    print("[MTB-ML-ABS] !!! FATAL: weights.obs_mean_s missing or not a string")
    weightsOk = false; W = nil; return false
  end
  local omean, nmean = parseFloats(mod.obs_mean_s)
  if omean == nil or nmean ~= FLAT_DIM then
    print("[MTB-ML-ABS] !!! FATAL: obs_mean parse count " .. tostring(nmean)
      .. " != " .. FLAT_DIM .. " (or non-numeric token)")
    weightsOk = false; W = nil; return false
  end
  parsed.obs_mean = omean

  if type(mod.obs_var_s) ~= "string" then
    print("[MTB-ML-ABS] !!! FATAL: weights.obs_var_s missing or not a string")
    weightsOk = false; W = nil; return false
  end
  local ovar, nvar = parseFloats(mod.obs_var_s)
  if ovar == nil or nvar ~= FLAT_DIM then
    print("[MTB-ML-ABS] !!! FATAL: obs_var parse count " .. tostring(nvar)
      .. " != " .. FLAT_DIM .. " (or non-numeric token)")
    weightsOk = false; W = nil; return false
  end
  parsed.obs_var = ovar

  -- --- layers ---
  if type(mod.layers) ~= "table" or #mod.layers == 0 then
    print("[MTB-ML-ABS] !!! FATAL: weights.layers missing/empty")
    weightsOk = false; W = nil; return false
  end

  -- Validate per-layer dims, parse strings, preallocate activation buffers.
  local inDim = FLAT_DIM
  actBuf = {}
  local nLayers = #mod.layers
  for li, L in ipairs(mod.layers) do
    local rows = L.rows
    local cols = L.cols
    if type(rows) ~= "number" or type(cols) ~= "number" then
      print("[MTB-ML-ABS] !!! FATAL: layer " .. li .. " missing rows/cols")
      weightsOk = false; W = nil; return false
    end
    if cols ~= inDim then
      print("[MTB-ML-ABS] !!! FATAL: layer " .. li .. " cols(" .. cols
        .. ") != in dim(" .. inDim .. ")")
      weightsOk = false; W = nil; return false
    end
    -- bias string
    if type(L.b_s) ~= "string" then
      print("[MTB-ML-ABS] !!! FATAL: layer " .. li .. " missing b_s string")
      weightsOk = false; W = nil; return false
    end
    local b, nb = parseFloats(L.b_s)
    if b == nil or nb ~= rows then
      print("[MTB-ML-ABS] !!! FATAL: layer " .. li .. " b parse count "
        .. tostring(nb) .. " != rows(" .. rows .. ")")
      weightsOk = false; W = nil; return false
    end
    -- weight matrix: one string (W_s) or chunked (W_chunks)
    local strList
    if type(L.W_s) == "string" then
      strList = { L.W_s }
    elseif type(L.W_chunks) == "table" and #L.W_chunks > 0 then
      strList = L.W_chunks
    else
      print("[MTB-ML-ABS] !!! FATAL: layer " .. li .. " missing W_s/W_chunks")
      weightsOk = false; W = nil; return false
    end
    local Wmat, nW = parseMatrix(strList, rows, cols)
    if Wmat == nil or nW ~= rows * cols then
      print("[MTB-ML-ABS] !!! FATAL: layer " .. li .. " W parse count "
        .. tostring(nW) .. " != rows*cols(" .. (rows * cols) .. ")")
      weightsOk = false; W = nil; return false
    end
    -- final shape guard: last row fully populated
    if type(Wmat[rows]) ~= "table" or #Wmat[rows] ~= cols then
      print("[MTB-ML-ABS] !!! FATAL: layer " .. li .. " W row " .. rows
        .. " has wrong col count")
      weightsOk = false; W = nil; return false
    end

    -- Activation is POSITIONAL: ReLU hidden, linear final (mu head).
    local layer = { _W = Wmat, b = b,
                    _act = (li < nLayers) and "relu" or "linear" }
    parsed.layers[li] = layer

    -- one preallocated output buffer per layer
    local buf = {}
    for o = 1, rows do buf[o] = 0 end
    actBuf[li] = buf
    inDim = rows
  end
  if inDim ~= ACT_DIM then
    print("[MTB-ML-ABS] !!! FATAL: final layer out dim(" .. inDim
      .. ") != action dim(" .. ACT_DIM .. ")")
    weightsOk = false; W = nil; return false
  end

  W = parsed
  weightsOk = true
  print("[MTB-ML-ABS] weights loaded OK (string-blob): " .. #parsed.layers
    .. " layers, flat=" .. FLAT_DIM .. ", actions=" .. ACT_DIM)
  return true
end


-- =====================================================================
-- Preallocate fixed buffers (independent of weights)
-- =====================================================================
local function preallocBuffers()
  for k = 1, OBS_DIM do rawObs[k] = 0 end
  stack = {}
  for f = 1, N_STACK do
    local row = {}
    for k = 1, OBS_DIM do row[k] = 0 end
    stack[f] = row
  end
  for k = 1, FLAT_DIM do flat[k] = 0; normFlat[k] = 0 end
  kGyRollBuf = {}
  for k = 1, kGyRollSize do kGyRollBuf[k] = 0 end
end


-- =====================================================================
-- Stack management, oldest FIRST (row 1), newest LAST (row N_STACK)
-- =====================================================================
local function resetStack()
  for f = 1, N_STACK do
    local row = stack[f]
    for k = 1, OBS_DIM do row[k] = 0 end
  end
end

-- push the current rawObs[] into the newest slot, shifting older toward front
local function pushStack()
  -- shift rows 2..N down to 1..N-1 (copy values, no realloc)
  for f = 1, N_STACK - 1 do
    local dst = stack[f]
    local src = stack[f + 1]
    for k = 1, OBS_DIM do dst[k] = src[k] end
  end
  local newest = stack[N_STACK]
  for k = 1, OBS_DIM do newest[k] = rawObs[k] end
end

-- flatten oldest-first/newest-last into flat[1..432]
local function flattenStack()
  local idx = 0
  for f = 1, N_STACK do
    local row = stack[f]
    for k = 1, OBS_DIM do
      idx = idx + 1
      flat[idx] = row[k]
    end
  end
end

-- VecNormalize: (x - mean)/sqrt(var+eps), clipped to [-CLIP_OBS, CLIP_OBS]
local function normalizeStack()
  local mean = W.obs_mean
  local var  = W.obs_var
  for i = 1, FLAT_DIM do
    local v = (flat[i] - mean[i]) / math.sqrt(var[i] + NORM_EPS)
    if v < -CLIP_OBS then v = -CLIP_OBS elseif v > CLIP_OBS then v = CLIP_OBS end
    normFlat[i] = v
  end
end


-- =====================================================================
-- Forward pass, plain Lua MLP over preallocated buffers
-- =====================================================================
-- Returns nothing; final layer output lands in actBuf[#layers][1..4] (the
-- pre-squash actor mean mu). Caller applies tanh + unscale.
local function forward()
  local layers = W.layers
  local input = normFlat
  for li = 1, #layers do
    local L = layers[li]
    local w = L._W                 -- canonicalized matrix (capital W from exporter)
    local b = L.b
    local out = actBuf[li]
    local act = L._act             -- positional: "relu" hidden, "linear" final
    local nOut = #b
    for o = 1, nOut do
      local wr = w[o]
      local s = b[o]
      for i = 1, #wr do
        s = s + wr[i] * input[i]
      end
      if act == "relu" then
        if s < 0 then s = 0 end
      elseif act == "tanh" then
        s = fastTanh(s)
      end
      -- "linear" / nil: leave as-is
      out[o] = s
    end
    input = out
  end
end


-- =====================================================================
-- Build the 27-dim raw obs for the current 200Hz tick, FROM readAll DATA
-- =====================================================================
-- Build raw obs from abstelemetry.buildSensorData() (the training producer).
-- prev_brakes and derivatives stay controller-local.
local function getf(t, k)
  local v = t[k]
  if type(v) == "number" then return v end
  return 0
end

local function buildRawObsFromData(data)
  -- ---- wheel speeds (env reads ws fields: ws3=FR, ws4=FL, ws1=RR, ws2=RL) ----
  local wsFR = math.abs(getf(data, "ws3"))
  local wsFL = math.abs(getf(data, "ws4"))
  local wsRR = math.abs(getf(data, "ws1"))
  local wsRL = math.abs(getf(data, "ws2"))

  -- ---- g-channels: poll-window kinematic avg + smoothed min/max (det+live) ----
  local gyAvg = getf(data, "gy_avg")
  local gyMin = getf(data, "gy_min")
  local gyMax = getf(data, "gy_max")
  local gxInst = getf(data, "gx_inst")
  local gzInst = getf(data, "gz_inst")

  -- ---- yaw: poll-window MEAN (env uses yaw_avg, NOT the instantaneous read) ----
  local yawRate = getf(data, "yaw_avg")

  -- ---- CAN-ish channels (sourced identically to the env) ----
  local rpm   = getf(data, "rpm")
  local gear  = getf(data, "gear")
  local steering = getf(data, "steering")   -- = electrics.steering_input or 0 (≈0 straight-line)

  -- ---- pitch / roll (env reads data.pitch / data.roll) ----
  local pitch = getf(data, "pitch")
  local roll  = getf(data, "roll")

  -- ---- driver pedals (env reads data.input_brake / data.input_throttle) ----
  local inBrake    = getf(data, "input_brake")
  local inThrottle = getf(data, "input_throttle")

  -- ---- derivatives (rates / wheel accels), 0 on first tick. Computed from the
  --      SAME data-sourced values the env differenced (dt = 1/200s). ----
  local waFR, waFL, waRR, waRL = 0, 0, 0, 0
  local pitchRate, rollRate = 0, 0
  if hasPrevState then
    local dt = TICK_STEP
    waFR = (wsFR - prevWsFR) / dt
    waFL = (wsFL - prevWsFL) / dt
    waRR = (wsRR - prevWsRR) / dt
    waRL = (wsRL - prevWsRL) / dt
    pitchRate = (pitch - prevPitch) / dt
    rollRate  = (roll  - prevRoll)  / dt
  end
  prevWsFR, prevWsFL, prevWsRR, prevWsRL = wsFR, wsFL, wsRR, wsRL
  prevPitch, prevRoll = pitch, roll
  hasPrevState = true

  -- ---- assemble raw[1..27] in the abs_env order ----
  rawObs[1]  = wsFR
  rawObs[2]  = wsFL
  rawObs[3]  = wsRR
  rawObs[4]  = wsRL
  rawObs[5]  = gyAvg
  rawObs[6]  = gxInst
  rawObs[7]  = yawRate
  rawObs[8]  = prevBrakeFR
  rawObs[9]  = prevBrakeFL
  rawObs[10] = prevBrakeRR
  rawObs[11] = prevBrakeRL
  rawObs[12] = gyMin
  rawObs[13] = gyMax
  rawObs[14] = rpm
  rawObs[15] = gear
  rawObs[16] = steering
  rawObs[17] = pitch
  rawObs[18] = roll
  rawObs[19] = inBrake
  rawObs[20] = inThrottle
  rawObs[21] = gzInst
  rawObs[22] = pitchRate
  rawObs[23] = rollRate
  rawObs[24] = waFR
  rawObs[25] = waFL
  rawObs[26] = waRR
  rawObs[27] = waRL

  -- ---- clip to sanity bounds (np.nan_to_num + clip) ----
  for k = 1, OBS_DIM do
    rawObs[k] = clampNum(rawObs[k], OBS_LOW[k], OBS_HIGH[k])
  end
end


-- =====================================================================
-- Apply per-wheel brakes (capacity-write pattern, same as abstelemetry)
-- cmd[] is in command-slot order FR,FL,RR,RL.
-- =====================================================================
local haveTelem = false

local function applyBrakesViaTelem()
  -- ---------------------------------------------------------------------
  -- Forces input.brake = maxBrake so deployed torque matches training.
  -- Without this, input.brake = 1.0 inflates every wheel by 1/maxBrake.
  -- ---------------------------------------------------------------------
  local maxBrake = math.max(cmd[1], cmd[2], cmd[3], cmd[4])
  if input then
    input.brake = maxBrake
    lastWrittenBrake = maxBrake          -- latch what we wrote (driver-intent guard)
  end

  if haveTelem and extensions and extensions.abstelemetry
     and extensions.abstelemetry.setBrakes then
    -- setBrakes signature: (fr, fl, rr, rl) == (cmd1, cmd2, cmd3, cmd4)
    extensions.abstelemetry.setBrakes(cmd[1], cmd[2], cmd[3], cmd[4])
    return
  end
  -- Fallback: write capacities directly (mirrors applyPerWheelBrakes math).
  if wheelCount < 4 then return end
  electrics.values.brake = maxBrake
  for i = 1, 4 do
    local wd = wheels.wheelRotators[i - 1]
    local cmdIdx = wheelToBrakeMap[i]
    local ot = origBrakeTorque[i] or 0
    if ot > 0 then
      if maxBrake > 0.001 then
        wd.brakeTorque = ot * (cmd[cmdIdx] or 0) / maxBrake
      else
        wd.brakeTorque = 0
      end
    end
  end
end

-- Restore full braking capacity to all wheels (hand back to stock pipeline).
local function restoreBrakes()
  if haveTelem and extensions and extensions.abstelemetry
     and extensions.abstelemetry.releaseBrakes then
    extensions.abstelemetry.releaseBrakes()
    return
  end
  if wheelCount < 4 then return end
  for i = 1, 4 do
    local wd = wheels.wheelRotators[i - 1]
    if origBrakeTorque[i] then wd.brakeTorque = origBrakeTorque[i] end
  end
end


-- =====================================================================
-- On engage: zero the stack + reset prev_brakes + clear rate state
-- =====================================================================
local function onEngage()
  active = true
  resetStack()
  prevBrakeFR, prevBrakeFL, prevBrakeRR, prevBrakeRL = 0, 0, 0, 0
  hasPrevState = false

  -- Reset abstelemetry poll window so first active tick gets clean decel only.
  if haveTelem and extensions and extensions.abstelemetry
     and extensions.abstelemetry.resetAccum then
    pcall(extensions.abstelemetry.resetAccum)
  end

  -- S1: arm the warmup-drain. The first WARMUP_TICKS runTicks will flush the
  -- accel->brake transition window instead of feeding it to the policy.
  pendingWarmup = WARMUP_TICKS
  -- seed cmd[] to the gentle warmup brake so the physics steps that elapse between
  -- this engage and the first runTick (timeAccum < TICK_STEP) re-assert a gentle,
  -- non-locking command rather than a stale last-episode command or hard zeros.
  cmd[1] = WARMUP_BRAKE; cmd[2] = WARMUP_BRAKE
  cmd[3] = WARMUP_BRAKE; cmd[4] = WARMUP_BRAKE

  -- Ext-mode handshake reset: mlabs_tickseq must be 0 during warmup.
  extTickSeq    = 0
  extSeqApplied = 0
  extSeqPending = 0
  extCmd[1] = WARMUP_BRAKE; extCmd[2] = WARMUP_BRAKE
  extCmd[3] = WARMUP_BRAKE; extCmd[4] = WARMUP_BRAKE
  if M.extMode and electrics and electrics.values then
    electrics.values.mlabs_seq     = 0
    electrics.values.mlabs_tickseq = 0
  end
end

local function onDisengage()
  active = false
  restoreBrakes()
  -- hand the brake pedal back to the driver / stock pipeline
  lastWrittenBrake = -1
  driverBrakeHeld = false
end


-- =====================================================================
-- 200Hz control tick
-- =====================================================================
local function runTick()
  local t0 = os.clock()

  -- ---- consume the EXACT producer of the training obs ----
  -- Consume the exact training obs producer. Also resets poll accumulators.
  local data = nil
  if haveTelem and extensions and extensions.abstelemetry
     and extensions.abstelemetry.buildSensorData then
    local okD, d = pcall(extensions.abstelemetry.buildSensorData)
    if okD and type(d) == "table" then data = d end
  end
  if data == nil then
    if not warnedNoData then
      warnedNoData = true
      print("[MTB-ML-ABS] !!! buildSensorData() unavailable -> obs source MISSING; "
        .. "brakes NOT written this engage (restoring stock pipeline).")
    end
    -- bail: keep stock braking, advance the tick counter for visibility.
    mlabsTicks = mlabsTicks + 1
    local t1 = os.clock()
    tickMsSum = tickMsSum + (t1 - t0) * 1000.0
    tickMsCount = tickMsCount + 1
    return
  end

  -- ---- S1 warmup-drain: discard the first WARMUP_TICKS post-engage windows ----
  -- Warmup drain: discard first WARMUP_TICKS windows, re-arm resetAccum.
  if pendingWarmup > 0 then
    pendingWarmup = pendingWarmup - 1
    if haveTelem and extensions and extensions.abstelemetry
       and extensions.abstelemetry.resetAccum then
      pcall(extensions.abstelemetry.resetAccum)
    end
    -- keep prev-state derivative seeds fresh: clear so the first real tick emits 0
    -- rates (matches abs_env's _has_prev_state=False at episode start).
    hasPrevState = false
    -- apply a gentle equal-per-wheel brake so the car decelerates without locking
    -- the unmodulated fronts during the flush (see WARMUP_BRAKE rationale above).
    cmd[1] = WARMUP_BRAKE; cmd[2] = WARMUP_BRAKE
    cmd[3] = WARMUP_BRAKE; cmd[4] = WARMUP_BRAKE
    applyBrakesViaTelem()
    mlabsTicks = mlabsTicks + 1
    local t1 = os.clock()
    tickMsSum = tickMsSum + (t1 - t0) * 1000.0
    tickMsCount = tickMsCount + 1
    return
  end

  buildRawObsFromData(data)
  pushStack()

  -- ---- EXT-MODE branch: Python mailboxes the action; NN forward is SKIPPED ----
  -- Ext-mode: publish obs before applying cmd. obs(t) carries action(t-1).
  if M.extMode then
    -- this is a real (post-warmup) tick: bump the monotonic tick counter FIRST so
    -- the env's published mlabs_tickseq reflects the obs it is about to read.
    extTickSeq = extTickSeq + 1
    extSeqApplied = extSeqPending          -- consume the pending mailbox seq

    -- publish obs + applied seq + tickseq (allocation-free; keys preallocated).
    -- Published BEFORE the cmd is applied so the env reads the obs window that the
    -- action(t) responds to (action(t) effect shows up in obs(t+1)).
    local ev = electrics.values
    for k = 1, OBS_DIM do ev[DBG_OBS_KEYS[k]] = rawObs[k] end

    -- apply the mailboxed action (clamped already in setExtCmd) into cmd[].
    cmd[1] = extCmd[1]; cmd[2] = extCmd[2]
    cmd[3] = extCmd[3]; cmd[4] = extCmd[4]

    -- prev_brakes for next tick = the applied action (logical FR,FL,RR,RL).
    prevBrakeFR = cmd[1]
    prevBrakeFL = cmd[2]
    prevBrakeRR = cmd[3]
    prevBrakeRL = cmd[4]

    applyBrakesViaTelem()

    -- publish the applied cmds + consumed seq + tickseq.
    for k = 1, ACT_DIM do ev[DBG_CMD_KEYS[k]] = cmd[k] end
    ev.mlabs_seq     = extSeqApplied
    ev.mlabs_tickseq = extTickSeq

    mlabsTicks = mlabsTicks + 1
    local t1 = os.clock()
    tickMsSum = tickMsSum + (t1 - t0) * 1000.0
    tickMsCount = tickMsCount + 1
    return
  end

  if not weightsOk then
    -- Defensive: never write torques without weights. Keep stock pipeline.
    mlabsTicks = mlabsTicks + 1
    local t1 = os.clock()
    tickMsSum = tickMsSum + (t1 - t0) * 1000.0
    tickMsCount = tickMsCount + 1
    return
  end

  flattenStack()
  normalizeStack()
  forward()

  -- Post-MLP head: sac_tanh01 or ppo_clip01, then brake = 0.01 + 0.99*a.
  local mu = actBuf[#W.layers]
  local headMode = W.head or "sac_tanh01"
  for j = 1, ACT_DIM do
    local a
    if headMode == "ppo_clip01" then
      a = mu[j]
      if a < 0.0 then a = 0.0 elseif a > 1.0 then a = 1.0 end
    else
      a = 0.5 * (fastTanh(mu[j]) + 1.0)            -- [0,1] (sac_tanh01)
    end
    local brake = 0.01 + 0.99 * a
    if brake < 0.01 then brake = 0.01 elseif brake > 1.0 then brake = 1.0 end
    -- action order: a[1]=FR, a[2]=FL, a[3]=RR, a[4]=RL
    -- fill command slots via the FR,FL,RR,RL command order directly.
    cmd[j] = brake
  end

  -- prev_brakes for next tick = the commanded fractions (logical FR,FL,RR,RL).
  prevBrakeFR = cmd[1]
  prevBrakeFL = cmd[2]
  prevBrakeRR = cmd[3]
  prevBrakeRL = cmd[4]

  applyBrakesViaTelem()

  -- ---- DEBUG: publish raw obs + commands to electrics (allocation-free) ----
  if M.debugObs then
    local ev = electrics.values
    for k = 1, OBS_DIM do ev[DBG_OBS_KEYS[k]] = rawObs[k] end
    for k = 1, ACT_DIM do ev[DBG_CMD_KEYS[k]] = cmd[k] end
  end

  mlabsTicks = mlabsTicks + 1

  local t1 = os.clock()
  tickMsSum = tickMsSum + (t1 - t0) * 1000.0
  tickMsCount = tickMsCount + 1
end


-- =====================================================================
-- init
-- =====================================================================
local function init(jbeamData)
  print("[MTB-ML-ABS] init, SAC ML ABS controller loading")
  -- Multi-model support: a jbeam part in MODEL_SLOT_TYPE can set
  -- {"weights": "mlabs_w_<run>"} to pick its own weights module. No override
  -- (nil) reproduces the original single-model default exactly.
  weightsModule = "controller/" .. ((jbeamData and jbeamData.weights) or "mtb_ml_weights")
  timeAccum = 0
  active = false
  mlabsTicks = 0
  warnedMissing = false

  -- capture per-wheel original brake torque capacities (idx 1=RR,2=RL,3=FR,4=FL)
  origBrakeTorque = {}
  wheelCount = wheels and wheels.wheelRotatorCount or 0
  for i = 0, wheelCount - 1 do
    origBrakeTorque[i + 1] = wheels.wheelRotators[i].brakeTorque or 0
  end

  -- reset all runtime state
  hasPrevState = false
  prevWsFR, prevWsFL, prevWsRR, prevWsRL = 0, 0, 0, 0
  prevPitch, prevRoll = 0, 0
  prevBrakeFR, prevBrakeFL, prevBrakeRR, prevBrakeRL = 0, 0, 0, 0
  kPrevSpeed = -1
  kSpeedStart = -1
  kDtSum = 0
  kGyFrames = 0
  kGyRollIdx = 0
  kGySmoothMin = 999999
  kGySmoothMax = -999999
  tickMsSum = 0
  tickMsCount = 0

  -- Ext-mode counters reset. M.extMode re-asserted by env after teleport.
  extSeqPending = 0
  extSeqApplied = 0
  extTickSeq    = 0
  extCmd[1] = WARMUP_BRAKE; extCmd[2] = WARMUP_BRAKE
  extCmd[3] = WARMUP_BRAKE; extCmd[4] = WARMUP_BRAKE

  preallocBuffers()
  loadWeights()           -- preallocates actBuf if successful

  -- expose status to electrics for external verification
  electrics.values.mlabs_active = 0
  electrics.values.mlabs_ticks = 0
  electrics.values.mlabs_tick_ms = 0
  electrics.values.mlabs_loaded = weightsOk and 1 or 0
  -- debug instrumentation toggle (0=off). Python sets to 1 to publish obs/cmds.
  if electrics.values.mlabs_debug == nil then electrics.values.mlabs_debug = 0 end
  -- ext-mode handshake electrics (exist from boot so the env never reads nil).
  electrics.values.mlabs_extmode = M.extMode and 1 or 0
  electrics.values.mlabs_warmup  = 0
  electrics.values.mlabs_seq     = 0
  electrics.values.mlabs_tickseq = 0
  electrics.values.mlabs_heading = obj:getDirection() or 0

  -- bring up the telemetry/actuation bridge (same as 1FEX), LAST.
  local okT = pcall(function() extensions.load('abstelemetry') end)
  haveTelem = okT and (extensions and extensions.abstelemetry ~= nil) or false
  if not haveTelem then
    print("[MTB-ML-ABS] WARN: abstelemetry not available; using direct brake-write fallback")
  end
end


-- =====================================================================
-- update(dtPhys), PHYSICS rate (~2000Hz). Self-subdivides to 200Hz.
-- =====================================================================
local function update(dtPhys)
  if dtPhys == nil or dtPhys <= 0 then return end

  -- ---- DEBUG toggle: mirror electrics.values.mlabs_debug into M.debugObs ----
  -- Lets Python flip instrumentation on/off at runtime with a single command.
  if electrics and electrics.values and electrics.values.mlabs_debug ~= nil then
    M.debugObs = (electrics.values.mlabs_debug == 1) or (electrics.values.mlabs_debug == true)
  end

  -- ---- body speed (for the engage/disengage state machine only) ----
  -- Obs channels now come from abstelemetry.buildSensorData().
  local speed = 0
  local vel = obj:getVelocity()
  if vel then
    local vx = vel.x or 0
    local vy = vel.y or 0
    local vz = vel.z or 0
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
  end

  -- ---- engage / disengage state machine ----
  -- Recover true driver pedal (we overwrite input.brake while active).
  local rawBrake = (input and input.brake) or 0
  local driverBrake
  if active and lastWrittenBrake >= 0 and math.abs(rawBrake - lastWrittenBrake) < 1e-6 then
    -- pedal untouched since our write -> driver is still holding the brake
    driverBrake = driverBrakeHeld and 1.0 or 0.0
  else
    -- genuine driver value (not our latch)
    driverBrake = rawBrake
  end

  if not active then
    local engageWant = (driverBrake > ENGAGE_BRAKE_THRESH) and (speed > ENGAGE_SPEED_MS)
    if engageWant then
      onEngage()
      driverBrakeHeld = true            -- latch: driver is braking
      lastWrittenBrake = -1             -- nothing written yet this cycle
      timeAccum = 0                     -- start fresh tick phase on engage
    end
  else
    -- update latched driver intent only from GENUINE driver reads
    if not (lastWrittenBrake >= 0 and math.abs(rawBrake - lastWrittenBrake) < 1e-6) then
      driverBrakeHeld = (rawBrake >= 0.05)
    end
    -- disengage on genuine driver release or near-stop
    if (not driverBrakeHeld) or speed < DISENGAGE_SPEED_MS then
      onDisengage()
    end
  end

  -- ---- 200Hz subdivision (only ticks while active) ----
  if active then
    timeAccum = timeAccum + dtPhys
    while timeAccum >= TICK_STEP do
      runTick()
      timeAccum = timeAccum - TICK_STEP
    end
    -- Re-assert per-wheel brakes every physics step (survive stock overwrite).
    if weightsOk or M.extMode then applyBrakesViaTelem() end
  end

  -- ---- publish status ----
  electrics.values.mlabs_active = active and 1 or 0
  electrics.values.mlabs_ticks = mlabsTicks
  electrics.values.mlabs_loaded = weightsOk and 1 or 0
  if tickMsCount > 0 then
    electrics.values.mlabs_tick_ms = tickMsSum / tickMsCount
  end

  -- ---- EXT-MODE per-physics-step status (spec §3) ----
  -- Ext-mode status: warmup flag, heading for crash terminal.
  electrics.values.mlabs_warmup  = (pendingWarmup > 0) and 1 or 0
  electrics.values.mlabs_extmode = M.extMode and 1 or 0
  electrics.values.mlabs_heading = obj:getDirection() or 0
end


local function reset(jbeamData)
  init(jbeamData)
end


-- =====================================================================
-- Exports
-- =====================================================================
M.init   = init
M.update = update
M.reset  = reset

return M
