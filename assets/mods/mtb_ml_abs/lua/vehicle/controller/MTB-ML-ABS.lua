local M = {}
M.type = "auxiliary"

-- =====================================================================
-- MTB-ML-ABS.lua  —  ML-policy ABS controller for the etk800 wagon
-- =====================================================================
-- Runs the SAC ABS policy that was trained in MachineTrainerBoy. The exact
-- observation / action / normalization contract is mirrored from:
--   abs_env.py            (_build_obs_from_data, 27-dim raw obs, clip bounds)
--   demo_collect.py       (FrameStacker: 16 frames, oldest-FIRST, newest-LAST)
--   train.py              (VecFrameStack(16) -> VecNormalize, norm_obs only)
--   abstelemetry.lua      (per-wheel brake actuation, wheel-order contract)
--   ABS-1FEX.lua          (physics-rate M.update + 200Hz self-subdivision)
--
-- Architecture:
--   * Dispatched as M.update at PHYSICS rate (~2000Hz) — same as 1FEX.
--   * Self-subdivides to a 200Hz control tick with a dt accumulator.
--   * Kinematic longitudinal g is computed from obj:getVelocity() deltas
--     accumulated over the physics substeps WITHIN each 200Hz tick (matches
--     abs_env's gy_avg = (pollSpeedStart - instSpeed) / pollDtSum exactly).
--   * Forward pass is plain Lua over weight tables loaded from
--     lua/vehicle/controller/mtb_ml_weights.lua (produced by the exporter).
--   * All activation buffers are preallocated ONCE at init — no per-tick alloc.
--   * Actuation is delegated to extensions.abstelemetry.setBrakes(fr,fl,rr,rl)
--     so the deployed controller and the trained policy share identical
--     brake semantics. On disengage we restore wd.ref.brakeTorque ourselves.
--
-- Wheel-order contract (load-bearing — see abs_env.py:938-941, ABS-1FEX:110):
--   wheelRotator / wheelData index -> corner:  1=RR, 2=RL, 3=FR, 4=FL
--   brake command slot order:                  1=FR, 2=FL, 3=RR, 4=RL
--   bridge wheelToBrakeMap = {3,4,1,2}  (logical wheel i -> command slot)
--   obs wheel-speed order (raw[0..3]):         FR, FL, RR, RL
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
-- Driver-pedal latch: while active, the controller OVERWRITES input.brake with
-- maxBrake (the actuation fix). That overwrite would otherwise corrupt the
-- engage/disengage logic (which keys off input.brake). lastWrittenBrake records
-- what WE last wrote; if input.brake still equals it, the driver hasn't touched
-- the pedal since -> treat intent as "still held". Any other value == genuine
-- driver change (release). -1 sentinel = we haven't written this engage cycle.
local lastWrittenBrake = -1
local driverBrakeHeld  = false             -- latched driver intent while active

-- ---- engage warmup (S1 fix: kill the accel->brake transition poll-window spike)
-- In abs_env, readAll() at reset (abs_env.py:575) DRAINS the entire acceleration
-- poll window right before handoff, so the policy's FIRST obs window spans only the
-- ~2 settle substeps after the throttle was cut — a small, DECEL-dominant window
-- (gy_max ~ +3.8, gy_avg ~ -0.2). In deployment NOTHING drains the window between
-- the accel phase and the first control tick: the first runTick consumed a window
-- that still spanned the throttle->brake transition with the car at peak speed and
-- brake torque still slewing in, yielding a huge ACCEL-signed g spike (gy_avg ~ -8,
-- gy_min ~ -12, gy_max ~ -9 — pure acceleration). That out-of-distribution first
-- frame poisoned the 16-frame stack and the policy ramped the FRONTS to lock within
-- ~5 ticks (det@80 0.84g, 27% front-lock).
--
-- Fix: for the first WARMUP_TICKS active ticks, BUILD the obs (which drains the
-- poll window) but DISCARD it (no stack push, no forward pass, no ML brake write),
-- and re-arm resetAccum so the NEXT window accumulates only clean post-engage decel.
-- The stock full-torque pipeline brakes during these <=2 ticks (~10ms) — identical
-- to training, where the car was already braking-decel before the first ML action.
-- WARMUP_TICKS flush windows; during them apply a GENTLE per-wheel brake (not the
-- stock full-torque pipeline, which would lock the unmodulated 3100Nm fronts at
-- ~37m/s within ~10ms before the policy ever acts, and not zero, which would stall
-- the car's decel). WARMUP_BRAKE mirrors the policy's own typical first action
-- (~0.05) so the car is in a gentle-decel state — exactly the world-state training's
-- first ML step inherited.
local WARMUP_TICKS  = 2
local WARMUP_BRAKE  = 0.05
local pendingWarmup = 0

-- ---- weights module (loaded lazily, fails loud if missing) ----------
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
-- When M.debugObs is true, every 200Hz tick publishes the 27 raw obs as
-- electrics mlabs_o0..mlabs_o26 and the 4 final brake commands as
-- mlabs_c0..mlabs_c3. Toggleable at runtime from Python via electrics:
--   vehicle.queue_lua_command("electrics.values.mlabs_debug = 1")
-- (update() copies that electrics flag into M.debugObs each physics step).
-- Allocation-free: the electrics key strings are preallocated ONCE here.
M.debugObs = false
local DBG_OBS_KEYS = {}                     -- ["mlabs_o0".."mlabs_o26"]
local DBG_CMD_KEYS = {}                     -- ["mlabs_c0".."mlabs_c3"]
for k = 1, OBS_DIM do DBG_OBS_KEYS[k] = "mlabs_o" .. (k - 1) end
for k = 1, ACT_DIM do DBG_CMD_KEYS[k] = "mlabs_c" .. (k - 1) end

-- ---- EXT-MODE (in-car training mailbox; OFF by default) -------------
-- When M.extMode is true, the controller does NOT run the in-Lua NN forward
-- pass. Instead a Python SAC env mailboxes a 4-float brake action per 200Hz
-- tick via M.setExtCmd(...); the controller assembles the obs EXACTLY as in
-- deployed NN mode (buildSensorData -> buildRawObsFromData -> pushStack ->
-- publish), then applies the mailboxed cmd. This makes the trained loop ==
-- the deployed loop (see BUILD_SPEC_incar_training.md). All ext-mode code is
-- guarded by M.extMode; when it is false the deployed NN path is byte-for-byte
-- unchanged. Allocation-free: extCmd is a preallocated 4-slot table; the
-- electrics key strings reuse DBG_OBS_KEYS/DBG_CMD_KEYS (already preallocated).
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
-- Module-scope sensor readers (hoisted out of buildRawObs so the 200Hz tick
-- does ZERO closure allocation). These reference only the vehicle-Lua globals
-- (obj, math) — no upvalues — so each is a single shared function object that
-- pcall() can call directly without building a closure every tick.
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
-- The weights module (produced by export_mlabs.py) is in STRING-BLOB format
-- (immune to LuaJIT's 65,536 number-constant-per-prototype cap: numeric weight
-- payloads live inside long string literals, which don't count against the
-- per-prototype number-constant budget). It returns a table:
--   mod.obs_dim=432, mod.act_dim=4, mod.clip_obs=<num>, mod.eps=<num>
--   mod.obs_mean_s, mod.obs_var_s   : strings of 432 space-separated floats
--   mod.layers = { {rows,cols, b_s="...", W_s="..." | W_chunks={...}}, ... }
--     W_s / W_chunks are ROW-MAJOR: element (r,c) at flat idx (r-1)*cols+c.
-- loadWeights() parses every string ONCE here at init into preallocated Lua
-- number tables, verifies counts == rows*cols / rows, and rebuilds the SAME
-- post-parse structure the forward pass expects:
--   W.obs_mean[1..432], W.obs_var[1..432]
--   W.layers[li]._W[r][c]  (row-major, 1-based), W.layers[li].b[o]
--   W.layers[li]._act      ("relu" hidden, "linear" final mu head)
-- Activation is POSITIONAL: layers 1..N-1 ReLU, final layer linear (mu);
-- tanh + unscale->[0,1] are applied by the controller AFTER the MLP.

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
  local ok, mod = pcall(require, "controller/mtb_ml_weights")
  if not ok or type(mod) ~= "table" then
    weightsOk = false
    W = nil
    if not warnedMissing then
      warnedMissing = true
      print("[MTB-ML-ABS] !!! FATAL: weights module 'controller/mtb_ml_weights' "
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
-- Stack management — oldest FIRST (row 1), newest LAST (row N_STACK)
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
-- Forward pass — plain Lua MLP over preallocated buffers
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
-- Build the 27-dim raw obs for the current 200Hz tick — FROM readAll DATA
-- =====================================================================
-- REWIRED 2026-06-05: instead of re-deriving every channel in Lua (which
-- diverged in closed loop — gy_min pinned by the engage transient, yaw using
-- the instantaneous ESC reading instead of the poll-window mean, etc.), we now
-- consume the EXACT same producer the policy was trained on:
--   extensions.abstelemetry.buildSensorData()  -- the raw Lua table that
--   readAll() JSON-encodes; abs_env._build_obs_from_data(data) was fed exactly
--   this dict over TCP once per 200Hz step.
-- buildSensorData() also RESETS the poll-window accumulators (pollSpeedStart,
-- pollDtSum, pollFrames, pollGy*, pollYaw*), so calling it once per tick gives
-- the same "window since the previous setBrakes" semantics as training (3RT:
-- setBrakes -> step -> readAll). We call it at the TOP of runTick, before the
-- new action is computed — the relative timing matches training exactly.
--
-- VERIFIED 2026-06-05 (probe_pollframes.py): abstelemetry.onPhysicsStep fires
-- and fills the poll accumulators in BOTH deterministic AND live mode on .tech
-- (poll_frames ~155/50ms live, ~20/readAll det; gy_avg/gy_min/gy_max/yaw_avg
-- all populated in both). So buildSensorData() is reliable in both clock modes —
-- no live-mode fallback needed. (The old 39-day-old "onPhysicsStep is det-only"
-- note did not hold for abstelemetry on this .tech build.)
--
-- prev_brakes (raw[8..11]) stays controller-local — it is the policy's own last
-- action, not a sensor. Derivatives (rates / wheel accels, raw[22..27]) also stay
-- controller-local: the env computes them from successive `data` dicts, so we
-- derive them from successive buildSensorData() values (the SAME source).
--
-- field -> obs mapping (abs_env._build_obs_from_data, dims 0-based -> raw 1-based):
--   raw1 ws_fr = |data.ws3|   raw2 ws_fl = |data.ws4|
--   raw3 ws_rr = |data.ws1|   raw4 ws_rl = |data.ws2|
--   raw5 gy_avg = data.gy_avg     raw6 gx = data.gx_inst   raw7 yaw = data.yaw_avg
--   raw8-11 prev_brakes (controller-local)
--   raw12 gy_min = data.gy_min    raw13 gy_max = data.gy_max
--   raw14 rpm = data.rpm  raw15 gear = data.gear  raw16 steering = data.steering
--   raw17 pitch = data.pitch  raw18 roll = data.roll
--   raw19 input_brake = data.input_brake  raw20 input_throttle = data.input_throttle
--   raw21 gz = data.gz_inst
--   raw22 pitch_rate  raw23 roll_rate  raw24-27 wheel accels (controller-local deriv)
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
  -- CRITICAL actuation contract (matches abs_env.py:620 training semantics):
  --   The stock brake pipeline applies  desiredBrakingTorque = capacity * pedal,
  --   where pedal == input.brake.  abstelemetry.setBrakes scales each wheel's
  --   capacity by  cmd/maxBrake.  So the NET per-wheel torque is
  --     origTorque * (cmd/maxBrake) * input.brake.
  --   For this to equal the trained target  origTorque * cmd  (which is exactly
  --   what training produced, because abs_env sets vehicle.control(brake=max(brakes))
  --   == maxBrake every step), the effective pedal MUST equal maxBrake.
  --   The live deploy / Python harness slams input.brake = 1.0, which inflates
  --   every wheel by 1/maxBrake and pins the high-capacity FRONTS to full lock
  --   (relative modulation is destroyed). We therefore force input.brake = maxBrake
  --   here, every time brakes are (re)applied, so the deployed torque matches
  --   training EXACTLY regardless of what the driver pedal is doing.
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

  -- RESET the abstelemetry poll window so the FIRST active tick gets a clean,
  -- braking-phase-only window — exactly like training. In abs_env the obs source
  -- (readAll) was drained every step during accel AND once more (start_data) right
  -- before the first braking step, and the car was already settled (throttle off ->
  -- neutral -> det -> step) so the first window was tiny and decel-only. Here,
  -- NOTHING drains abstelemetry's poll accumulators before engage (runTick only runs
  -- while active), so without this the first window spans the accel->brake transition
  -- and yields a huge NEGATIVE gy_avg/gy_min spike (speed still rising) that poisons
  -- the 16-frame stack.
  --
  -- resetAccum() (vs a plain buildSensorData drain) ALSO sets physPrevSpeed = -1, so
  -- the next onPhysicsStep re-seeds the speed baseline from the current (braking)
  -- speed and the first accumulated window contains only post-engage deceleration —
  -- this is what kills the first-tick spike. It also clears the brake-event state
  -- machine + lastBrake* (unused in deploy: distance is measured Python-side).
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

  -- EXT-MODE per-episode handshake reset. mlabs_tickseq is the env's warmup-done
  -- signal (it checks mlabs_active==1 && mlabs_warmup==0 && mlabs_tickseq>=1), so
  -- it MUST be 0 across the warmup ticks of every new episode. Also re-seed extCmd
  -- to the gentle warmup brake and clear stale published seq/tickseq electrics so a
  -- previous episode's values can never satisfy the env's detection early.
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
  -- buildSensorData() returns the raw Lua table (pre-JSON) that readAll() encodes
  -- and that abs_env._build_obs_from_data() was fed once per step. It also RESETS
  -- the poll-window accumulators, so this single call gives the "window since the
  -- previous setBrakes" semantics that training relied on (we call it at the TOP
  -- of the tick, before computing the new action).
  --
  -- HARD GUARD: if abstelemetry / buildSensorData is unavailable, fail loud and do
  -- NOT write brakes this tick (weightsOk-style discipline — never feed the policy
  -- a fabricated obs in closed loop).
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
  -- buildSensorData() above already drained the poll accumulators for THIS window
  -- (which spans the accel->brake transition on the very first tick). We throw that
  -- obs away, re-arm resetAccum so the NEXT window seeds cleanly from the current
  -- (now braking) speed, and let the stock full-torque pipeline keep braking this
  -- tick. After WARMUP_TICKS the first window the policy sees is decel-only, in
  -- distribution with training's post-handoff first frame.
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
  -- Ordering (spec §1): buildSensorData (above) -> buildRawObsFromData -> pushStack
  -- (above) -> publish obs + tickseq -> apply pending mailbox cmd. prev_brakes
  -- phasing is IDENTICAL to NN mode: buildRawObsFromData already consumed
  -- prevBrake* (= the previous tick's applied action) into obs[8..11]; we then
  -- overwrite cmd with the mailboxed action and latch prevBrake* = cmd for the
  -- NEXT tick. So obs(t) carries action(t-1) in both modes.
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

  -- Head applied AFTER the MLP, selected by the weights module's M.head field
  -- (PPO_V2 deploy support, 2026-07-11):
  --   "sac_tanh01" (default/legacy): a = 0.5*(tanh(mu)+1)   -> [0,1]
  --   "ppo_clip01": a = clip(mu, 0, 1)  (PPO deterministic action = Gaussian mean,
  --                 clipped to the Box(0,1) exactly as SB3 predict() does)
  -- env brake mapping (both): brake = clamp(0.01 + 0.99*a, 0.01, 1.0).
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
  print("[MTB-ML-ABS] init — SAC ML ABS controller loading")
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

  -- EXT-MODE runtime state reset. M.extMode itself is left at its module default
  -- (false) — teleport(reset=True) re-inits the controller and the env re-asserts
  -- ext mode via setExtMode() afterward (spec §4). Here we only zero the per-tick
  -- handshake counters + re-seed the mailbox to the gentle warmup brake.
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

  -- bring up the telemetry/actuation bridge (same as 1FEX) — LAST.
  local okT = pcall(function() extensions.load('abstelemetry') end)
  haveTelem = okT and (extensions and extensions.abstelemetry ~= nil) or false
  if not haveTelem then
    print("[MTB-ML-ABS] WARN: abstelemetry not available; using direct brake-write fallback")
  end
end


-- =====================================================================
-- update(dtPhys) — PHYSICS rate (~2000Hz). Self-subdivides to 200Hz.
-- =====================================================================
local function update(dtPhys)
  if dtPhys == nil or dtPhys <= 0 then return end

  -- ---- DEBUG toggle: mirror electrics.values.mlabs_debug into M.debugObs ----
  -- Lets Python flip instrumentation on/off at runtime with a single command.
  if electrics and electrics.values and electrics.values.mlabs_debug ~= nil then
    M.debugObs = (electrics.values.mlabs_debug == 1) or (electrics.values.mlabs_debug == true)
  end

  -- ---- body speed (for the engage/disengage state machine only) ----
  -- The kinematic-g / gy_min / gy_max accumulators that used to live here are GONE:
  -- those obs channels now come straight from abstelemetry.buildSensorData() (the
  -- training producer), so re-deriving them here is both redundant and was the
  -- source of the closed-loop divergence. We still need the body speed for the
  -- engage threshold + near-stop disengage.
  local speed = 0
  local vel = obj:getVelocity()
  if vel then
    local vx = vel.x or 0
    local vy = vel.y or 0
    local vz = vel.z or 0
    speed = math.sqrt(vx * vx + vy * vy + vz * vz)
  end

  -- ---- engage / disengage state machine ----
  -- Recover the TRUE driver pedal. While active we overwrite input.brake with
  -- maxBrake, so a raw read no longer reflects the driver. If input.brake still
  -- matches our last write, the driver hasn't changed it -> intent unchanged
  -- (still held). If it differs, the driver/Python moved the pedal for real.
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
    -- re-assert per-wheel brakes every physics step (survive stock overwrite),
    -- exactly like abstelemetry.onPhysicsStep does. cmd[] holds either the ML action
    -- (normal) or the gentle WARMUP_BRAKE (during the engage flush) — both must be
    -- re-asserted every physics step so the stock pipeline doesn't reclaim full torque
    -- and lock the fronts. runTick has set cmd[] on at least the first tick of either
    -- phase before this runs. In ext-mode the policy runs Python-side (no weights
    -- needed), so re-assert whenever weights are loaded OR ext-mode is active.
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
  -- Published EVERY physics step so the env can poll them without draining the
  -- poll window: warmup flag (env's warmup-done gate), ext-mode flag (re-assert
  -- check after teleport), and a NON-draining heading scalar matching exactly the
  -- abstelemetry `heading` field (obj:getDirection() or 0) the env's crash
  -- terminal compares against. These are cheap (one obj:getDirection() call) and
  -- harmless when ext-mode is off, but only ext-mode/diagnostics read them.
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
