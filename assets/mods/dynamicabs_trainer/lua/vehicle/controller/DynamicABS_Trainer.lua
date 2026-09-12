-- DynamicABS_Trainer V4.00T, 400 Hz variant for distillation experiments
-- Slip-target PID ABS on fused speed. Target comes from the vehicle's own jbeam
-- slipRatioTarget; the D-estimator scales it down and boosts gain on low grip only.
-- ---------------------------------------------------------------------------
local M = {}
M.type = "auxiliary"
M.version = "4.00"

-- PID ABS, no-cheat.

local TICK_RATE_HZ = 400
local TICK_STEP = 1 / TICK_RATE_HZ
local timeAccum = 0

local origBrakeTorque = {}

-- wheelToBrakeMap[logicalIdx] = wheelRotator index (1-based) Logical order
local wheelToBrakeMap = {1, 2, 3, 4}

-- Which logical indices are rear / front wheels
local rearLogicalIndices  = {1, 2}  -- default RR=1, RL=2
local frontLogicalIndices = {3, 4}  -- default FR=3, FL=4

local N_WHEELS = 4  -- set in init(); all tables sized to this

local wasBraking = false

-- Shared sensor state (detectMu writes at 2kHz, runTick reads at 200Hz)
local fusedSpeed = 0
local fusedPrevWs = {}
local fusedInitialized = false
local latestWheelSpeed = {}
local latestSensorY = 0
local wasBrakingMu = false
local zAccum = 0
local zCount = 0
local filtZ = 0
local wsMin1 = {9999, 9999, 9999, 9999}
local wsMin2 = {9999, 9999, 9999, 9999}

-- Snap-up gap gate (TEST)
local SNAP_GAP_MAX = 0.75
-- Snap-up low-speed cutout
local SNAP_MIN_SPEED = 1.1176  -- m/s = 2.5 mph

-- Experiment
local SEED_FUSED_FROM_AVG = false

-- IMU speed ceiling (anti-wheelspin).
local imuSpeed = 0
-- Default for IMU_TRUST_WINDOW is 2.0
local IMU_TRUST_WINDOW = 2.0  -- m/s: how far a wheel may lead imuSpeed and still be trusted
-- Slip-anchored fused stepdown: during braking the wheels lag true speed by ~the commanded slip.
local ENABLE_SLIP_STEPDOWN   = true
-- Default for SLIP_STEPDOWN_RATIO is 1.01
local SLIP_STEPDOWN_RATIO    = 1.01  -- fire when measured gap > this * expected(slip-explained) gap
-- Default for SLIP_STEPDOWN_SUSTAIN is 0.12
local SLIP_STEPDOWN_SUSTAIN  = 0.12  -- s: gap must stay over threshold this long before firing
-- Default for SLIP_STEPDOWN_COOLDOWN is 0.30
local SLIP_STEPDOWN_COOLDOWN = 0.30  -- s: no re-fire during this window after a correction
-- Default for SLIP_STEPDOWN_VMIN is 2.2352
local SLIP_STEPDOWN_VMIN     = 2.2352  -- m/s: only active above this speed (= 5 mph)
-- Default for SLIP_STEPDOWN_GAPFLOOR is 0.5
local SLIP_STEPDOWN_GAPFLOOR = 0.5  -- m/s: ignore gaps smaller than this (noise / tiny s_t)
-- Default for SLIP_STEPDOWN_STFLOOR is 0.03
local SLIP_STEPDOWN_STFLOOR  = 0.03  -- min commanded slip target used in the expected-gap calc
-- Default for SLIP_STEPDOWN_SNAPFRAC is 0.5
local SLIP_STEPDOWN_SNAPFRAC = 0.5  -- where in the [fastest wheel, slip-implied speed] window to
                                      -- land fused on a fire.
local slipStepdownTimer    = 0
local slipStepdownCooldown = 0
local slipStepdownCount    = 0
-- IMU ceiling slew

-- 2D planar speed (drift handling).
local ENABLE_2D_SPEED = true
                                -- survive a multi-second open-loop slide).
local fwdVel = 0  -- estimated forward velocity, body frame (m/s) ,  internal 2D state
local vLat = 0  -- estimated lateral (sideslip) velocity, body frame (m/s)

local vVert = 0

-- IMU input filtering (change #2).
local ENABLE_IMU_FILTERS = true  -- IMU input filters ON (per test config 2026-07-06)
-- Default for TAU_ROT is 0.02
local TAU_ROT = 0.02  -- s: EMA time constant for pitch/roll rates (2026-07-12: tested 0.01 softer -> reverted, no etk
-- Default for TAU_AZ is 0.05
local TAU_AZ  = 0.05  -- s: EMA time constant for vertical accel (2026-07-12: tested 0.025 softer -> reverted, SBR needs
-- Default for ACCEL_DEADBAND is 0.05
local ACCEL_DEADBAND = 0.05  -- m/s^2: kills stationary sensor creep on ay/az
local filtPitchRate = 0
local filtRollRate  = 0
local filtAz        = 0
-- Median spike-filter on the non-brake accel axes (2026-07-12).
local medFilt = {
  ENABLE = false,  -- 2026-07-12: median A/B'd (6+6 per car) = WASH. SBR mean/spread unchanged, etk
                   -- variance-dominated.
  WIN_AZ = 5,  -- vertical median taps (bigger = more spike removal; priority denoise axis)
  WIN_AY = 3,  -- lateral median taps (smaller = LIGHTER; slide/yaw safety, keep < WIN_AZ)
  az = { buf = {}, idx = 0, n = 0, tmp = {} },
  ay = { buf = {}, idx = 0, n = 0, tmp = {} },
}
-- Push v into a fixed-capacity ring (s.buf/s.idx/s.n, capacity win)
function medFilt.push(s, win, v)
  s.idx = (s.idx % win) + 1
  s.buf[s.idx] = v
  if s.n < win then s.n = s.n + 1 end
  local t = s.tmp
  for i = 1, s.n do t[i] = s.buf[i] end
  table.sort(t)
  local mid = math.floor(s.n / 2)
  if s.n % 2 == 1 then return t[mid + 1] else return 0.5 * (t[mid] + t[mid + 1]) end
end
local lastPitch = 0
local lastRoll = 0
local pitchRateLog = 0
local rollRateLog = 0
local pitchLog = 0
local rollLog = 0
local isLogging = false
local logData = {}
local logTimer = 0
-- imuSpeed reports GROUND SPEED = sqrt(fwdVel^2 + vLat^2)

-- Reverse support: bypass PID in reverse, handle arcade-mode input routing correctly.
local motionDirection = 1  -- +1 forward, -1 reverse (hysteresis)
-- Default for REVERSE_DETECT_THRESHOLD is 1.0
local REVERSE_DETECT_THRESHOLD = 1.0  -- m/s ,  min wheel speed to trust direction signal
-- Default for REVERSE_LOCKIN_MPS is 0.5
local REVERSE_LOCKIN_MPS       = 0.5  -- m/s ,  signed avg must exceed this to flip

-- Wheel decel lockup guard
local WHEEL_DECEL_LIMIT = -100  -- m/s²

-- [SLIP GOVERNOR 2026-07-15] Common-mode slip governor.
local slipGov = {
  ENABLE = true,
  GM_MIN = 0.90, GM_MAX = 1.10,  -- authority trim bounds (multiplies absCoef, clamped 0..1)
  RATE = 0.40,  -- trim slew per second
  DB = 0.02,  -- deadband around lambda* (slip units)
  SPEED_GATE = 8.0,  -- m/s; below this trim relaxes to neutral
  SEED = 0.10,  -- lambda* until the fit is trusted
  LAM_MIN = 0.06, LAM_MAX = 0.30,  -- lambda* clamp
  LAM_SLEW = 0.05,  -- lambda* max change per second
  FIT_MIN_N = 400,  -- RLS updates before the fit drives lambda*
  GUARD_DECEL = -35.0,  -- m/s^2 pack rel. decel (10ms-smoothed) -> guard
  GUARD_HOLD = 0.06,  -- s of authority dip per guard event
  GUARD_GM = 0.94,  -- authority ceiling while guard holds
  -- state (module-local => persists across stops within a session: warm start)
  gm = 1.0, lambdaStar = 0.10, fitN = 0, events = 0, guardHoldLeft = 0,
  sPack = 0, wsAvgF = 0, fusedF = 0, wsAvgPrev = nil, fusedPrev = nil,
  c2 = {12.0, 20.0, 32.0, 48.0}, th = nil, P = nil, ew = nil, decim = 0, F0 = 0,
}

-- Brake-event recorder: FIFO of last 4 events with peak fused-vs-airspeed divergence.
local brakeEvents = {}
local currentBrakeEvent = nil
local brakeSimTime = 0
local brakeUiAccum = 0
-- Default for BRAKE_EVENT_MIN_AIR is 1.0
local BRAKE_EVENT_MIN_AIR = 1.0  -- m/s: don't even start an event below this
-- Default for BRAKE_EVENT_MIN_PEAK_AIR is 2.2352
local BRAKE_EVENT_MIN_PEAK_AIR = 2.2352  -- m/s (= 5 mph): don't update peak divergence below this
-- Default for BRAKE_EVENT_MIN_DUR is 0.3
local BRAKE_EVENT_MIN_DUR = 0.3  -- s
-- Default for BRAKE_EVENT_PRESS_THRESHOLD is 0.05
local BRAKE_EVENT_PRESS_THRESHOLD = 0.05
-- Default for BRAKE_EVENT_MAX is 6
local BRAKE_EVENT_MAX = 6  -- FIFO size
-- Default for BRAKE_EVENT_FILE is "settings/blake_abs_brake_events.json"
local BRAKE_EVENT_FILE = "settings/blake_abs_brake_events.json"

-- wheelAvg speed estimator.
local wa = {
  speed = 0,
  updated = false,
  speedTwo = 0,
  twoUpdated = false,
  prevWs = {},
  locked = {},
  decelBuf = {},
  decelIdx = {},
  -- Default for LOCK_WINDOW is 8
  LOCK_WINDOW = 8,  -- 40ms at 200Hz
  -- Default for LOCK_DECEL is -5
  LOCK_DECEL = -5,  -- hard-lockup threshold (m/s²)
}

-- UI state: which speed source PID is using this tick
local absSpeedSource = { wheelAvgActive = false, fusedActive = true }

-- Safety + counters: bundled to stay under LuaJIT's 60-upvalue limit
local safety = {
  snapUp = 0, snapUpRej = 0,  -- snapUpRej = up-snaps rejected by the gap gate
  imuClamps = 0,  -- times the IMU ceiling capped fused (anti-wheelspin)
  slipRatios = {},
  lastAbsCoefs = {},
  -- Low-speed brake boost
  lowSpeed = { ENABLE = false, BOOST_MAX = 1.25, SPEED_THRESH = 13.41 },
  -- Low-mu ABS-off extension: on slippery surfaces keep ABS active DEEPER into the stop.
  lowMu = { ENABLE = true, MU_THRESH = 0.50, SUSTAIN = 0.2, SPEED_GATE = 4.4704, ABS_OFF_SPEED = 1.118 },
  lowMuTimer = 0,
  -- Recovery-peak vRef re-anchor (production trick, Bosch/WABCO
  recAnchor = { ENABLE = true, CMD_MAX = 0.35, RISE_MIN = 0.20, DROP = 0.08, CORR_MAX = 0.75, COOLDOWN = 0.10, GATE = 2.0 },
  recSt   = {0, 0, 0, 0},  -- 0=idle 1=watching 2=armed
  recWsF  = {0, 0, 0, 0},  -- light-EMA wheel speed for the detector
  recBase = {0, 0, 0, 0},
  recPeak = {0, 0, 0, 0},
  recCool = {0, 0, 0, 0},
  recCount = 0,
  -- Derivative low-pass (see the big note at the KD tunable).
  DERIV_FILTER = true,
  -- Default for DFILT is 0.6
  DFILT = 0.6,
  -- Low-speed fused re-anchor: below REANCHOR_SPEED, set fused = weighted avg of imuSpeed (the accelerometer integral
  LOWSPEED_REANCHOR = true,  -- kept ON (safety): reanchor-OFF let the accel integral over-read ~+2mph
                             -- near stop (@8mph +2.0).
  REANCHOR_SPEED = 5.0,  -- m/s = 11.2 mph (2026-07-12: lowered from 11.176/25mph). Dual role
                             -- (1) reanchor only fires <11mph where the accel integral genuinely over-reads near stop (@8mph)
  REANCHOR_BLEND = 0.5,
}

local lastLowSpeedBoost = 1.0

-- Per-wheel surface-grip detector (ANCHORED brake-acceptance).
local grip = {
  -- Default for ENABLE_PERWHEEL_D is false
  ENABLE_PERWHEEL_D = false,  -- master switch; false => original global broadcast (escape hatch)
  -- Default for FRONT_FRAC is 0.5 (fallback; computed in init())
  FRONT_FRAC = 0.5,  -- static front weight fraction ,  computed in init(), fallback 0.5
  -- Default for H_CG is 0.55 (fallback; read from jbeamData in init())
  H_CG = 0.55,  -- CG height (m) ,  read from jbeamData or universal fallback
  -- Default for WHEELBASE is 2.6 (fallback; computed in init())
  WHEELBASE = 2.6,  -- wheelbase (m) ,  computed from axle node positions in init()
  -- Default for GRAV is 9.81
  GRAV = 9.81,
  -- Default for FZ_MIN=200, SAT=0.97, ROLL_MIN=1.0, TORQUE_MIN_FRAC=0.05
  FZ_MIN = 200, SAT = 0.97, ROLL_MIN = 1.0, TORQUE_MIN_FRAC = 0.05,
  -- Default for G_SMOOTH=0.9, DECAY=0.05, EPS=1e-3
  G_SMOOTH = 0.9, DECAY = 0.05, EPS = 1e-3,
  -- Default for mass is 1500 (fallback; obj:getMass() in init())
  mass = 1500,
  Fz0 = {},  -- per-wheel static load (set in init)
  gEMA = {},
  Dwheel = {},
  confident = {},
  yawRate = 0,  -- last read yaw rate (rad/s), published for sign-check
}

-- Per-wheel PID state
local slipIntegral = {}
local lastSlipError = {}
local prevTickWheelSpeed = {}
local lastEffectiveTargets = {0, 0, 0, 0}
local lastSlipErrors = {0, 0, 0, 0}
local lastSlipDerivatives = {0, 0, 0, 0}

-- ABS dashboard telltale state.
local absLightPulse = 0
local absLightTimer = 0
-- Default for ABS_LIGHT_PULSE_PERIOD is 0.15
local ABS_LIGHT_PULSE_PERIOD = 0.15  -- s, matches stock warningLightsDelayTime



-- Per-wheel adaptive slip targets (D-estimator writes, PID reads)
local slipTargets = {}
-- Default for SLIP_TARGET_MIN is 0.02
local SLIP_TARGET_MIN = 0.02
-- Default for SLIP_TARGET_MAX is 1.0
local SLIP_TARGET_MAX = 1.0
local TARGET_SMOOTHING = 0.95  -- Default is 0.95

-- [REMOVED 2026-07-12] USE_TIRE_SLIP_TARGET / tireSlipTarget

-- Low-speed slip deepening: the (1+D)/carSpeed term added to effectiveTarget
local ENABLE_LOWSPEED_SLIP_DEEPEN = true
-- [JBEAM SLIP TARGET 2026-07-13] Use BeamNG's OWN per-wheel ABS slip target (read from the brakes
local USE_JBEAM_SLIP_TARGET = true
local absTargetBase = {0.18, 0.18, 0.18, 0.18}  -- per LOGICAL wheel (1=RR 2=RL 3=FR 4=FL); set at init

-- D estimator (peak-decel window).
local dest = {
  -- Default for EST_MIN is 0.10
  EST_MIN          = 0.10,
  -- Default for EST_MAX is 2.25
  EST_MAX          = 2.25,
  consensusD       = 1.0,
  SMOOTHING        = 0.80,  -- Default is 0.80
  UPDATE_MIN_DECEL = 0.5,  -- Default is 0.5
  SLIP_GATE        = 0.03,  -- Default is 0.03
  -- Default for baseTarget is 0.14
  baseTarget       = 0.14,
  retroResets      = 0,
  WINDOW_SIZE      = 40,  -- Default is 40 (0.2s at 200Hz)
  window           = {},
  windowIdx        = 0,
  stableD          = 1.0,
  stableTicks      = 0,
  -- Default for SETTLE_TICKS is 30
  SETTLE_TICKS     = 30,  -- 0.15s before surface-change detection arms
  -- Default for CHANGE_THRESHOLD is 0.40
  CHANGE_THRESHOLD = 0.40,  -- >40% D jump = surface change, blow away the window

  -- Effort gate: OR'd with the slip gate.
  EFFORT_GATE      = false,
  -- D-freeze
  FREEZE_SPEED     = 0,

  absEventActive   = false,
  absEventSum      = 0,
  absEventTicks    = 0,
  absEventTotalTicks = 0,

  -- Per-braking-event surface-change counter.
  surfaceChangeCount = 0,
}

-- D-estimator window aggregation: how the sliding decel window collapses to one number.
local D_AGG_MODE = "peak"
-- Default for D_AGG_TOPN is 10
local D_AGG_TOPN = 10

-- Collapse dest.window to a single decel value per D_AGG_MODE.
local function aggregateDecelWindow()
  local w = dest.window
  if D_AGG_MODE == "mean" then
    local sum, n = 0, 0
    for j = 1, dest.WINDOW_SIZE do local v = w[j]; if v then sum = sum + v; n = n + 1 end end
    return n > 0 and (sum / n) or 0
  elseif D_AGG_MODE == "topn" then
    local vals = {}
    for j = 1, dest.WINDOW_SIZE do local v = w[j]; if v then vals[#vals + 1] = v end end
    local n = #vals
    if n == 0 then return 0 end
    table.sort(vals)  -- ascending
    local k = math.min(D_AGG_TOPN, n)
    local s = 0
    for i = n, n - k + 1, -1 do s = s + vals[i] end
    return s / k
  else  -- "peak" (default / historical)
    local peak = 0
    for j = 1, dest.WINDOW_SIZE do local v = w[j]; if v and v > peak then peak = v end end
    return peak
  end
end

-- Hybrid Peak-Hunter & Brake Simulator State
local ph = {
  -- Default for ENABLE is false
  ENABLE = false,  -- DISABLED 2026-07-12. Was true-but-STUCK (jammed in SETTLE after one +0.05 step ->
                   -- why on/off felt identical).
  ENABLE_THROTTLE_LOCKOUT = true,
  seekerSuppressed = false,
  
  turn = 1,  -- 1 = Front, 2 = Rear
  phase = 0,  -- 0=STEP, 1=SETTLE, 2=MEASURE, 3=EVALUATE
  
  frontOffset = 0,
  rearOffset = 0,
  frontDirection = 1,
  rearDirection = 1,
  
  lastFrontEfficiency = 0,
  lastRearEfficiency = 0,
  
  settleTicks = 0,
  measureTicks = 0,
  startSpeed = 0,
}

-- PID tunables Default for ENABLE_DYNAMIC_PID is true
local ENABLE_DYNAMIC_PID = true
-- Default for KP is 6.0
local KP = 6.0
-- Default for KP_MAX is 50.0
local KP_MAX = 50.0
local current_KP = 6.0
-- Default for KI is 0.8
local KI = 0.8
-- Default for KD is 0.08
local KD = 0.08
-- Derivative low-pass: EMA-smooth the slip-error derivative before KD uses it.
local INTEGRAL_MIN = -1.0
-- Default for INTEGRAL_MAX is 1.0
local INTEGRAL_MAX = 1.0
-- Default for MIN_SPEED is 3.0
local MIN_SPEED = 3.0  -- ~6.7 mph
-- Default for MIN_ADAPT_SPEED is 3.2
local MIN_ADAPT_SPEED = 3.2  -- ~7.1 mph

-- Misc Default for NON_BRAKING_DECEL_FILTER is -5.0
local NON_BRAKING_DECEL_FILTER = -5.0
-- Default for STANDSTILL_WS_THRESHOLD is 0.3
local STANDSTILL_WS_THRESHOLD = 0.3
-- Default for STANDSTILL_FUSED_THRESHOLD is 2.0
local STANDSTILL_FUSED_THRESHOLD = 2.0
-- Default for STANDSTILL_ACCEL_MAX is 0.5
local STANDSTILL_ACCEL_MAX = 0.5  -- m/s^2: only count as stopped when NOT decelerating. Locked wheels
                                   -- read ~0 while the car still moves under braking
local STANDSTILL_PHYS_TICKS = 100
local standstillCounter = 0

local uiAccum = 0
local function getCondition(slip)
  if slip <= 0.06 then return "Ice"
  elseif slip <= 0.10 then return "Snow"
  elseif slip <= 0.13 then return "Wet"
  else return "Dry Asphalt"
  end
end

-- Persist brake-event history to disk so it survives vehicle switches / config changes.
local function saveBrakeEvents()
  pcall(jsonWriteFile, BRAKE_EVENT_FILE, brakeEvents, false)
end

local function loadBrakeEvents()
  local ok, data = pcall(jsonReadFile, BRAKE_EVENT_FILE)
  if ok and type(data) == "table" then
    brakeEvents = {}
    for i, e in ipairs(data) do
      if i > BRAKE_EVENT_MAX then break end
      table.insert(brakeEvents, e)
    end
  end
end


local function initDecelWindow()
  dest.window = {}
  dest.windowIdx = 0
  dest.stableD = 1.0
  dest.stableTicks = 0
  dest.consensusD = 1.0
end


-- (2) buildWheelMaps: 1:1 copy of the stock BeamNG method from drivingDynamics/sensors/vehicleData.lua  ->  initSecondStage().
local function buildWheelMaps()
  wheelToBrakeMap = {}

  local ok, err = pcall(function()
    -- Stock: jbeamData.cornerWheels or {"FR", "FL", "RR", "RL"}
    local cornerWheelData = {"FR", "FL", "RR", "RL"}
    local cornerWheels = {}
    for _, wheelName in pairs(cornerWheelData) do
      cornerWheels[wheelName] = true
    end

    -- Stock
    local avgWheelPos = vec3(0, 0, 0)
    for _, wheel in pairs(wheels.wheels) do
      if cornerWheels[wheel.name] then
        local wheelNodePos = v.data.nodes[wheel.node1].pos
        avgWheelPos = avgWheelPos + wheelNodePos
      end
    end
    avgWheelPos = avgWheelPos / #wheels.wheels  -- stock divides by total wheel count

    -- Stock: build reference frame from the vehicle's ref nodes
    local refNodes = v.data.refNodes[0]
    local vectorForward = vec3(v.data.nodes[refNodes.ref].pos) - vec3(v.data.nodes[refNodes.back].pos)
    local vectorUp      = vec3(v.data.nodes[refNodes.up].pos)  - vec3(v.data.nodes[refNodes.ref].pos)
    local vectorRight   = vectorForward:cross(vectorUp)

    local foundWheelsCount = 0

    -- Stock: classify each corner wheel using dot products
    for _, wheel in pairs(wheels.wheels) do
      if cornerWheels[wheel.name] then
        local wheelNodePos = vec3(v.data.nodes[wheel.node1].pos)
        local wheelVector  = wheelNodePos - avgWheelPos
        local dotForward   = vectorForward:dot(wheelVector)
        local dotRight     = vectorRight:dot(wheelVector)  -- stock calls this "dotLeft" but tests >= 0 for right

        -- Map wheel name  ->  wheelRotator index  ->  1-based slot for our tables
        local rotIdx = wheels.wheelRotatorIDs[wheel.name]
        if rotIdx == nil then error("wheelRotatorIDs missing for '" .. wheel.name .. "'") end
        local slot = rotIdx + 1  -- wheelRotatorIDs is 0-based; our tables are 1-based

        -- Stock convention: dotRight >= 0 is right side, dotForward >= 0 is front
        if dotRight >= 0 then
          if dotForward >= 0 then
            wheelToBrakeMap[3] = slot  -- FR = logical 3
          else
            wheelToBrakeMap[1] = slot  -- RR = logical 1
          end
        else
          if dotForward >= 0 then
            wheelToBrakeMap[4] = slot  -- FL = logical 4
          else
            wheelToBrakeMap[2] = slot  -- RL = logical 2
          end
        end
        foundWheelsCount = foundWheelsCount + 1
      end
    end

    if foundWheelsCount ~= 4 or not (wheelToBrakeMap[1] and wheelToBrakeMap[2] and wheelToBrakeMap[3] and wheelToBrakeMap[4]) then
      error("Could not classify all 4 corner wheels (found " .. foundWheelsCount .. ")")
    end

    print("[ABS-1FEX] (2) wheelToBrakeMap built (stock geometry method): RR="
      .. tostring(wheelToBrakeMap[1]) .. " RL=" .. tostring(wheelToBrakeMap[2])
      .. " FR=" .. tostring(wheelToBrakeMap[3]) .. " FL=" .. tostring(wheelToBrakeMap[4]))
  end)

  if not ok then
    -- Fallback: identity map
    for i = 1, N_WHEELS do wheelToBrakeMap[i] = i end
    print("[ABS-1FEX] (2) WARNING: Stock geometry method failed (" .. tostring(err) .. "). Using fallback identity map.")
  end

  rearLogicalIndices  = {1, 2}
  frontLogicalIndices = {3, 4}
end


-- (2) buildGeometry: computes WHEELBASE and FRONT_FRAC from wheel axle node positions.
local function buildGeometry(jbeamData)
  -- CG height: try jbeamData first (our own ABS jbeam may define it), then universal fallback.
  local cgH = 0.55
  if jbeamData then
    pcall(function()
      if jbeamData.cgHeight and type(jbeamData.cgHeight) == "number" then
        cgH = jbeamData.cgHeight
      end
    end)
  end
  grip.H_CG = cgH

  -- Attempt to read axle node positions for each named wheel.
  local wheelNames = { "RR", "RL", "FR", "FL" }
  local logicalMap = { RR = 1, RL = 2, FR = 3, FL = 4 }
  local positions = {}  -- [logicalIdx] = {x, y, z} body-frame midpoint

  for _, name in ipairs(wheelNames) do
    local logical = logicalMap[name]
    if wheels.wheelRotatorIDs and wheels.wheelRotatorIDs[name] ~= nil then
      local physIdx = wheels.wheelRotatorIDs[name]
      local wr = wheels.wheelRotators[physIdx]
      if wr and wr.node1 ~= nil and wr.node2 ~= nil then
        local pos = nil
        pcall(function()
          local p1 = obj:getNodePositionRelative(wr.node1)
          local p2 = obj:getNodePositionRelative(wr.node2)
          -- Average of the two axle nodes = wheel center in body frame
          pos = {
            x = (p1.x + p2.x) * 0.5,
            y = (p1.y + p2.y) * 0.5,
            z = (p1.z + p2.z) * 0.5,
          }
        end)
        if pos then positions[logical] = pos end
      end
    end
  end

  -- Wheelbase: distance along Y between front axle avg and rear axle avg.
  local frontY, rearY = nil, nil
  for _, li in ipairs(frontLogicalIndices) do
    if positions[li] then
      frontY = (frontY or 0) + positions[li].y
    end
  end
  for _, li in ipairs(rearLogicalIndices) do
    if positions[li] then
      rearY = (rearY or 0) + positions[li].y
    end
  end
  if frontY and rearY then
    frontY = frontY / #frontLogicalIndices
    rearY  = rearY  / #rearLogicalIndices
    local wb = math.abs(rearY - frontY)
    if wb > 0.5 then  -- sanity: anything under 0.5m is probably a bad read
      grip.WHEELBASE = wb
    end
  end

  -- FRONT_FRAC: fraction of static weight on front axle.
  if frontY and rearY and grip.WHEELBASE > 0.5 then
    -- CG is at Y=0 in body frame (origin)
    local frac = rearY / grip.WHEELBASE
    if frac > 0.2 and frac < 0.8 then  -- sanity bounds
      grip.FRONT_FRAC = frac
    end
  end

  print(string.format("[ABS-1FEX] (2) Geometry: WB=%.2fm FRONT_FRAC=%.2f H_CG=%.2fm",
    grip.WHEELBASE, grip.FRONT_FRAC, grip.H_CG))
end


local function init(jbeamData)
  print("[ABS-1FEX] (2) canonical build loaded")

  -- Read wheel count first; everything else is sized to this.
  N_WHEELS = wheels.wheelRotatorCount or 4

  -- Initialize all N_WHEELS-sized tables
  origBrakeTorque       = {}
  slipIntegral          = {}
  lastSlipError         = {}
  prevTickWheelSpeed    = {}
  latestWheelSpeed      = {}
  fusedPrevWs           = {}
  slipTargets           = {}
  ph.simulatedTorque    = {}
  ph.brakeInRate        = {}
  ph.brakeOutRate       = {}
  safety.slipRatios     = {}
  safety.lastAbsCoefs   = {}
  grip.Fz0              = {}
  grip.gEMA             = {}
  grip.Dwheel           = {}
  grip.confident        = {}
  wa.prevWs             = {}
  wa.locked             = {}
  wa.decelBuf           = {}
  wa.decelIdx           = {}

  for i = 1, N_WHEELS do
    slipIntegral[i]       = 0
    lastSlipError[i]      = 0
    lastSlipErrors[i]     = 0
    lastSlipDerivatives[i]= 0
    lastEffectiveTargets[i]= 0.14
    prevTickWheelSpeed[i] = 0
    latestWheelSpeed[i]   = 0
    fusedPrevWs[i]        = 0
    slipTargets[i]        = 0.14
    ph.simulatedTorque[i] = 0
    ph.brakeInRate[i]     = 0
    ph.brakeOutRate[i]    = 0
    safety.slipRatios[i]  = 0
    safety.lastAbsCoefs[i]= 1
    grip.Fz0[i]           = 0
    grip.gEMA[i]          = 1
    grip.Dwheel[i]        = 1
    grip.confident[i]     = false
    wa.prevWs[i]          = 0
    wa.locked[i]          = false
    wa.decelBuf[i]        = {}
    wa.decelIdx[i]        = 0
    wheelToBrakeMap[i]    = i  -- safe default until buildWheelMaps overrides
  end

  wasBraking = false
  wasBrakingMu = false
  timeAccum = 0
  uiAccum = 0
  standstillCounter = 0
  safety.snapUp = 0
  safety.snapUpRej = 0
  safety.imuClamps = 0
  slipStepdownTimer = 0
  slipStepdownCooldown = 0
  slipStepdownCount = 0
  -- Reset mid-event state; load persisted history from disk (survives vehicle switches).
  currentBrakeEvent = nil
  brakeSimTime = 0
  brakeUiAccum = 0
  loadBrakeEvents()

  initDecelWindow()
  dest.retroResets = 0
  dest.surfaceChangeCount = 0
  dest.baseTarget = 0.14
  dest.consensusD = 1.0
  dest.window = {}
  dest.windowIdx = 0
  dest.stableD = 1.0
  dest.stableTicks = 0
  dest.absEventActive = false
  dest.absEventSum = 0
  dest.absEventTicks = 0
  dest.absEventTotalTicks = 0

  ph.frontOffset = 0
  ph.rearOffset = 0
  ph.frontDirection = 1
  ph.rearDirection = 1
  ph.phase = 0
  ph.turn = 1

  ph.seekerSuppressed = false

  fusedSpeed = 0
  imuSpeed = 0
  fwdVel = 0
  vLat = 0
  vVert = 0
  lastPitch = 0
  lastRoll = 0
  pitchRateLog = 0
  rollRateLog = 0
  pitchLog = 0
  rollLog = 0
  
  -- Logging state
  isLogging = false
  logData = {}
  logTimer = 0
  fusedInitialized = false
  latestSensorY = 0
  zAccum = 0
  zCount = 0
  filtZ = 0
  for i=1, N_WHEELS do
    wsMin1[i] = 9999
    wsMin2[i] = 9999
  end

  -- Read static brake torques and hydraulic delays.
  local rawBrakeTorque = {}
  local rawInDelay = {}
  local rawOutDelay = {}
  for i = 0, N_WHEELS - 1 do
    rawBrakeTorque[i + 1] = wheels.wheelRotators[i].brakeTorque or 0
    rawInDelay[i + 1] = wheels.wheelRotators[i].brakePressureInDelay or 0.04
    rawOutDelay[i + 1] = wheels.wheelRotators[i].brakePressureOutDelay or 0.04
  end

  -- (2) Build wheel name -> index map (replaces hardcoded wheelToBrakeMap = {3,4,1,2})
  buildWheelMaps()

  -- Second pass: remap brake torques from raw wheelRotator order to logical order
  for i = 1, N_WHEELS do
    local slot = wheelToBrakeMap[i]  -- physical wheelRotator slot (1-based)
    local maxT = slot and rawBrakeTorque[slot] or rawBrakeTorque[i]
    origBrakeTorque[i] = maxT
    local inD = slot and rawInDelay[slot] or rawInDelay[i]
    local outD = slot and rawOutDelay[slot] or rawOutDelay[i]
    ph.brakeInRate[i] = maxT / (inD + 1e-30)
    ph.brakeOutRate[i] = maxT / (outD + 1e-30)
    -- [JBEAM SLIP TARGET] per-wheel ABS target; front/rear split lives in the brakes jbeam
    local wr = slot and wheels.wheelRotators[slot - 1]
    local ww = (wheels.wheels and slot) and wheels.wheels[slot - 1]
    absTargetBase[i] = (wr and (wr.slipRatioTarget or wr.absSlipRatioTarget))
                    or (ww and (ww.slipRatioTarget or ww.absSlipRatioTarget))
                    or 0.18
  end
  print(string.format("[ABS jbeam slip targets] RR=%.3f RL=%.3f FR=%.3f FL=%.3f",
    absTargetBase[1], absTargetBase[2], absTargetBase[3], absTargetBase[4]))
  -- [FORCE SEEKER] cache rotator tables and front flags in safety, avoiding new upvalues
  safety.fsWr = {}
  safety.fsIsFront = {false, false, false, false}
  for i2 = 1, N_WHEELS do
    local slot2 = wheelToBrakeMap[i2]
    safety.fsWr[i2] = slot2 and wheels.wheelRotators[slot2 - 1] or nil
  end
  for _, li in ipairs(frontLogicalIndices) do safety.fsIsFront[li] = true end
  safety.fsState = nil  -- created at brake onset in runTick

  -- (2) Vehicle mass ,  static property, read once at init. obj:getMass() is confirmed API.
  pcall(function() grip.mass = obj:getMass() or grip.mass end)

  -- Disable per-wheel D on vehicles with fewer than 4 wheels
  grip.ENABLE_PERWHEEL_D = (N_WHEELS >= 4) and grip.ENABLE_PERWHEEL_D or false

  -- (2) Compute geometry from wheel axle node positions (wheelbase, FRONT_FRAC) and read H_CG from jbeamData
  buildGeometry(jbeamData)

  -- Static corner loads from mass + computed front/rear fraction
  do
    local fAxle = grip.mass * grip.GRAV * grip.FRONT_FRAC / 2  -- per front wheel
    local rAxle = grip.mass * grip.GRAV * (1 - grip.FRONT_FRAC) / 2  -- per rear wheel
    for _, li in ipairs(rearLogicalIndices)  do grip.Fz0[li] = rAxle end
    for _, li in ipairs(frontLogicalIndices) do grip.Fz0[li] = fAxle end
    -- Any logical index not in either list (non-standard layout) gets average
    local avgFz = grip.mass * grip.GRAV / N_WHEELS
    for i = 1, N_WHEELS do
      if grip.Fz0[i] == 0 then grip.Fz0[i] = avgFz end
    end
  end
  -- [CROSS-AXLE CURVE SAMPLING] static axle loads for force normalization (mu = F/Fz)
  safety.fsFzF, safety.fsFzR = 0, 0
  for i = 1, N_WHEELS do
    if safety.fsIsFront and safety.fsIsFront[i] then safety.fsFzF = safety.fsFzF + grip.Fz0[i]
    else safety.fsFzR = safety.fsFzR + grip.Fz0[i] end
  end

  extensions.load('abstelemv2')
  extensions.load('absTelemetryLogger')
end


-- detectMu(dtPhys) ,  2kHz sensor fusion only
local function detectMu(dtPhys)
  local rawSY = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorY) or 0
  -- BeamNG's ffiSensors.sensorY is already body-frame, gravity-cancelled.
  latestSensorY = rawSY
  local rawSZ = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorZ) or 0
  zAccum = zAccum + rawSZ
  zCount = zCount + 1

  local signedSum, signedCount = 0, 0
  -- Read wheel speeds indexed by LOGICAL order (1=RR,2=RL,3=FR,4=FL).
  for i = 1, N_WHEELS do
    local slot = wheelToBrakeMap[i]
    local signedWs = wheels.wheelRotators[slot - 1].wheelSpeed or 0  -- wheelRotators is 0-based
    latestWheelSpeed[i] = math.abs(signedWs)
    if latestWheelSpeed[i] < wsMin1[i] then
      wsMin2[i] = wsMin1[i]
      wsMin1[i] = latestWheelSpeed[i]
    elseif latestWheelSpeed[i] < wsMin2[i] then
      wsMin2[i] = latestWheelSpeed[i]
    end
    if math.abs(signedWs) > REVERSE_DETECT_THRESHOLD then
      signedSum = signedSum + signedWs
      signedCount = signedCount + 1
    end
  end

  -- Motion-direction detection with hysteresis: flip only when signed avg clearly crosses threshold.
  if signedCount > 0 then
    local signedAvg = signedSum / signedCount
    if signedAvg > REVERSE_LOCKIN_MPS then motionDirection = 1
    elseif signedAvg < -REVERSE_LOCKIN_MPS then motionDirection = -1 end
  end

  if not fusedInitialized then
    local maxInit = 0
    for i = 1, N_WHEELS do
      if latestWheelSpeed[i] > maxInit then maxInit = latestWheelSpeed[i] end
    end
    fusedSpeed = maxInit
    imuSpeed = maxInit
    fwdVel = maxInit
    for i = 1, N_WHEELS do fusedPrevWs[i] = latestWheelSpeed[i] end
    fusedInitialized = true
  end

  local brakeInput = input.brake or 0
  local isBraking = brakeInput > 0

  -- Integrate raw decel, floor at 0.
  fusedSpeed = math.max(0, fusedSpeed - latestSensorY * dtPhys)

  local maxWs, minWs, maxIdx = 0, math.huge, 1
  local wsSum = 0
  for i = 1, N_WHEELS do
    local ws = latestWheelSpeed[i]
    wsSum = wsSum + ws
    if ws > maxWs then maxWs = ws; maxIdx = i end
    if ws < minWs then minWs = ws end
  end
  local avgWs = wsSum / N_WHEELS

  if isBraking then
    if SEED_FUSED_FROM_AVG and (not wasBrakingMu) then
      -- First braking frame: seed the estimate from the 4-wheel AVERAGE, not the fastest wheel.
      fusedSpeed = avgWs
      fwdVel = avgWs
      vLat = 0
      vVert = 0
    else
      -- Snap-up gap gate (TEST)
      local bothStopped = fusedSpeed < SNAP_MIN_SPEED and avgWs < SNAP_MIN_SPEED
      if maxWs > fusedSpeed and not bothStopped and fusedSpeed < safety.REANCHOR_SPEED then
        if (maxWs - fusedSpeed) <= SNAP_GAP_MAX then
          fusedSpeed = maxWs
          safety.snapUp = safety.snapUp + 1
        else
          safety.snapUpRej = safety.snapUpRej + 1
        end
      end
    end
  else
    local goodSum, goodCount = 0, 0
    for i = 1, N_WHEELS do
      local d = (latestWheelSpeed[i] - fusedPrevWs[i]) / dtPhys
      if d > NON_BRAKING_DECEL_FILTER then
        goodSum = goodSum + latestWheelSpeed[i]
        goodCount = goodCount + 1
      end
    end
    if goodCount > 0 then fusedSpeed = goodSum / goodCount end
  end

  -- Speed integral.
  if ENABLE_2D_SPEED then
    local yr = 0
    pcall(function() yr = obj:getYawAngularVelocity() or 0 end)
    local roll, pitch = 0, 0
    pcall(function() roll, pitch, _ = obj:getRollPitchYaw() end)

    local pitchRate, rollRate = 0, 0
    if dtPhys > 0 then
      local dPitch = (pitch - (lastPitch or 0)) % (2 * math.pi)
      if dPitch > math.pi then dPitch = dPitch - 2 * math.pi end
      pitchRate = dPitch / dtPhys

      local dRoll = (roll - (lastRoll or 0)) % (2 * math.pi)
      if dRoll > math.pi then dRoll = dRoll - 2 * math.pi end
      rollRate = dRoll / dtPhys
    end
    lastPitch = pitch
    lastRoll = roll
    pitchLog = pitch
    rollLog = roll
    pitchRateLog = pitchRate
    rollRateLog = rollRate

    local ax = -latestSensorY  -- forward accel
    local ay = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorX) or 0  -- lateral accel
    local az = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorZ) or 0  -- vertical accel

    -- Median spike-kill on az (vertical) + ay (lateral) BEFORE the EMA.
    if medFilt.ENABLE then
      az = medFilt.push(medFilt.az, medFilt.WIN_AZ, az)
      ay = medFilt.push(medFilt.ay, medFilt.WIN_AY, ay)
    end

    -- (change #2) Filter the noisy signals before integration.
    if ENABLE_IMU_FILTERS then
      local aRot = dtPhys / (TAU_ROT + dtPhys)
      filtPitchRate = filtPitchRate + (pitchRate - filtPitchRate) * aRot
      filtRollRate  = filtRollRate  + (rollRate  - filtRollRate)  * aRot
      pitchRate = filtPitchRate
      rollRate  = filtRollRate

      local aAz = dtPhys / (TAU_AZ + dtPhys)
      filtAz = filtAz + (az - filtAz) * aAz
      az = filtAz
      if math.abs(az) < ACCEL_DEADBAND then az = 0 end
      if math.abs(ay) < ACCEL_DEADBAND then ay = 0 end
    end

    -- Full 3D strapdown integration (Coriolis/Centripetal cross-coupling)
    local dotFwd  = ax + vLat * yr + (vVert or 0) * pitchRate
    local dotLat  = -ay - fwdVel * yr + (vVert or 0) * rollRate
    local dotVert = -az - fwdVel * pitchRate - vLat * rollRate

    local newFwd = fwdVel + dotFwd * dtPhys
    vLat         = vLat   + dotLat * dtPhys
    vVert        = (vVert or 0) + dotVert * dtPhys
    fwdVel = newFwd  -- forward comp may pass through/below 0

    -- Kinematic Lateral Anchor: Tie vLat decay to steering angle and yaw rate
    local steering = math.abs(electrics.values.steering or 0)
    local absYaw = math.abs(yr)
    if absYaw < 0.05 and steering < 0.05 then
      -- Driving perfectly straight: scrub phantom lateral noise aggressively
      vLat = vLat * 0.95
    else
      -- Steering or drifting: use a dynamic decay that scales with yaw/steering severity.
      local adaptiveDecay = math.max(0.99, 1.0 - (0.01 / (1.0 + absYaw * 10.0 + steering * 5.0)))
      vLat = vLat * adaptiveDecay
    end
    
    vVert = vVert * 0.98  -- always gently bleed vertical velocity to prevent drift
    imuSpeed = math.sqrt(fwdVel * fwdVel + vLat * vLat + vVert * vVert)  -- true 3D ground speed (magnitude)
  else
    vLat = 0
    vVert = 0
    fwdVel = math.max(0, fwdVel - latestSensorY * dtPhys)
    imuSpeed = fwdVel
  end

  -- IMU speed ceiling: cap fused at the accelerometer-derived ground speed.
  if (not isBraking) and (maxWs - minWs) < 1.0 and maxWs <= imuSpeed + IMU_TRUST_WINDOW then
    fwdVel = maxWs
    vLat = 0  -- wheels agree => assume no sideslip, reset lateral est.
    vVert = 0  -- reset vertical est.
    imuSpeed = maxWs
  end

  -- IMU ceiling constraint
  local brakingJustStartedMu = isBraking and not wasBrakingMu
  if not isBraking then
    if fusedSpeed > imuSpeed then
      fusedSpeed = imuSpeed
      safety.imuClamps = safety.imuClamps + 1
    end
  elseif brakingJustStartedMu then
    if fusedSpeed > imuSpeed then
      fusedSpeed = imuSpeed
      safety.imuClamps = safety.imuClamps + 1
    end
  end
  wasBrakingMu = isBraking

  -- Slip-anchored fused stepdown (tunables at top).
  if slipStepdownCooldown > 0 then slipStepdownCooldown = slipStepdownCooldown - dtPhys end
  if ENABLE_SLIP_STEPDOWN and isBraking and fusedSpeed > SLIP_STEPDOWN_VMIN then
    local st = math.min(math.max(lastEffectiveTargets[maxIdx] or 0.14, SLIP_STEPDOWN_STFLOOR), 0.5)
    local estimatedSlipSpeed = maxWs / (1 - st)  -- where true speed should be
    local expectedGap = estimatedSlipSpeed - maxWs  -- = maxWs * st / (1 - st)
    local measuredGap = fusedSpeed - maxWs
    if measuredGap > SLIP_STEPDOWN_GAPFLOOR and measuredGap > SLIP_STEPDOWN_RATIO * expectedGap then
      slipStepdownTimer = slipStepdownTimer + dtPhys
      if slipStepdownTimer >= SLIP_STEPDOWN_SUSTAIN and slipStepdownCooldown <= 0 then
        -- Land fused SNAPFRAC of the way from the fastest wheel up to the slip-implied speed
        local snapTarget = maxWs + SLIP_STEPDOWN_SNAPFRAC * expectedGap
        fusedSpeed = snapTarget
        fwdVel     = snapTarget
        vLat       = 0
        vVert      = 0
        imuSpeed   = snapTarget
        slipStepdownTimer = 0
        slipStepdownCooldown = SLIP_STEPDOWN_COOLDOWN
        -- Count only real moving-car fires (airspeed > 1 mph)
        if (electrics.values.airspeed or 0) > 0.44704 then
          slipStepdownCount = slipStepdownCount + 1
        end
      end
    else
      slipStepdownTimer = 0
    end
  else
    slipStepdownTimer = 0
  end

  for i = 1, N_WHEELS do fusedPrevWs[i] = latestWheelSpeed[i] end

  -- Low-speed re-anchor (safety.LOWSPEED_REANCHOR)
  if safety.LOWSPEED_REANCHOR and isBraking and fusedSpeed < safety.REANCHOR_SPEED
     and maxWs > 1.0 and maxWs <= imuSpeed then
    fusedSpeed = (1 - safety.REANCHOR_BLEND) * imuSpeed + safety.REANCHOR_BLEND * maxWs
  end

  -- [RECOVERY-PEAK RE-ANCHOR
  if safety.recAnchor.ENABLE and isBraking then
    local ra = safety.recAnchor
    for k = 1, N_WHEELS do
      local cmd = safety.lastAbsCoefs[k] or 1
      local w = safety.recWsF[k]
      w = (w == 0) and latestWheelSpeed[k] or (w * 0.8 + latestWheelSpeed[k] * 0.2)  -- ~2.5ms EMA
      safety.recWsF[k] = w
      if safety.recCool[k] > 0 then safety.recCool[k] = safety.recCool[k] - dtPhys end
      if cmd >= ra.CMD_MAX then
        safety.recSt[k] = 0  -- brake re-applied: discard partial recovery
      elseif safety.recSt[k] == 0 then
        safety.recBase[k] = w; safety.recPeak[k] = w; safety.recSt[k] = 1
      else
        if w > safety.recPeak[k] then safety.recPeak[k] = w end
        if safety.recSt[k] == 1 and safety.recPeak[k] > safety.recBase[k] + ra.RISE_MIN then
          safety.recSt[k] = 2  -- genuine recovery in progress
        end
        if safety.recSt[k] == 2 and w < safety.recPeak[k] - ra.DROP then
          -- rollover while still released => wheel touched ground speed at recPeak
          if safety.recCool[k] <= 0 and safety.recPeak[k] > 1.0 then
            local corr = safety.recPeak[k] - fusedSpeed
            if corr > -ra.GATE and corr < ra.GATE then  -- reject spike-sized disagreements
              if corr > ra.CORR_MAX then corr = ra.CORR_MAX
              elseif corr < -ra.CORR_MAX then corr = -ra.CORR_MAX end
              fusedSpeed = fusedSpeed + corr
              fwdVel     = fwdVel + corr
              imuSpeed   = imuSpeed + corr
              safety.recCount = safety.recCount + 1
              safety.recCool[k] = ra.COOLDOWN
              electrics.values.abs_recanchor = safety.recCount
            end
          end
          safety.recSt[k] = 0
        end
      end
    end
  else
    for k = 1, N_WHEELS do safety.recSt[k] = 0; safety.recWsF[k] = 0 end
  end

  if maxWs < STANDSTILL_WS_THRESHOLD and fusedSpeed < STANDSTILL_FUSED_THRESHOLD
     and math.abs(latestSensorY) < STANDSTILL_ACCEL_MAX then
    standstillCounter = standstillCounter + 1
    if standstillCounter >= STANDSTILL_PHYS_TICKS then
      fusedSpeed = 0
      imuSpeed = 0
      fwdVel = 0
      vLat = 0
      vVert = 0
    end
  else
    standstillCounter = 0
  end


end


-- runTick(dt) ,  200Hz: wheel-avg + PID + D-estimator
local function runTick(dt)
  if zCount > 0 then
    for i=1, N_WHEELS do
      if wsMin2[i] == 9999 then
        latestWheelSpeed[i] = wsMin1[i]
      else
        latestWheelSpeed[i] = (wsMin1[i] + wsMin2[i]) / 2
      end
      wsMin1[i] = 9999
      wsMin2[i] = 9999
    end
  end
  local brakeInput = input.brake or 0
  local brakingJustStarted = brakeInput > 0 and not wasBraking
  wasBraking = brakeInput > 0
  local isBraking = brakeInput > 0

  local abstelem = extensions.abstelemv2
  local haveTelem = abstelem ~= nil and abstelem.setBrakes ~= nil

  -- Brake-event recorder: track peak fused-vs-airspeed divergence per event
  do
    local airspeed = electrics.values.airspeed or 0
    local pressed = brakeInput > BRAKE_EVENT_PRESS_THRESHOLD
    if pressed and airspeed > BRAKE_EVENT_MIN_AIR then
      if currentBrakeEvent == nil then
        currentBrakeEvent = {
          peakDiffAbs = 0, peakDiffMs = 0,  -- m/s absolute + signed; UI converts to mph for display
          airAtPeak = airspeed, fusedAtPeak = fusedSpeed,
          startAir = airspeed, startTime = brakeSimTime, duration = 0,
        }
      end
      currentBrakeEvent.duration = brakeSimTime - currentBrakeEvent.startTime
      -- Track peak ABSOLUTE divergence (m/s) ,  only above 5 mph so low-speed sensor noise doesn't
      if airspeed >= BRAKE_EVENT_MIN_PEAK_AIR then
        local diffMs = fusedSpeed - airspeed
        if math.abs(diffMs) > currentBrakeEvent.peakDiffAbs then
          currentBrakeEvent.peakDiffAbs = math.abs(diffMs)
          currentBrakeEvent.peakDiffMs = diffMs
          currentBrakeEvent.airAtPeak = airspeed
          currentBrakeEvent.fusedAtPeak = fusedSpeed
        end
      end
    elseif currentBrakeEvent and not pressed then
      if currentBrakeEvent.duration >= BRAKE_EVENT_MIN_DUR then
        table.insert(brakeEvents, 1, currentBrakeEvent)
        while #brakeEvents > BRAKE_EVENT_MAX do table.remove(brakeEvents) end
        saveBrakeEvents()
      end
      currentBrakeEvent = nil
    end
  end

  -- Reverse bypass: no PID modulation when moving backward.
  if motionDirection < 0 then
    local arcadeWantsReverseThrottle = (electrics.values.throttle or 0) > 0.05
    local effectiveBrake = arcadeWantsReverseThrottle and 0 or brakeInput

    for i = 1, N_WHEELS do
      slipIntegral[i] = 0
      lastSlipError[i] = 0
      prevTickWheelSpeed[i] = latestWheelSpeed[i]
    end
    if effectiveBrake > 0 then
      local cmd = {}
      for i = 1, N_WHEELS do cmd[wheelToBrakeMap[i]] = effectiveBrake end
      if haveTelem then abstelem.setBrakes(cmd) end
    else
      if haveTelem and abstelem.releaseBrakes then abstelem.releaseBrakes() end
    end
    return
  end

  -- Wheel-average speed estimator (200Hz in the PID tick).
  do
    local goodSpeeds = {}
    local goodSum, goodCount = 0, 0
    for i = 1, N_WHEELS do
      local wDecel = (latestWheelSpeed[i] - wa.prevWs[i]) / dt

      wa.decelIdx[i] = (wa.decelIdx[i] % wa.LOCK_WINDOW) + 1
      wa.decelBuf[i][wa.decelIdx[i]] = wDecel

      if wa.locked[i] then
        local latest = wa.decelBuf[i][wa.decelIdx[i]]
        if latest >= 0 then
          local hasPositive = false
          for j = 1, wa.LOCK_WINDOW do
            local v = wa.decelBuf[i][j]
            if v and v > 0 then
              hasPositive = true
              break
            end
          end
          if hasPositive then
            wa.locked[i] = false
          end
        end
      else
        if wDecel < wa.LOCK_DECEL then
          wa.locked[i] = true
        end
      end

      if not wa.locked[i] and wDecel > -0.01 then
        goodSum = goodSum + latestWheelSpeed[i]
        goodCount = goodCount + 1
        goodSpeeds[#goodSpeeds + 1] = latestWheelSpeed[i]
      end
    end
    wa.updated = goodCount > 0
    if wa.updated then
      wa.speed = goodSum / goodCount
    end

    -- Two-wheel consensus: pairs within 2mph of each other
    wa.twoUpdated = false
    for a = 1, #goodSpeeds - 1 do
      for b = a + 1, #goodSpeeds do
        if math.abs(goodSpeeds[a] - goodSpeeds[b]) < 0.894 then
          wa.speedTwo = (goodSpeeds[a] + goodSpeeds[b]) / 2
          wa.twoUpdated = true
        end
      end
    end

    for i = 1, N_WHEELS do wa.prevWs[i] = latestWheelSpeed[i] end
  end

  -- carSpeed = fusedSpeed only (this is the 1F variant ,  no virtualAirspeed)
  local carSpeed = fusedSpeed
  absSpeedSource.wheelAvgActive = false
  absSpeedSource.fusedActive = true

  -- [SLIP GOVERNOR] pack slip feeds the Burckhardt fit, updating lambda* and trim
  if slipGov.ENABLE then
    local vsRef = math.max(fusedSpeed, 0.5)
    local sSum, wsSum = 0, 0
    for i = 1, N_WHEELS do
      local si = (vsRef - latestWheelSpeed[i]) / vsRef
      if si < 0 then si = 0 elseif si > 1 then si = 1 end
      sSum = sSum + si
      wsSum = wsSum + latestWheelSpeed[i]
    end
    local sPack = sSum / N_WHEELS
    slipGov.sPack = sPack
    local wsAvg = wsSum / N_WHEELS

    -- 10ms-EMA derivatives of pack wheel speed and fusedSpeed (for force fit + guard)
    if slipGov.wsAvgPrev == nil then
      slipGov.wsAvgPrev = wsAvg; slipGov.fusedPrev = fusedSpeed
    end
    local aF = dt / (dt + 0.010)
    slipGov.wsAvgF = slipGov.wsAvgF + aF * (((wsAvg - slipGov.wsAvgPrev) / dt) - slipGov.wsAvgF)
    slipGov.fusedF = slipGov.fusedF + aF * (((fusedSpeed - slipGov.fusedPrev) / dt) - slipGov.fusedF)
    slipGov.wsAvgPrev = wsAvg; slipGov.fusedPrev = fusedSpeed

    if isBraking and carSpeed > slipGov.SPEED_GATE then
      -- normalization constant: total max brake torque (per-session constant)
      if slipGov.F0 <= 0 then
        local tsum = 0
        for i = 1, N_WHEELS do tsum = tsum + (origBrakeTorque[i] or 0) end
        slipGov.F0 = math.max(tsum, 1)
      end
      -- Burckhardt RLS on decimated (sPack, normalized pack road-torque) samples (~250 Hz).
      slipGov.decim = slipGov.decim + 1
      if slipGov.decim >= 8 and sPack > 0.01 and sPack < 0.6 then
        slipGov.decim = 0
        local tqSum = 0
        for i = 1, N_WHEELS do tqSum = tqSum + (ph.simulatedTorque[i] or 0) end
        local fN = (tqSum + (4 * 1.2 / 0.33) * slipGov.wsAvgF) / slipGov.F0
        if fN > 0.05 and fN < 2.0 then
          if slipGov.th == nil then
            slipGov.th, slipGov.P, slipGov.ew = {}, {}, {}
            for b = 1, 4 do
              slipGov.th[b] = {1.0, 0.5}
              slipGov.P[b] = {10, 0, 0, 10}
              slipGov.ew[b] = 0.01
            end
          end
          for b = 1, 4 do
            local c2v = slipGov.c2[b]
            local h1 = 1 - math.exp(-c2v * sPack)
            local h2 = -sPack
            local P = slipGov.P[b]
            local p11, p12, p21, p22 = P[1] / 0.9995, P[2] / 0.9995, P[3] / 0.9995, P[4] / 0.9995
            local th = slipGov.th[b]
            local e = fN - (h1 * th[1] + h2 * th[2])
            local Sb = h1 * (p11 * h1 + p12 * h2) + h2 * (p21 * h1 + p22 * h2) + 1.0
            local k1 = (p11 * h1 + p12 * h2) / Sb
            local k2 = (p21 * h1 + p22 * h2) / Sb
            th[1] = th[1] + k1 * e
            th[2] = th[2] + k2 * e
            local hp1 = h1 * p11 + h2 * p21
            local hp2 = h1 * p12 + h2 * p22
            P[1] = p11 - k1 * hp1; P[2] = p12 - k1 * hp2
            P[3] = p21 - k2 * hp1; P[4] = p22 - k2 * hp2
            slipGov.ew[b] = 0.999 * slipGov.ew[b] + 0.001 * e * e
          end
          slipGov.fitN = slipGov.fitN + 1
        end
      end
      -- lambda* from best-fit c2 (closed form), rate-limited; SEED until fit trusted
      local lamTgt = slipGov.SEED
      if slipGov.fitN >= slipGov.FIT_MIN_N and slipGov.th then
        local best, bew = 1, 1e30
        for b = 1, 4 do
          if slipGov.ew[b] < bew then bew = slipGov.ew[b]; best = b end
        end
        local c1v, c3v, c2v = slipGov.th[best][1], slipGov.th[best][2], slipGov.c2[best]
        if c1v > 0.05 and c3v > 1e-3 and c1v * c2v > c3v then
          local so = math.log(c1v * c2v / c3v) / c2v
          if so > 0.02 and so < 0.8 then
            if so < slipGov.LAM_MIN then so = slipGov.LAM_MIN
            elseif so > slipGov.LAM_MAX then so = slipGov.LAM_MAX end
            lamTgt = so
          end
        end
      end
      local dl = lamTgt - slipGov.lambdaStar
      local dlMax = slipGov.LAM_SLEW * dt
      if dl > dlMax then dl = dlMax elseif dl < -dlMax then dl = -dlMax end
      slipGov.lambdaStar = slipGov.lambdaStar + dl

      -- pack-runaway guard: wheels outpacing car decel while meaningfully slipping
      local relDecel = slipGov.wsAvgF - slipGov.fusedF
      if relDecel < slipGov.GUARD_DECEL and sPack > 0.6 * slipGov.lambdaStar then
        if slipGov.guardHoldLeft <= 0 then slipGov.events = slipGov.events + 1 end
        slipGov.guardHoldLeft = slipGov.GUARD_HOLD
      end

      -- authority trim toward target (deadband around lambda*)
      local err = sPack - slipGov.lambdaStar
      local gmTarget = 1.0
      if err < -slipGov.DB then gmTarget = slipGov.GM_MAX
      elseif err > slipGov.DB then gmTarget = slipGov.GM_MIN end
      if slipGov.guardHoldLeft > 0 then
        slipGov.guardHoldLeft = slipGov.guardHoldLeft - dt
        if gmTarget > slipGov.GUARD_GM then gmTarget = slipGov.GUARD_GM end
      end
      local dg = gmTarget - slipGov.gm
      local dgMax = slipGov.RATE * dt
      if dg > dgMax then dg = dgMax elseif dg < -dgMax then dg = -dgMax end
      slipGov.gm = slipGov.gm + dg
    else
      -- not braking / low speed
      local dg = 1.0 - slipGov.gm
      local dgMax = slipGov.RATE * dt
      if dg > dgMax then dg = dgMax elseif dg < -dgMax then dg = -dgMax end
      slipGov.gm = slipGov.gm + dg
      slipGov.guardHoldLeft = 0
    end
  else
    slipGov.gm = 1.0
  end

  -- Low-mu ABS-off extension: sustain-timer on consensusD < MU_THRESH (see safety.lowMu).
  if safety.lowMu.ENABLE and dest.consensusD < safety.lowMu.MU_THRESH then
    safety.lowMuTimer = safety.lowMuTimer + dt
  else
    safety.lowMuTimer = 0
  end

  -- [FORCE SEEKER ,  electrics.values.abs_force_seek==1] per-axle FORCE OBSERVER + slope-driven adaptation of the slip-target base.
  if safety.fsWr and safety.fsWr[1] then
    if isBraking and carSpeed > 4.0 then
      local fs = safety.fsState
      if not fs then
        fs = {baseF = 0, baseR = 0, prevV = {0,0,0,0}, dvF = {0,0,0,0},
              mS = {0,0}, mF = {0,0}, cov = {0,0}, var = {0,0}, warm = 0}
        -- seed from the jbeam bases (front avg / rear avg)
        local sf, nf, sr, nr = 0, 0, 0, 0
        for k = 1, N_WHEELS do
          if safety.fsIsFront[k] then sf = sf + absTargetBase[k]; nf = nf + 1
          else sr = sr + absTargetBase[k]; nr = nr + 1 end
        end
        fs.baseF = nf > 0 and sf / nf or 0.18
        fs.baseR = nr > 0 and sr / nr or 0.14
        -- carry the learned bases between stops, the surface does not change when the pedal lifts
        local carry = safety.fsCarry
        if carry and (electrics.values.abs_force_seek or 0) == 1 then
          fs.baseF, fs.baseR = carry.baseF, carry.baseR
        end
        for k = 1, N_WHEELS do fs.prevV[k] = latestWheelSpeed[k] end
        safety.fsState = fs
      end
      -- per-wheel force, summed per axle; per-axle mean slip
      local Fax, Sax, Nax = {0, 0}, {0, 0}, {0, 0}
      local vRefFS = math.max(carSpeed, 0.5)
      for k = 1, N_WHEELS do
        local wr = safety.fsWr[k]
        local dvRaw = (latestWheelSpeed[k] - fs.prevV[k]) / dt
        fs.prevV[k] = latestWheelSpeed[k]
        fs.dvF[k] = fs.dvF[k] * 0.98 + dvRaw * 0.02  -- EMA ~25ms @2kHz
        local r = (wr and wr.radius) or 0.33
        if r < 0.15 or r > 0.6 then r = 0.33 end
        local T = (wr and (wr.brakingTorque or wr.lastBrakeTorque)) or 0
        local Fk = (1.3 * fs.dvF[k] / r + math.abs(T)) / r
        local ax = safety.fsIsFront[k] and 1 or 2
        Fax[ax] = Fax[ax] + Fk
        Sax[ax] = Sax[ax] + math.max(0, math.min((vRefFS - latestWheelSpeed[k]) / vRefFS, 1))
        Nax[ax] = Nax[ax] + 1
      end
      -- [CROSS-AXLE CURVE SAMPLING] instantaneous slope between the two axle operating points, load-normalized (mu = F/Fz0).
      if Nax[1] > 0 and Nax[2] > 0 and safety.fsFzF > 0 and safety.fsFzR > 0 then
        local dS = Sax[1] / Nax[1] - Sax[2] / Nax[2]
        if dS > 0.008 or dS < -0.008 then  -- need real slip separation
          local kx = (Fax[1] / safety.fsFzF - Fax[2] / safety.fsFzR) / dS
          fs.xslope = fs.xslope and (fs.xslope * 0.98 + kx * 0.02) or kx  -- ~25ms EMA
          electrics.values.abs_xslope = fs.xslope
        end
      end
      if (electrics.values.abs_force_seek or 0) == 1 then
        fs.warm = fs.warm + dt
        local aW = dt / 0.08  -- ~80ms regression window
        for ax = 1, 2 do
          if Nax[ax] > 0 then
            local s, F = Sax[ax] / Nax[ax], Fax[ax]
            fs.mS[ax]  = fs.mS[ax] + (s - fs.mS[ax]) * aW
            fs.mF[ax]  = fs.mF[ax] + (F - fs.mF[ax]) * aW
            fs.cov[ax] = fs.cov[ax] + ((s - fs.mS[ax]) * (F - fs.mF[ax]) - fs.cov[ax]) * aW
            fs.var[ax] = fs.var[ax] + ((s - fs.mS[ax]) ^ 2 - fs.var[ax]) * aW
            if fs.warm > 0.15 and fs.var[ax] > 1e-6 and carSpeed > 6.7 then
              local slope = fs.cov[ax] / fs.var[ax]  -- dF/dslip (N per unit slip)
              if ax == 1 then fs.slopeF = slope else fs.slopeR = slope end
              local step = math.max(-1, math.min(1, slope / 20000)) * 0.10 * dt  -- gain
              if ax == 1 then fs.baseF = math.max(0.08, math.min(0.32, fs.baseF + step))
              else fs.baseR = math.max(0.08, math.min(0.32, fs.baseR + step)) end
            end
          end
        end
        electrics.values.abs_fs_front = fs.baseF
        electrics.values.abs_fs_rear  = fs.baseR
      end
    elseif not isBraking then
      if safety.fsState then
        safety.fsCarry = {baseF = safety.fsState.baseF, baseR = safety.fsState.baseR}
      end
      safety.fsState = nil  -- observer restarts, learned bases survive in fsCarry
    end
  end

  -- Turn handling: read yaw rate (honest ESC sensor) + soft deadband.
  local yawRate = 0
  pcall(function() yawRate = obj:getYawAngularVelocity() or 0 end)
  grip.yawRate = yawRate
  

  -- Reset per-wheel PID and counters on new brake event
  if brakingJustStarted then
    for i = 1, N_WHEELS do
      slipIntegral[i] = 0
      lastSlipError[i] = 0
    end
    safety.snapUp = 0
    safety.snapUpRej = 0
    dest.surfaceChangeCount = 0  -- reset per-braking-event surface-change tally
  end

  -- Per-wheel PID on slip error
  local cmd = {}
  local slipRatios = {}
  local slipErrors = {}
  local absCoefs = {}
  local effectiveTargets = {}
  for i = 1, N_WHEELS do
    cmd[i] = 0
    slipRatios[i] = 0
    slipErrors[i] = 0
    absCoefs[i] = 0
    effectiveTargets[i] = 0
  end

  local lowSpeedBoost = 1.0
  if safety.lowSpeed.ENABLE and carSpeed < safety.lowSpeed.SPEED_THRESH then
    local ratio = 1.0 - (carSpeed / safety.lowSpeed.SPEED_THRESH)
    lowSpeedBoost = 1.0 + (safety.lowSpeed.BOOST_MAX - 1.0) * ratio * ratio
  end
  lastLowSpeedBoost = lowSpeedBoost

  local active_KP = KP
  if ENABLE_DYNAMIC_PID then
    local mu_floor, speed_floor = 0.1, 2.0
    -- [SBR FIX 2026-07-13] grip_mult floored at 1.0
    local grip_mult = math.min(math.max(1.0 / math.max(dest.consensusD, mu_floor), 1.0), 8.0)
    local currentSensorX = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorX) or 0
    local lat_g = math.abs(currentSensorX) / 9.81
    local corner_mult = math.min(1.0 + lat_g, 2.5)
    -- [JBEAM SLIP TARGET] the higher target supplies low-speed authority, so the KP boost is off
    local speed_mult = 1.0
    if not USE_JBEAM_SLIP_TARGET then
      speed_mult = math.min(math.max(1.0, 10.0 / math.max(carSpeed, speed_floor)), 5.0)
      if carSpeed < speed_floor then speed_mult = 5.0 end
    end

    active_KP = KP * grip_mult * corner_mult * speed_mult
    active_KP = math.min(active_KP, KP_MAX)
  end
  current_KP = active_KP

  for i = 1, N_WHEELS do
    local slot = wheelToBrakeMap[i]

    -- Low-mu ABS-off extension
    local absOffSpeed = MIN_SPEED
    if safety.lowMu.ENABLE and safety.lowMuTimer >= safety.lowMu.SUSTAIN
       and carSpeed <= safety.lowMu.SPEED_GATE then
      absOffSpeed = safety.lowMu.ABS_OFF_SPEED
    end
    if carSpeed > absOffSpeed and isBraking then
      -- per-wheel turn-compensated reference speed (vehicle speed + yaw geometry) Yaw compensation disabled
      
      local vRef = math.max(carSpeed, 0.5)
      local slip = math.max(0, math.min((vRef - latestWheelSpeed[i]) / vRef, 1))
      local deepen = ENABLE_LOWSPEED_SLIP_DEEPEN and ((1 + dest.consensusD) / carSpeed) or 0
      local effectiveTarget
      if USE_JBEAM_SLIP_TARGET then
        -- stock ABS target: per-wheel jbeam base + 2/airspeed (matches wheels.lua:510).
        local fsBase = absTargetBase[i]
        local fsSt = safety.fsState
        if fsSt and (electrics.values.abs_force_seek or 0) == 1 then
          fsBase = safety.fsIsFront[i] and fsSt.baseF or fsSt.baseR
        end
        -- test channel: replaces the jbeam base so a sweep can locate the real optimum
        local ovr = electrics.values.abs_base_override or 0
        if ovr > 0.02 and ovr < 0.90 then fsBase = ovr end
        -- [ICE FIX 2026-07-13] mu-scale the target: the jbeam base + 2/v is DRY-tuned
        local muScale = math.min(math.max(dest.consensusD, 0.15), 1.0)
        effectiveTarget = math.min((fsBase + 2.0 / math.max(carSpeed, 0.5)) * muScale, 1.0)
      else
        effectiveTarget = math.min(slipTargets[i] + deepen, 1.0)
      end
      local slipError = effectiveTarget - slip

      -- KI ceiling-bug fix
      slipIntegral[i] = math.max(INTEGRAL_MIN, math.min(INTEGRAL_MAX, slipIntegral[i] + slipError * KI * dt))

      local slipErrorDerivative = 0
      if lastSlipError[i] ~= 0 then
        slipErrorDerivative = (slipError - lastSlipError[i]) / dt
      end
      lastSlipError[i] = slipError

      -- Derivative low-pass (DERIV_FILTER)
      if safety.DERIV_FILTER then
        lastSlipDerivatives[i] = lastSlipDerivatives[i] * safety.DFILT + slipErrorDerivative * (1 - safety.DFILT)
      else
        lastSlipDerivatives[i] = slipErrorDerivative
      end
      local dTerm = lastSlipDerivatives[i]

      -- KI already baked into slipIntegral above, so the integral term is added as-is here.
      local absCoef = math.max(0.0, math.min(1,
        slipError * current_KP + slipIntegral[i] + dTerm * KD))

      -- Low-speed brake boost
      if carSpeed < safety.lowSpeed.SPEED_THRESH then
        absCoef = math.min(1.0, absCoef * lowSpeedBoost)
      end

      -- Lockup guard: wheel decel too fast -> soft dump brake to 70% immediately
      local wheelDecel = (latestWheelSpeed[i] - prevTickWheelSpeed[i]) / dt
      if wheelDecel < WHEEL_DECEL_LIMIT then
        absCoef = absCoef * 0.70
        slipIntegral[i] = math.max(slipIntegral[i], 0)  -- no negative windup
      end

      -- [SLIP GOVERNOR] common-mode authority trim (computed once per tick above the loop)
      if slipGov.ENABLE then
        absCoef = absCoef * slipGov.gm
        if absCoef > 1.0 then absCoef = 1.0 elseif absCoef < 0.0 then absCoef = 0.0 end
      end

      slipRatios[i] = slip
      slipErrors[i] = slipError
      lastSlipErrors[i] = slipError
      -- lastSlipDerivatives[i] is set above (it's the filter state); do NOT overwrite with the raw value.

      absCoefs[i] = absCoef
      effectiveTargets[i] = effectiveTarget
      safety.slipRatios[i] = slip

      cmd[slot] = absCoef * brakeInput
    else
      cmd[slot] = brakeInput
      slipIntegral[i] = 0
      lastSlipError[i] = 0
      absCoefs[i] = 1.0
      effectiveTargets[i] = slipTargets[i]
      safety.slipRatios[i] = 0
    end
  end
  for i = 1, N_WHEELS do safety.lastAbsCoefs[i] = absCoefs[i] end
  for i = 1, N_WHEELS do lastEffectiveTargets[i] = effectiveTargets[i] end

  -- Brake Delay Simulator & Global Peak-Hunter
  local totalTorque = 0
  local avgWheelDecel = 0
  local decelCount = 0
  
  for i = 1, N_WHEELS do
    local commandedTorque = (safety.lastAbsCoefs[i] or 1) * (origBrakeTorque[i] or 0) * brakeInput
    local current = ph.simulatedTorque[i] or 0
    if commandedTorque > current then
      current = math.min(current + (ph.brakeInRate[i] or 9999) * dt, commandedTorque)
    else
      current = math.max(current - (ph.brakeOutRate[i] or 9999) * dt, commandedTorque)
    end
    ph.simulatedTorque[i] = current
    totalTorque = totalTorque + current
    
    local wDecel = (prevTickWheelSpeed[i] - latestWheelSpeed[i]) / dt
    if wDecel > 0.1 then 
      avgWheelDecel = avgWheelDecel + wDecel
      decelCount = decelCount + 1
    end
  end

  local throttleInput = input.throttle or 0
  if ph.ENABLE_THROTTLE_LOCKOUT then
    ph.seekerSuppressed = (brakeInput > 0.05 and throttleInput > 0.05)
  else
    ph.seekerSuppressed = false
  end

  if ph.ENABLE and isBraking and not ph.seekerSuppressed then
    local latAccel = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorX) or 0
    local isStraight = math.abs(latAccel) < 3.0  -- roughly 0.3g
    
    
    if carSpeed > 6.7 and isStraight then
      if ph.phase == 0 then  -- STEP PHASE
        -- Adaptive step size based on speed (fusedSpeed mapped from 15mph to 60mph)
        local speedRatio = math.max(0, math.min(1, (carSpeed - 6.7) / 20.0))
        local stepSize = 0.005 + (0.05 - 0.005) * speedRatio
        
        if ph.turn == 1 then
          ph.frontOffset = math.max(-0.05, math.min(0.15, ph.frontOffset + ph.frontDirection * stepSize))
        else
          ph.rearOffset = math.max(-0.05, math.min(0.15, ph.rearOffset + ph.rearDirection * stepSize))
        end
        ph.phase = 1
        ph.settleTicks = 0
        
      elseif ph.phase == 1 then  -- SETTLE PHASE (TIME-BASED - 2026-07-12
        -- a settled-slip gate never holds for 8 consecutive ticks under aggressive ABS
        ph.settleTicks = ph.settleTicks + 1
        if ph.settleTicks >= 8 then  -- 40ms at 200Hz
          ph.phase = 2
          ph.measureTicks = 0
          ph.accelSum = 0
        end

      elseif ph.phase == 2 then  -- MEASURE PHASE (ACCELEROMETER-BASED decel - 2026-07-12
        -- WAS decel = (startSpeed - fusedSpeed)/time, which differences the noisy fusedSpeed.
        if isStraight then
          ph.accelSum = ph.accelSum + latestSensorY  -- raw forward decel m/s^2 (>0 while braking)
          ph.measureTicks = ph.measureTicks + 1
          if ph.measureTicks >= 10 then  -- 50ms averaged
            ph.phase = 3
          end
        else
          ph.phase = 1
          ph.settleTicks = 0
        end

      elseif ph.phase == 3 then  -- EVALUATE PHASE
        local decel = (ph.measureTicks > 0) and (ph.accelSum / ph.measureTicks) or 0  -- mean accel decel
        
        if ph.turn == 1 then
          local diff = decel - ph.lastFrontEfficiency
          if math.abs(diff) < math.max(0.001, math.abs(ph.lastFrontEfficiency) * 0.005) then
            -- Flat, hold direction (kills noise thrashing bug)
          elseif diff < 0 then
            ph.frontDirection = -ph.frontDirection
          end
          ph.lastFrontEfficiency = decel
          ph.turn = 2  -- handoff
        else
          local diff = decel - ph.lastRearEfficiency
          if math.abs(diff) < math.max(0.001, math.abs(ph.lastRearEfficiency) * 0.005) then
            -- Flat, hold direction
          elseif diff < 0 then
            ph.rearDirection = -ph.rearDirection
          end
          ph.lastRearEfficiency = decel
          ph.turn = 1  -- handoff
        end
        ph.phase = 0
      end
    end
  else
    if not isBraking then
      if carSpeed > 13.41 then  -- ~30mph (Reset fully indicating a new driving context)
        ph.frontOffset = 0
        ph.rearOffset = 0
        ph.lastFrontEfficiency = 0
        ph.lastRearEfficiency = 0
      else
        -- Soft decay toward 0 instead of snap to 0
        ph.frontOffset = ph.frontOffset * 0.999
        ph.rearOffset = ph.rearOffset * 0.999
      end
      ph.phase = 0
      ph.settleTicks = 0
      ph.measureTicks = 0
      ph.turn = 1
    end
  end

  -- D estimator: peak sensorY decel over a sliding window, D = peak / g.
  if isBraking and carSpeed > MIN_ADAPT_SPEED then
    local tickSensorX = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorX) or 0
    local avgZ = (zCount > 0) and (zAccum / zCount) or 0
    -- wheel speeds are already averaged at the top of runTick
    zAccum = 0
    zCount = 0

    -- Heavy EMA smoothing on Z (alpha = 0.05, roughly 0.1s time constant at 200Hz)
    filtZ = filtZ + (avgZ - filtZ) * 0.05

    local roll, pitch = 0, 0
    pcall(function() roll, pitch, _ = obj:getRollPitchYaw() end)
    local gravity_z = 9.81 * math.cos(pitch) * math.cos(roll)
    local true_z_load = filtZ + gravity_z
    local a_z_safe = math.max(math.abs(true_z_load), 0.1)

    local estimated_mu = math.sqrt((latestSensorY * latestSensorY) + (tickSensorX * tickSensorX)) / a_z_safe
    local measuredDecel = estimated_mu * 9.81

    local maxSlip = 0
    for j = 1, N_WHEELS do
      if (safety.slipRatios[j] or 0) > maxSlip then
        maxSlip = safety.slipRatios[j]
      end
    end

    -- Effort gate (dest.EFFORT_GATE)
    local absWorking = false
    if dest.EFFORT_GATE then
      for j = 1, N_WHEELS do
        if (absCoefs[j] or 1) < grip.SAT and latestWheelSpeed[j] > grip.ROLL_MIN then
          absWorking = true
          break
        end
      end
    end

    local baseGate = (maxSlip >= dest.SLIP_GATE or absWorking)

    if baseGate then
      dest.windowIdx = (dest.windowIdx % dest.WINDOW_SIZE) + 1
      dest.window[dest.windowIdx] = measuredDecel

      local aggDecel = aggregateDecelWindow()  -- peak / mean / topN per D_AGG_MODE

      -- 1.00 (2026-07-12, : was 0.91 (deliberate conservative D bias).
      local instantD = math.max(dest.EST_MIN, math.min(dest.EST_MAX, aggDecel / 9.81 * 1.00))

      -- Surface-change detection: D jump >40% = reset window
      dest.stableTicks = dest.stableTicks + 1
      if dest.stableTicks > dest.SETTLE_TICKS then
        local dChange = math.abs(instantD - dest.stableD) / math.max(dest.stableD, 0.1)
        if dChange > dest.CHANGE_THRESHOLD then
          dest.window = {}
          dest.windowIdx = 1
          dest.window[1] = measuredDecel
          dest.retroResets = dest.retroResets + 1
          dest.surfaceChangeCount = dest.surfaceChangeCount + 1
          dest.stableTicks = 0
          dest.consensusD = instantD
          ph.frontOffset = 0; ph.rearOffset = 0
          ph.lastFrontEfficiency = 0; ph.lastRearEfficiency = 0
        end
        dest.stableD = dest.consensusD
      end

      -- Smooth consensus toward peak-decel estimate.
      if carSpeed > dest.FREEZE_SPEED then
        dest.consensusD = dest.consensusD * dest.SMOOTHING + instantD * (1.0 - dest.SMOOTHING)
      end

      -- Map D to slip target
      dest.baseTarget = math.max(SLIP_TARGET_MIN, math.min(SLIP_TARGET_MAX,
        0.04 + dest.consensusD * 0.10))

      -- Track average D over the ABS event
      local isAbsActive = false
      for j = 1, N_WHEELS do
        if (safety.lastAbsCoefs[j] or 1) < 0.99 then
          isAbsActive = true
          break
        end
      end
      
      if isAbsActive then
        if not dest.absEventActive then
          dest.absEventActive = true
          dest.absEventSum = 0
          dest.absEventTicks = 0
          dest.absEventTotalTicks = 0
          dest.retroResets = 0
        end
        
        dest.absEventTotalTicks = dest.absEventTotalTicks + 1
        
        -- Filter out first 100ms (20 ticks at 200Hz) and any readings below 18mph (8.04 m/s)
        if dest.absEventTotalTicks > 20 and carSpeed >= 8.04 then
          dest.absEventSum = dest.absEventSum + dest.consensusD
          dest.absEventTicks = dest.absEventTicks + 1
        end
      else
        if dest.absEventActive then
          if dest.absEventTicks >= 10 and dest.retroResets == 0 then  -- >50ms at 200Hz
            local avgD = dest.absEventSum / dest.absEventTicks
            dest.consensusD = avgD
            dest.stableD = avgD
          end
          dest.absEventActive = false
        end
      end

      if not grip.ENABLE_PERWHEEL_D then
        -- Escape hatch: original global broadcast ,  every wheel takes the global target.
        for j = 1, N_WHEELS do
          grip.Dwheel[j] = dest.consensusD
          local isFront = false
          for _, li in ipairs(frontLogicalIndices) do if li == j then isFront = true end end
          local trim = isFront and ph.frontOffset or ph.rearOffset
          local baseSlip = dest.baseTarget
          local finalTarget = math.max(SLIP_TARGET_MIN, math.min(SLIP_TARGET_MAX, baseSlip + trim))
          slipTargets[j] = slipTargets[j] * TARGET_SMOOTHING + finalTarget * (1.0 - TARGET_SMOOTHING)
        end
      else
        -- Per-wheel ANCHORED brake-acceptance.
        local aLong = dest.consensusD * grip.GRAV
        local dFz = grip.mass * aLong * grip.H_CG / (2 * grip.WHEELBASE)
        local sumG, nG = 0, 0
        for j = 1, N_WHEELS do
          local maxT = origBrakeTorque[j] or 0
          local applied = absCoefs[j] * maxT
          -- (2) front/rear distinction via frontLogicalIndices: front gets +dFz, rear gets -dFz
          local isFront = false
          for _, fi in ipairs(frontLogicalIndices) do
            if j == fi then isFront = true; break end
          end
          local fz = grip.Fz0[j] + (isFront and dFz or -dFz)
          if fz < grip.FZ_MIN then fz = grip.FZ_MIN end
          local gj = applied / fz
          -- confident only while ABS-regulating unsaturated, rolling, with real brake applied
          local conf = (absCoefs[j] < grip.SAT)
            and (latestWheelSpeed[j] > grip.ROLL_MIN)
            and (applied > grip.TORQUE_MIN_FRAC * maxT)
          grip.confident[j] = conf

          if conf then
            grip.gEMA[j] = grip.gEMA[j] * grip.G_SMOOTH + gj * (1 - grip.G_SMOOTH)
            sumG = sumG + grip.gEMA[j]
            nG = nG + 1
          end
        end

        if nG > 0 then
          local gBar = sumG / nG
          if gBar < grip.EPS then gBar = grip.EPS end
          for j = 1, N_WHEELS do
            -- non-confident wheels drift their EMA back toward the confident mean
            if not grip.confident[j] then
              grip.gEMA[j] = grip.gEMA[j] + (gBar - grip.gEMA[j]) * grip.DECAY
            end
            -- anchored: mean of D stays = dest.consensusD; ratio carries the per-wheel split
            local Dj = dest.consensusD * (grip.gEMA[j] / gBar)
            if Dj < dest.EST_MIN then Dj = dest.EST_MIN elseif Dj > dest.EST_MAX then Dj = dest.EST_MAX end
            grip.Dwheel[j] = Dj
            local isFront = false
            for _, li in ipairs(frontLogicalIndices) do if li == j then isFront = true end end
            local trim = isFront and ph.frontOffset or ph.rearOffset
            local baseTarget = math.max(SLIP_TARGET_MIN, math.min(SLIP_TARGET_MAX, 0.04 + Dj * 0.10 + trim))
            slipTargets[j] = slipTargets[j] * TARGET_SMOOTHING + baseTarget * (1.0 - TARGET_SMOOTHING)
          end
        else
          -- no confident wheel (light braking / all locked / first ticks): fall back to global
          for j = 1, N_WHEELS do
            grip.Dwheel[j] = dest.consensusD
            local isFront = false
            for _, li in ipairs(frontLogicalIndices) do if li == j then isFront = true end end
            local trim = isFront and ph.frontOffset or ph.rearOffset
            local baseSlip = dest.baseTarget
          local finalTarget = math.max(SLIP_TARGET_MIN, math.min(SLIP_TARGET_MAX, baseSlip + trim))
            slipTargets[j] = slipTargets[j] * TARGET_SMOOTHING + finalTarget * (1.0 - TARGET_SMOOTHING)
          end
        end
      end
    end
  end

  -- Push brake commands
  if haveTelem then
    abstelem.setBrakes(cmd)
  end
  electrics.values.abs_cmd_fr = cmd[wheelToBrakeMap[3]] or 0
  electrics.values.abs_cmd_fl = cmd[wheelToBrakeMap[4]] or 0
  electrics.values.abs_cmd_rr = cmd[wheelToBrakeMap[1]] or 0
  electrics.values.abs_cmd_rl = cmd[wheelToBrakeMap[2]] or 0

  local telemetryLogger = extensions.absTelemetryLogger
  if telemetryLogger and telemetryLogger.setCustomTelemetry then
    telemetryLogger.setCustomTelemetry("Dynamic_ABS", {
      fusedSpeed = fusedSpeed,
      nextD = dest.consensusD,
      slipTargets = {slipTargets[1], slipTargets[2], slipTargets[3], slipTargets[4]},
      pidOut = {safety.lastAbsCoefs[1] or 1, safety.lastAbsCoefs[2] or 1, safety.lastAbsCoefs[3] or 1, safety.lastAbsCoefs[4] or 1},
      maxBrake = {origBrakeTorque[1] or 0, origBrakeTorque[2] or 0, origBrakeTorque[3] or 0, origBrakeTorque[4] or 0},
      trimOffset = ph.rearOffset or 0,
      trimDirection = ph.rearDirection or 1,
      pidGains = {KP, KI, KD},
      imuSpeed = imuSpeed,
      imuClamps = safety.imuClamps,
      fwdVel = fwdVel,
      vLat = vLat,
      vVert = vVert,
      effectiveTarget = {lastEffectiveTargets[4], lastEffectiveTargets[3], lastEffectiveTargets[2], lastEffectiveTargets[1]},
      gripD = {grip.Dwheel[4], grip.Dwheel[3], grip.Dwheel[2], grip.Dwheel[1]},
      gripConfident = (function() local n=0 for j=1,4 do if grip.confident[j] then n=n+1 end end return n end)(),
      lowSpeedBoost = lastLowSpeedBoost,
      frontTrimOffset = ph.frontOffset or 0,
      frontTrimDirection = ph.frontDirection or 1,
      slipError = {lastSlipErrors[4], lastSlipErrors[3], lastSlipErrors[2], lastSlipErrors[1]},
      slipIntegralState = {slipIntegral[4], slipIntegral[3], slipIntegral[2], slipIntegral[1]},
      slipDerivative = {lastSlipDerivatives[4], lastSlipDerivatives[3], lastSlipDerivatives[2], lastSlipDerivatives[1]},
      snapUpCount = safety.snapUp,
      fsSeek = electrics.values.abs_force_seek or 0,
      baseOverride = electrics.values.abs_base_override or 0,
      fsBaseF = safety.fsState and safety.fsState.baseF or 0,
      fsBaseR = safety.fsState and safety.fsState.baseR or 0,
      fsSlopeF = safety.fsState and safety.fsState.slopeF or 0,
      fsSlopeR = safety.fsState and safety.fsState.slopeR or 0,
      fsXSlope = safety.fsState and safety.fsState.xslope or 0,
      -- [SLIP GOVERNOR] logger reads this pushed table, not getTelemetry; keep both in sync
      trueSlipPack = slipGov.sPack,
      lambdaStar = slipGov.lambdaStar,
      govMult = slipGov.gm,
      govEvents = slipGov.events
    })
  end

  -- prevTickWheelSpeed for next tick's lockup guard
  for i = 1, N_WHEELS do
    prevTickWheelSpeed[i] = latestWheelSpeed[i]
  end
end


-- update(dtPhys) ,  2kHz orchestrator
local function update(dtPhys)
  detectMu(dtPhys)
  brakeSimTime = brakeSimTime + dtPhys

  timeAccum = timeAccum + dtPhys
  while timeAccum >= TICK_STEP do
    runTick(TICK_STEP)
    timeAccum = timeAccum - TICK_STEP
  end

  -- Data Logging (50Hz) to track down IMU drift
  local brakeInput = electrics.values.brake or 0
  if brakeInput > 0.01 then
    if not isLogging then
      isLogging = true
      logData = {}
      table.insert(logData, "Time,Airspeed,FusedSpeed,ImuSpeed,FwdVel,VLat,VVert,Ax,Ay,Az,YawRate,PitchRate,RollRate,Pitch,Roll")
      logTimer = 0
    end
    
    logTimer = logTimer + dtPhys
    if logTimer >= 0.02 then
      logTimer = 0
      local row = string.format("%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f", 
        brakeSimTime, electrics.values.airspeed or 0, fusedSpeed, imuSpeed, fwdVel, vLat, vVert,
        -latestSensorY, (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorX) or 0,
        (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorZ) or 0,
        (obj:getYawAngularVelocity() or 0), pitchRateLog, rollRateLog, pitchLog, rollLog
      )
      table.insert(logData, row)
    end
  else
    if isLogging then
      isLogging = false
      if #logData > 1 then
        local file = io.open("abs_imu_log.csv", "w")
        if file then
          file:write(table.concat(logData, "\n"))
          file:close()
        end
      end
      logData = {}
    end
  end

  -- Broadcast brake events to UI at 5Hz
  brakeUiAccum = brakeUiAccum + dtPhys
  if brakeUiAccum >= 0.2 then
    brakeUiAccum = 0
    if guihooks then
      guihooks.trigger('updateABSBrakeEvents', {
        events = brakeEvents,
        live = currentBrakeEvent,
      })
    end
  end

  -- Publish D-estimator state for external reading (test harness etc.)
  electrics.values.abs_force_seek = electrics.values.abs_force_seek or 1
  electrics.values.abs_consensusD = dest.consensusD
  electrics.values.abs_baseTarget = dest.baseTarget
  electrics.values.abs_retroResets = dest.retroResets
  electrics.values.abs_surfaceChangeCount = dest.surfaceChangeCount
  electrics.values.abs_surface = getCondition(slipTargets[1])
  electrics.values.abs_wheelAvgSpeed = wa.speed
  electrics.values.abs_wheelAvgSpeedTwo = wa.speedTwo
  -- Per-wheel surface grip (RR/RL/FR/FL) + how many wheels are confidently sensed
  electrics.values.abs_D_RR = grip.Dwheel[1]
  electrics.values.abs_D_RL = grip.Dwheel[2]
  electrics.values.abs_D_FR = grip.Dwheel[3]
  electrics.values.abs_D_FL = grip.Dwheel[4]
  do
    local confCount = 0
    for j = 1, N_WHEELS do if grip.confident[j] then confCount = confCount + 1 end end
    electrics.values.abs_grip_conf = confCount
  end
  electrics.values.abs_yawRate = grip.yawRate  -- rad/s; for verifying the turn-correction sign

  -- UI update at ~30Hz (1/30 s).
  uiAccum = uiAccum + dtPhys
  if uiAccum >= 1/30 then
    uiAccum = 0
    if guihooks then
      guihooks.trigger('updateABSGrip', {
        RR = { surfaceMu = string.format("%.2f", grip.Dwheel[1]), slipMu = string.format("%.2f", safety.slipRatios[1]) },
        RL = { surfaceMu = string.format("%.2f", grip.Dwheel[2]), slipMu = string.format("%.2f", safety.slipRatios[2]) },
        FR = { surfaceMu = string.format("%.2f", grip.Dwheel[3]), slipMu = string.format("%.2f", safety.slipRatios[3]) },
        FL = { surfaceMu = string.format("%.2f", grip.Dwheel[4]), slipMu = string.format("%.2f", safety.slipRatios[4]) },
        speeds = {
          airspeed = string.format("%.1f", electrics.values.airspeed or 0),
          fusedSpeed = string.format("%.1f", fusedSpeed or 0),
          plausibleSpeed = string.format("%.1f", wa.speed or 0),
          virtualAirspeed = string.format("%.1f", electrics.values.virtualAirspeed or 0),
          snapUpCount = safety.snapUp,
          snapUpRejCount = safety.snapUpRej,
          imuSpeed = string.format("%.1f", imuSpeed or 0),
          imuClampCount = safety.imuClamps,
          slipStepdownCount = slipStepdownCount,
          vLat = string.format("%.1f", vLat or 0),
          yaw2d = string.format("%.2f", grip.yawRate or 0),
          fusedActive = absSpeedSource.fusedActive,
          nextDEstimate = string.format("%.2f", dest.consensusD),
          surfaceChangeCount = dest.surfaceChangeCount,
        }
      })
    end
  end
end


local function reset(jbeamData)
  init(jbeamData)
end


-- Clear brake event history (callable from UI via controller.getControllerSafe).
local function clearBrakeEvents()
  brakeEvents = {}
  currentBrakeEvent = nil
  pcall(jsonWriteFile, BRAKE_EVENT_FILE, {}, false)
  if guihooks then
    guihooks.trigger('updateABSBrakeEvents', { events = {}, live = nil })
  end
end


local function getCounts()
  return {snapUp = safety.snapUp, snapUpRej = safety.snapUpRej,
          imuClamps = safety.imuClamps}
end
local function resetCounts()
  safety.snapUp = 0; safety.snapUpRej = 0; safety.imuClamps = 0
end
local function setPerWheelD(val) grip.ENABLE_PERWHEEL_D = val end

-- Switch the D-estimator window aggregation live (no reload). mode = "peak"|"mean"|"topn".
local function setDAggMode(mode, topn)
  if mode then D_AGG_MODE = mode end
  if topn then D_AGG_TOPN = topn end
  print("[ABS-1FEX] D_AGG_MODE = " .. tostring(D_AGG_MODE) .. "  TOPN = " .. tostring(D_AGG_TOPN))
end

local function getGripDebug()
  return {D = {grip.Dwheel[1], grip.Dwheel[2], grip.Dwheel[3], grip.Dwheel[4]},
          conf = {grip.confident[1], grip.confident[2], grip.confident[3], grip.confident[4]},
          Fz0 = {grip.Fz0[1], grip.Fz0[2], grip.Fz0[3], grip.Fz0[4]},
          consensusD = dest.consensusD, enabled = grip.ENABLE_PERWHEEL_D}
end

local function getTelemetry()
  return {
    fusedSpeed = fusedSpeed,
    nextD = dest.consensusD,
    slipTargets = {slipTargets[1], slipTargets[2], slipTargets[3], slipTargets[4]},
    pidOut = {safety.lastAbsCoefs[1] or 1, safety.lastAbsCoefs[2] or 1, safety.lastAbsCoefs[3] or 1, safety.lastAbsCoefs[4] or 1},
    trimOffset = ph.rearOffset or 0,
    trimDirection = ph.rearDirection or 1,
    pidGains = {KP, KI, KD},
    imuSpeed = imuSpeed,
    imuClamps = safety.imuClamps,
    fwdVel = fwdVel,
    vLat = vLat,
    vVert = vVert,
    effectiveTarget = {lastEffectiveTargets[4], lastEffectiveTargets[3], lastEffectiveTargets[2], lastEffectiveTargets[1]},
    gripD = {grip.Dwheel[4], grip.Dwheel[3], grip.Dwheel[2], grip.Dwheel[1]},
    gripConfident = (function() local n=0 for j=1,4 do if grip.confident[j] then n=n+1 end end return n end)(),
    lowSpeedBoost = lastLowSpeedBoost,
    frontTrimOffset = ph.frontOffset or 0,
    frontTrimDirection = ph.frontDirection or 1,
    slipError = {lastSlipErrors[4], lastSlipErrors[3], lastSlipErrors[2], lastSlipErrors[1]},
    slipIntegralState = {slipIntegral[4], slipIntegral[3], slipIntegral[2], slipIntegral[1]},
    slipDerivative = {lastSlipDerivatives[4], lastSlipDerivatives[3], lastSlipDerivatives[2], lastSlipDerivatives[1]},
    snapUpCount = safety.snapUp,
    -- [SLIP GOVERNOR] logged by absTelemetryLogger as TrueSlipPack/LambdaStar/GovMult/GovEvents
    trueSlipPack = slipGov.sPack,
    lambdaStar = slipGov.lambdaStar,
    govMult = slipGov.gm,
    govEvents = slipGov.events
  }
end

-- ABS warning light.
local function updateGFX(dtSim)
  local braking = (input.brake or 0) > 0.1
  local absActive = false
  if braking then
    for i = 1, N_WHEELS do
      if (safety.lastAbsCoefs[i] or 1) < 0.9 then
        absActive = true
        break
      end
    end
  end

  absLightTimer = absLightTimer + dtSim
  if absLightTimer >= ABS_LIGHT_PULSE_PERIOD then
    absLightTimer = 0
    absLightPulse = absActive and (1 - absLightPulse) or 0
  end

  electrics.values.hasABS = true  -- car is ABS-equipped (telltale allowed to show)
  electrics.values.abs = absLightPulse  -- pulsed telltale value stock dashboards read (>0.5 = lit)
  electrics.values.absActive = absActive  -- steady flag for anything reading absActive
end

M.init = init
M.update = update
M.updateGFX = updateGFX
M.reset = reset
M.clearBrakeEvents = clearBrakeEvents
M.getCounts = getCounts
M.resetCounts = resetCounts
M.setPerWheelD = setPerWheelD
M.setDAggMode = setDAggMode
M.getTelemetry = getTelemetry

M.getGripDebug = getGripDebug

return M
