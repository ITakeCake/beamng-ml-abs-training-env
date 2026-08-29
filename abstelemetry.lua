-- abstelemetry.lua v3.4-ppo_v2 — Telemetry bridge + per-wheel brake control + 2000Hz PD controller
-- (PPO_V2 changes vs v3.0: geometric wheel-order resolution [Dynamic_ABS parity],
--  measured steering [electrics.steering, Dynamic_ABS parity]. Gyro pitch/roll
--  estimate exists but is DEBUG-ONLY — obs pitch/roll are ground truth, reverted 2026-07-11)
--
-- ResidualABS-local copy: rebuilt 2026-08-27 on the PPO_V3_AxleCurriculum v3.4-ppo_v2
-- base (the copy the live .tech userpath has run since 07-11) + the arc-length brake
-- metric. An earlier rebuild used the older MachineTrainerBoy base and silently lost
-- the armBrakeSlam/reorderWheelDataLogical machinery abs_env_incar.py requires
-- (parked as abstelemetry_SACBASE_BAK.lua). Auto-deployed to the userpath by
-- abs_env.py's _install_tech_assets() on every launch from this folder; other
-- training folders carry their own copies and are unaffected by this one.
-- Arc fields (last_brake_avg_g_arc / last_brake_dist_arc) are telemetry-only —
-- not wired into any reward — and unverified live until an env from this folder
-- launches and logs a stop.
--
-- *** IMPORTANT: This file is the source of truth. ***
-- *** Python/ML code must adapt to this Lua, not the other way around. ***
-- *** Do not modify this file to accommodate Python-side changes. ***
--
-- v3.0 CHANGES:
--   Added optional PD brake controller running at 2000Hz physics rate.
--   Enabled via enablePDController(config). Particle filter (60Hz) feeds
--   the slip target via setOptimalSlip(). PD controller does speed estimation,
--   slip calculation, and per-wheel brake modulation at physics rate.
--   When PD is disabled, everything works exactly as v2.1 (ML training path).

local M = {}

local wheelData = {}
local initialized = false
local heartbeat = 0
local simTime = 0

-- Per-wheel brake control
local perWheelMode = false
local brakeCmd = {0, 0, 0, 0}  -- FR, FL, RR, RL
local origBrakeTorque = {}
-- wheelData: 1=RR, 2=RL, 3=FR, 4=FL  — GUARANTEED by reorderWheelDataLogical()
-- (PPO_V2: stock-geometry classification, same method as Dynamic_ABS buildWheelMaps;
-- previously this order was just assumed from the etk800's physical rotator order)
-- brakeCmd:  1=FR, 2=FL, 3=RR, 4=RL
local wheelToBrakeMap = {3, 4, 1, 2}

-- (PPO_V2) IMU-style attitude estimate: pure body-gyro integration
-- (obj:getPitchAngularVelocity / getRollAngularVelocity), zeroed by resetAccum() at
-- each brake-event arm. DEBUG-ONLY as of 2026-07-11 (published as pitch_gyro/
-- roll_gyro + tel_imu_*): the obs pitch/roll reverted to ground-truth orientation
-- per Blake ("for now"). Flip the two blocks in buildSensorData to re-enable.
local imuPitch, imuRoll = 0, 0

-- Per-wheel min speed accumulators
local accumWsMin = {}

-- =====================================================
-- MODE 1: Poll accumulator (2000Hz via onPhysicsStep)
-- =====================================================
local physPrevSpeed = -1

local pollSpeedStart = -1
local pollDtSum = 0
local pollFrames = 0
-- Rolling window for smoothed min/max
local pollGyRollingBuf = {}
local pollGyRollingIdx = 0
local pollGyRollingSize = 6  -- ~3ms at 2000Hz
local pollGySmoothedMin = 999999
local pollGySmoothedMax = -999999

-- gx (lateral) and gz (vertical) windowing — mirrors gy pattern
-- Sourced from IMU sensorX/sensorZ at 2000Hz. Used by SpeedLSTM data export.
local pollGxRollingBuf = {}
local pollGxRollingIdx = 0
local pollGxSmoothedMin = 999999
local pollGxSmoothedMax = -999999
local pollGxSum = 0
local pollGzRollingBuf = {}
local pollGzRollingIdx = 0
local pollGzSmoothedMin = 999999
local pollGzSmoothedMax = -999999
local pollGzSum = 0
local pollYawSum = 0
local pollYawAbsSum = 0
local pollYawMin = 999999
local pollYawMax = -999999

-- Instantaneous values (last physics substep)
local instGy = 0
local instGx = 0
local instGz = 0
local instYaw = 0
local instSpeed = 0

-- Per-60Hz-frame peak and avg (independent of ML poll accumulator)
local gfxPeakGy = 0
local gfxGySum = 0
local gfxGyCount = 0

-- =====================================================
-- MODE 2: Brake event — exact copy of BeamNG's
-- updateBrakingDistance() from wheels.lua
-- 3-state: idle → waiting → measuring → idle
-- Python sets targetSpeed via setTargetSpeed(ms).
-- =====================================================
local brakeState = "idle"
local brakeTargetSpeed = 0     -- m/s, set by Python via setTargetSpeed()
local brakeStartPosition = nil
local brakeStartTime = 0

-- Arc-length accumulator (path distance, not straight-line chord): a policy that
-- yaws while braking shortens the chord below the true path length, which
-- inflates chord-based avg-G. Accumulated every physics tick while measuring;
-- reward code should use min(chord_g, arc_g) once wired to a reward that reads it.
local brakeArcDist = 0
local brakePrevPos = nil

-- Last completed brake event
local lastBrakeAvgGy = 0      -- m/s^2
local lastBrakeAvgG = 0       -- in g's (using powertrain.currentGravity)
local lastBrakeAvgGArc = 0    -- in g's, computed from path (arc) distance instead of chord
local lastBrakeDist = 0       -- meters
local lastBrakeDistArc = 0    -- meters, true path length between measure-start and stop
local lastBrakeStartSpeed = 0 -- m/s (= targetSpeed, the measurement start)
local lastBrakeDuration = 0   -- seconds

-- =====================================================
-- PD CONTROLLER (2000Hz) — optional, enabled by particle ABS
-- When disabled, this entire section is skipped.
-- =====================================================
local pdEnabled = false

-- PD config (set via enablePDController)
local pdSlipGain = 90        -- P gain, dt-scaled (equiv to 1.5 at 60Hz)
local pdDampGain = 6.0       -- D gain, dt-scaled (equiv to 0.1 at 60Hz)
local pdLockupSafety = true
local pdLockupGLimit = 2.5   -- wheel decel in g's triggers safety release
local pdHandoffDecelLimit = 4.905  -- 0.5g in m/s²
local pdSpeedFloor = 3.0     -- m/s minimum when g-force says still moving
local pdNoiseAmount = 0.03   -- ±1.5% noise per step for filter diversity
local pdTireRadius = 0.327   -- tire radius in meters
local pdMinBrake = 0.15      -- floor for brake ratio
local pdMaxBrake = 1.0       -- ceiling for brake ratio

-- PD state (maintained at 2000Hz)
local pdCarSpeed = 0
local pdOptimalSlip = 0.10
local pdBrakeRatio = {0, 0, 0, 0}  -- per wheel (wheelData order: RR, RL, FR, FL)
local pdPrevSlips = {0, 0, 0, 0}
local pdPrevWheelSpeed = {0, 0, 0, 0}
local pdWasBraking = false
local pdAbsActive = false
local pdHandedOff = false
local pdWheelSlips = {0, 0, 0, 0}
local pdLockupTriggered = {false, false, false, false}

-- =====================================================
-- FUSED SPEED ESTIMATOR (2000Hz) — for threshold ABS
-- Wheel-speed average when wheels are trustworthy,
-- raw sensorY integration when they're all locking.
-- Policy (decel filter) pushed in by the ABS module.
-- =====================================================
-- All fused-speed state packed in a single table to stay under Lua's 60-upvalue
-- limit per function. onPhysicsStep already closes over ~50 module-locals.
--
-- Fusion model (v2): integrator as dead-reckoning floor.
--   * Integrator runs every physics tick: speed -= sensorY * dt
--   * During braking: wheels can only pull integrator UP (lockup protection)
--   * During non-braking: wheels are authoritative, integrator re-syncs to them
--   * Output = integrator
-- This avoids the cascade where locked-at-zero wheels pass the decel filter
-- (because their tick-to-tick delta is near zero) and drag carSpeed to garbage.
local fused = {
  decelFilter = -0.001,       -- overridden by setDecelFilter (tunable from ABS)
  speed = 0,                   -- output: fused car speed (m/s)
  integratedSpeed = 0,         -- running integrator anchor
  prevWs = {0, 0, 0, 0},
  initialized = false,
  lastGoodCount = 0,
  accSign = 1,                 -- forward-only for now
  mode = 1,                    -- 1 = wheels tracking, 0 = integrator carrying (lockup)
  lastSensorY = 0,             -- last raw sensorY value (m/s²)
  accelTicks = 0,              -- 2000Hz ticks spent in integrator-carrying mode this brake event
  wheelTicks = 0,              -- 2000Hz ticks spent in wheel-tracking mode this brake event
  brakeWasHeld = false,
}


-- (PPO_V2) Reorder wheelData into LOGICAL order {1=RR, 2=RL, 3=FR, 4=FL} using the
-- stock geometric method — 1:1 port of Dynamic_ABS's buildWheelMaps() (itself a copy
-- of drivingDynamics/sensors/vehicleData.lua -> initSecondStage()). This removes the
-- silent assumption that the jbeam's physical rotator order is RR,RL,FR,FL (true for
-- the etk800, NOT guaranteed for other cars). On failure keeps physical order + warns.
local function reorderWheelDataLogical()
  if #wheelData ~= 4 then return end  -- only meaningful for 4-corner cars
  local ok, err = pcall(function()
    local cornerWheels = { FR = true, FL = true, RR = true, RL = true }

    -- Stock: average corner-wheel position (divides by TOTAL wheel count, as stock does)
    local avgWheelPos = vec3(0, 0, 0)
    for _, wheel in pairs(wheels.wheels) do
      if cornerWheels[wheel.name] then
        avgWheelPos = avgWheelPos + vec3(v.data.nodes[wheel.node1].pos)
      end
    end
    avgWheelPos = avgWheelPos / #wheels.wheels

    -- Stock: reference frame from the vehicle's ref nodes
    local refNodes = v.data.refNodes[0]
    local vectorForward = vec3(v.data.nodes[refNodes.ref].pos) - vec3(v.data.nodes[refNodes.back].pos)
    local vectorUp      = vec3(v.data.nodes[refNodes.up].pos)  - vec3(v.data.nodes[refNodes.ref].pos)
    local vectorRight   = vectorForward:cross(vectorUp)

    -- Stock: classify each corner wheel front/rear + left/right via dot products
    local logicalOf = {}   -- wheel name -> logical idx (1=RR, 2=RL, 3=FR, 4=FL)
    local found = 0
    for _, wheel in pairs(wheels.wheels) do
      if cornerWheels[wheel.name] then
        local wheelVector = vec3(v.data.nodes[wheel.node1].pos) - avgWheelPos
        local dotForward  = vectorForward:dot(wheelVector)
        local dotRight    = vectorRight:dot(wheelVector)
        if dotRight >= 0 then
          logicalOf[wheel.name] = (dotForward >= 0) and 3 or 1   -- FR / RR
        else
          logicalOf[wheel.name] = (dotForward >= 0) and 4 or 2   -- FL / RL
        end
        found = found + 1
      end
    end
    if found ~= 4 then error("classified " .. found .. " corner wheels, need 4") end

    local ordered = {}
    for _, wd in ipairs(wheelData) do
      local li = logicalOf[wd.name]
      if not li then error("wheel '" .. tostring(wd.name) .. "' not classified") end
      ordered[li] = wd
    end
    if not (ordered[1] and ordered[2] and ordered[3] and ordered[4]) then
      error("incomplete logical map")
    end
    wheelData = ordered
    print(string.format(
      "=== TELEMETRY: wheelData logical order (stock geometry method) RR=%s RL=%s FR=%s FL=%s ===",
      wheelData[1].name, wheelData[2].name, wheelData[3].name, wheelData[4].name))
  end)
  if not ok then
    print("=== TELEMETRY WARNING: logical wheel reorder FAILED (" .. tostring(err)
      .. ") — keeping physical rotator order; VERIFY it is RR,RL,FR,FL for this car! ===")
  end
end


local function tryInitWheels()
  wheelData = {}
  local ok1, err1 = pcall(function()
    if wheels and wheels.wheelRotators and wheels.wheelRotatorCount and wheels.wheelRotatorCount > 0 then
      for i = 0, wheels.wheelRotatorCount - 1 do
        local w = wheels.wheelRotators[i]
        if w then
          table.insert(wheelData, {
            name = w.name or ("rot_" .. i),
            ref = w,
            method = "wheelRotator"
          })
        end
      end
    end
  end)

  if #wheelData > 0 then
    reorderWheelDataLogical()  -- (PPO_V2) RR,RL,FR,FL by geometry, not by jbeam order
    initialized = true
    local absInfo = ""
    for i, wd in ipairs(wheelData) do
      origBrakeTorque[i] = wd.ref.brakeTorque or 0
      accumWsMin[i] = 999999
      pcall(function()
        if wd.ref.absSlipRatioTarget ~= nil then
          wd.ref.absSlipRatioTarget = 0
          absInfo = absInfo .. wd.name .. ":absOff "
        end
        if wd.ref.absEnabled ~= nil then
          wd.ref.absEnabled = false
          absInfo = absInfo .. wd.name .. ":disabled "
        end
        if wd.ref.absActive ~= nil then
          wd.ref.absActive = false
        end
      end)
    end
    local brakeInfo = ""
    for i, wd in ipairs(wheelData) do
      brakeInfo = brakeInfo .. string.format("%s=%.0fNm ", wd.name, origBrakeTorque[i])
    end
    electrics.values.tel_status = "OK_" .. #wheelData .. "_wheels|brakes:" .. brakeInfo
    print("=== TELEMETRY v3.0: " .. electrics.values.tel_status .. " ===")
    if #absInfo > 0 then
      print("=== TELEMETRY: ABS disable attempted: " .. absInfo .. " ===")
    end
    return
  end

  -- Fallback: wheels.wheels
  local ok2, err2 = pcall(function()
    if wheels and wheels.wheels then
      for id, w in pairs(wheels.wheels) do
        table.insert(wheelData, {
          name = w.name or ("w_" .. id),
          ref = w,
          method = "wheels"
        })
      end
    end
  end)

  if #wheelData > 0 then
    reorderWheelDataLogical()  -- (PPO_V2) RR,RL,FR,FL by geometry, not by jbeam order
    initialized = true
    for i, wd in ipairs(wheelData) do
      origBrakeTorque[i] = wd.ref.brakeTorque or 0
      accumWsMin[i] = 999999
    end
    electrics.values.tel_status = "OK_" .. #wheelData .. "_wheels_v3.0"
    return
  end

  local errMsg = "ERROR_no_wheels"
  if err1 then errMsg = errMsg .. "|rot:" .. tostring(err1) end
  if err2 then errMsg = errMsg .. "|whl:" .. tostring(err2) end
  electrics.values.tel_status = errMsg
end


local function applyPerWheelBrakes()
  if not perWheelMode or not initialized or #wheelData < 4 then return end
  local maxBrake = math.max(brakeCmd[1], brakeCmd[2], brakeCmd[3], brakeCmd[4])
  for i, wd in ipairs(wheelData) do
    local cmdIdx = wheelToBrakeMap[i]
    if cmdIdx and origBrakeTorque[i] and origBrakeTorque[i] > 0 then
      if maxBrake > 0.001 then
        wd.ref.brakeTorque = origBrakeTorque[i] * (brakeCmd[cmdIdx] or 0) / maxBrake
      else
        wd.ref.brakeTorque = 0
      end
    end
  end
end

-- =====================================================
-- setBrakes(fr, fl, rr, rl) — called from Python/ML
-- NOT used when PD controller is active.
-- =====================================================
local function setBrakes(fr, fl, rr, rl)
  if pdEnabled then return end  -- PD controller owns brakes
  fr = math.max(0, math.min(1, fr or 0))
  fl = math.max(0, math.min(1, fl or 0))
  rr = math.max(0, math.min(1, rr or 0))
  rl = math.max(0, math.min(1, rl or 0))

  brakeCmd = {fr, fl, rr, rl}
  perWheelMode = true

  local maxBrake = math.max(fr, fl, rr, rl)
  electrics.values.brake = maxBrake

  applyPerWheelBrakes()
end


-- releaseBrakes(): unlatch perWheelMode so BeamNG's stock brake pipeline regains control.
-- Restore wd.ref.brakeTorque to its original (max) value so max*pedal math works out.
local function releaseBrakes()
  perWheelMode = false
  brakeCmd = {0, 0, 0, 0}
  for i, wd in ipairs(wheelData) do
    if origBrakeTorque[i] then
      wd.ref.brakeTorque = origBrakeTorque[i]
    end
  end
end


local function setTargetSpeed(speed_ms)
  brakeTargetSpeed = speed_ms or 0
  brakeState = "idle"
  brakeStartPosition = nil
end


local function resetAccum()
  for i = 1, #wheelData do
    accumWsMin[i] = 999999
  end
  imuPitch = 0; imuRoll = 0  -- (PPO_V2) re-zero the gyro-integrated attitude per brake event
  brakeState = "idle"
  brakeStartPosition = nil; brakeStartTime = 0
  lastBrakeAvgGy = 0; lastBrakeAvgG = 0; lastBrakeAvgGArc = 0; lastBrakeDist = 0; lastBrakeDistArc = 0; brakeArcDist = 0; brakePrevPos = nil
  lastBrakeStartSpeed = 0; lastBrakeDuration = 0
  pollSpeedStart = -1; pollDtSum = 0; pollFrames = 0
  pollGyRollingBuf = {}; pollGyRollingIdx = 0
  pollGySmoothedMin = 999999; pollGySmoothedMax = -999999
  pollGxRollingBuf = {}; pollGxRollingIdx = 0
  pollGxSmoothedMin = 999999; pollGxSmoothedMax = -999999
  pollGxSum = 0
  pollGzRollingBuf = {}; pollGzRollingIdx = 0
  pollGzSmoothedMin = 999999; pollGzSmoothedMax = -999999
  pollGzSum = 0
  pollYawSum = 0; pollYawAbsSum = 0
  pollYawMin = 999999; pollYawMax = -999999
  physPrevSpeed = -1
end


-- =====================================================
-- PD CONTROLLER API — called by pacejka_particle_abs
-- =====================================================
local function enablePDController(config)
  config = config or {}
  pdEnabled = true
  perWheelMode = true
  pdSlipGain = config.slipGain or 90
  pdDampGain = config.dampGain or 6.0
  pdLockupSafety = config.lockupSafety ~= false
  pdLockupGLimit = config.lockupGLimit or 2.5
  pdOptimalSlip = config.initialSlip or 0.10
  pdHandoffDecelLimit = config.handoffDecelLimit or 4.905
  pdSpeedFloor = config.speedFloor or 3.0
  pdNoiseAmount = config.noiseAmount or 0.03
  pdTireRadius = config.tireRadius or 0.327
  pdMinBrake = config.minBrake or 0.15
  pdMaxBrake = config.maxBrake or 1.0
  pdCarSpeed = 0
  pdBrakeRatio = {0, 0, 0, 0}
  pdPrevSlips = {0, 0, 0, 0}
  pdPrevWheelSpeed = {0, 0, 0, 0}
  pdWasBraking = false
  print("=== TELEMETRY: PD controller enabled at 2000Hz ===")
end

local function disablePDController()
  pdEnabled = false
  print("=== TELEMETRY: PD controller disabled ===")
end

local function setOptimalSlip(slip)
  pdOptimalSlip = slip
end

-- =====================================================
-- FUSED SPEED API — called by threshold ABS module
-- =====================================================
local function setDecelFilter(value)
  fused.decelFilter = value or -0.001
end

local function getFusedSpeed()
  return fused.speed
end


local function getPDState()
  return {
    enabled = pdEnabled,
    carSpeed = pdCarSpeed,
    optimalSlip = pdOptimalSlip,
    absActive = pdAbsActive,
    handedOff = pdHandedOff,
    slips = {pdWheelSlips[1], pdWheelSlips[2], pdWheelSlips[3], pdWheelSlips[4]},
    brakeRatios = {pdBrakeRatio[1], pdBrakeRatio[2], pdBrakeRatio[3], pdBrakeRatio[4]},
    lockups = {pdLockupTriggered[1], pdLockupTriggered[2], pdLockupTriggered[3], pdLockupTriggered[4]},
  }
end


-- =====================================================
-- onPhysicsStep helpers — extracted to keep onPhysicsStep
-- under Lua's hard 60-upvalue-per-function limit (BeamNG.tech).
-- Per-tick computed values are passed as ARGUMENTS so they do
-- not count as upvalues; module-local accumulators stay at
-- module scope and are read/written here as shared upvalues.
-- =====================================================

-- Poll + per-frame accumulators (gx/gy/gz rolling windows, yaw, gfx peak/avg).
-- Extracted verbatim from onPhysicsStep; sequence of operations unchanged.
local function updatePollAccumulators(substepDecel, sensorX, sensorZ, yawRate, dtPhys, physPrevSpeed)
  -- Per-60Hz-frame accumulators for particle filter
  if substepDecel > gfxPeakGy then gfxPeakGy = substepDecel end
  gfxGySum = gfxGySum + substepDecel
  gfxGyCount = gfxGyCount + 1

  -- Poll accumulator
  pollFrames = pollFrames + 1
  pollDtSum = pollDtSum + dtPhys
  if pollSpeedStart < 0 then
    pollSpeedStart = physPrevSpeed
  end

  -- Rolling window for min/max
  pollGyRollingIdx = (pollGyRollingIdx % pollGyRollingSize) + 1
  pollGyRollingBuf[pollGyRollingIdx] = substepDecel
  if pollFrames >= pollGyRollingSize then
    local sum = 0
    for k = 1, pollGyRollingSize do
      sum = sum + (pollGyRollingBuf[k] or 0)
    end
    local smoothed = sum / pollGyRollingSize
    if smoothed < pollGySmoothedMin then pollGySmoothedMin = smoothed end
    if smoothed > pollGySmoothedMax then pollGySmoothedMax = smoothed end
  end

  -- gx rolling window (mirrors gy)
  pollGxSum = pollGxSum + sensorX
  pollGxRollingIdx = (pollGxRollingIdx % pollGyRollingSize) + 1
  pollGxRollingBuf[pollGxRollingIdx] = sensorX
  if pollFrames >= pollGyRollingSize then
    local sumX = 0
    for k = 1, pollGyRollingSize do
      sumX = sumX + (pollGxRollingBuf[k] or 0)
    end
    local smoothedX = sumX / pollGyRollingSize
    if smoothedX < pollGxSmoothedMin then pollGxSmoothedMin = smoothedX end
    if smoothedX > pollGxSmoothedMax then pollGxSmoothedMax = smoothedX end
  end

  -- gz rolling window (mirrors gy)
  pollGzSum = pollGzSum + sensorZ
  pollGzRollingIdx = (pollGzRollingIdx % pollGyRollingSize) + 1
  pollGzRollingBuf[pollGzRollingIdx] = sensorZ
  if pollFrames >= pollGyRollingSize then
    local sumZ = 0
    for k = 1, pollGyRollingSize do
      sumZ = sumZ + (pollGzRollingBuf[k] or 0)
    end
    local smoothedZ = sumZ / pollGyRollingSize
    if smoothedZ < pollGzSmoothedMin then pollGzSmoothedMin = smoothedZ end
    if smoothedZ > pollGzSmoothedMax then pollGzSmoothedMax = smoothedZ end
  end

  -- Yaw
  if yawRate < pollYawMin then pollYawMin = yawRate end
  if yawRate > pollYawMax then pollYawMax = yawRate end
  pollYawSum = pollYawSum + yawRate
  pollYawAbsSum = pollYawAbsSum + math.abs(yawRate)

  -- (PPO_V2) IMU attitude: integrate body gyro rates at 2000Hz. Same API family
  -- as the yaw channel (obj:get*AngularVelocity — see BeamNG's own
  -- tech/cosimulationCoupling.lua). Zeroed at each brake-event arm (resetAccum),
  -- so pitch/roll are "attitude change since arm" — exactly what a production
  -- gyro cluster can honestly know short-horizon (no ground-truth orientation).
  local gyroPitchRate, gyroRollRate = 0, 0
  pcall(function()
    gyroPitchRate = obj:getPitchAngularVelocity() or 0
    gyroRollRate  = obj:getRollAngularVelocity() or 0
  end)
  imuPitch = imuPitch + gyroPitchRate * dtPhys
  imuRoll  = imuRoll  + gyroRollRate * dtPhys
  electrics.values.tel_imu_pitch = imuPitch
  electrics.values.tel_imu_roll  = imuRoll
end


-- =====================================================
-- FUSED SPEED ESTIMATOR at 2000Hz — integrator-floor model
-- Consumed by threshold ABS via electrics.values.tel_fused_speed
-- Extracted verbatim from onPhysicsStep.
-- =====================================================
local function updateFusedSpeed(dtPhys)
  if initialized and #wheelData >= 4 then
    local ws = {}
    for i, wd in ipairs(wheelData) do
      ws[i] = math.abs(wd.ref.wheelSpeed or 0)
    end

    -- First-tick init: seed integrator from max wheel speed at spawn
    if not fused.initialized then
      local maxInit = 0
      for i = 1, #wheelData do
        if ws[i] > maxInit then maxInit = ws[i] end
      end
      fused.speed = maxInit
      fused.integratedSpeed = maxInit
      for i = 1, #wheelData do fused.prevWs[i] = ws[i] end
      fused.initialized = true
    end

    -- 1. Single pass over wheels:
    --    - maxWsAll: max of all wheel speeds (used for braking-mode anchor)
    --    - goodSum/goodCount: wheels passing decel filter (used for non-braking re-sync only)
    local goodSum, goodCount, maxWsAll = 0, 0, 0
    for i = 1, #wheelData do
      if ws[i] > maxWsAll then maxWsAll = ws[i] end
      local d = (ws[i] - (fused.prevWs[i] or ws[i])) / dtPhys
      if d > fused.decelFilter then
        goodSum = goodSum + ws[i]
        goodCount = goodCount + 1
      end
    end

    -- 2. Brake state + per-event counter reset
    local brakeInputLocal = input.brake or 0
    local isBraking = brakeInputLocal > 0.01
    if not isBraking then
      if fused.brakeWasHeld then
        fused.accelTicks = 0
        fused.wheelTicks = 0
      end
      fused.brakeWasHeld = false
    else
      fused.brakeWasHeld = true
    end

    -- 3. Integrate sensorY every tick (dead-reckoning floor)
    --    sensors.ffiSensors.sensorY is raw m/s² in vehicle frame.
    --    Positive when decelerating (confirmed from stock code + log evidence).
    local longAccel = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorY) or 0
    fused.integratedSpeed = math.max(0, fused.integratedSpeed - longAccel * dtPhys * fused.accSign)
    fused.lastSensorY = longAccel

    -- 4. Anchor integrator based on brake state.
    --    During braking: integrator is AUTHORITATIVE. Wheels can ONLY pull up
    --    (never down), because every wheel reads BELOW truth during braking
    --    due to slip — syncing down to max wheel bakes slip bias into the
    --    estimate, which compounds into the big drift we saw in the old log.
    --    Upward pull is still allowed so that sensor positive-bias drift
    --    gets corrected when a wheel legitimately reads higher than integrator.
    --    Downward correction of any accelerometer drift only happens at
    --    brake release (non-braking branch).
    if isBraking then
      if maxWsAll > fused.integratedSpeed then
        fused.integratedSpeed = maxWsAll  -- recovery / anti-drift upward
      end
      -- else: integrator runs pure on sensorY. Max drift per brake event ≈
      -- sensor bias × duration. For a 5s hard brake with 0.5 m/s² bias that's
      -- 2.5 m/s — much smaller than the wheel-slip bias we were absorbing.
    else
      -- Non-braking: wheels are authoritative, re-sync integrator to avg of good wheels.
      -- This kills any accelerometer drift that accumulated during the brake event.
      if goodCount > 0 then
        fused.integratedSpeed = goodSum / goodCount
      end
      -- If goodCount == 0 (rare), keep last integrator value
    end

    fused.speed = fused.integratedSpeed

    -- 5. Mode flag: are wheels tracking the integrator, or is integrator carrying alone?
    --    Tolerance: 8% of current speed or 0.5 m/s, whichever is larger.
    local tol = math.max(0.5, fused.integratedSpeed * 0.08)
    if maxWsAll >= fused.integratedSpeed - tol then
      fused.mode = 1   -- wheels healthy
    else
      fused.mode = 0   -- integrator carrying (lockup detected)
    end

    -- 6. Per-tick counters during brake events
    if isBraking then
      if fused.mode == 1 then
        fused.wheelTicks = fused.wheelTicks + 1
      else
        fused.accelTicks = fused.accelTicks + 1
      end
    end

    fused.lastGoodCount = goodCount
    for i = 1, #wheelData do fused.prevWs[i] = ws[i] end

    electrics.values.tel_fused_speed = fused.speed
    electrics.values.tel_fused_filter = fused.decelFilter
    electrics.values.tel_fused_goodcount = goodCount
    electrics.values.tel_fused_mode = fused.mode
    electrics.values.tel_fused_sensory = fused.lastSensorY
    electrics.values.tel_fused_accel_ticks = fused.accelTicks
    electrics.values.tel_fused_wheel_ticks = fused.wheelTicks
    electrics.values.tel_fused_integrated = fused.integratedSpeed
    electrics.values.tel_fused_maxws = maxWsAll
  end
end


-- =====================================================
-- PD CONTROLLER at 2000Hz (only when enabled)
-- Extracted verbatim from onPhysicsStep.
-- =====================================================
local function updatePDController(dtPhys, substepDecel, brakeInput)
  if pdEnabled and initialized and #wheelData >= 4 then
    local R = pdTireRadius

    -- Read wheel speeds directly from wheel objects (2000Hz fresh)
    local ws = {}
    for i, wd in ipairs(wheelData) do
      ws[i] = wd.ref.wheelSpeed or 0
    end

    -- Max wheel speed in m/s (floor for speed estimate)
    local maxWheelMs = 0
    for i = 1, #wheelData do
      local ms = math.abs(ws[i])  -- wheelSpeed is already m/s in BeamNG
      if ms > maxWheelMs then maxWheelMs = ms end
    end

    -- Brake start detection
    local brakingJustStarted = brakeInput > 0 and not pdWasBraking
    pdWasBraking = brakeInput > 0

    -- Speed estimation at 2000Hz (uses physics-rate decel, much more accurate)
    if brakingJustStarted then
      pdCarSpeed = maxWheelMs
      for i = 1, #wheelData do
        pdBrakeRatio[i] = brakeInput
      end
    end

    if brakeInput > 0 then
      -- Use substepDecel directly (2000Hz kinematic, no electrics delay)
      if substepDecel > 0 then
        pdCarSpeed = pdCarSpeed - substepDecel * dtPhys
      end
      pdCarSpeed = math.max(pdCarSpeed, maxWheelMs)
    else
      pdCarSpeed = maxWheelMs
    end

    -- ABS always active when braking, no speed-based handoff
    pdAbsActive = brakeInput > 0
    pdHandedOff = false

    -- Slip ratios — no speed gate, calculate always
    for w = 1, #wheelData do
      if pdCarSpeed > 0.01 then
        pdWheelSlips[w] = math.max(0, math.min((pdCarSpeed - math.abs(ws[w])) / pdCarSpeed, 1.0))
      else
        pdWheelSlips[w] = 0
      end
    end

    -- PD controller per wheel
    local cmds = {0, 0, 0, 0}  -- FR, FL, RR, RL

    for i = 1, #wheelData do
      pdLockupTriggered[i] = false

      if pdAbsActive and brakeInput > 0 then
        local slipError = pdOptimalSlip - pdWheelSlips[i]
        local slipRate = (pdWheelSlips[i] - (pdPrevSlips[i] or 0)) / dtPhys

        -- dt-scaled adjustment: gains are per-second, dtPhys converts to per-step
        local adjustment = (slipError * pdSlipGain - slipRate * pdDampGain) * dtPhys
        pdBrakeRatio[i] = math.max(pdMinBrake, math.min(pdMaxBrake, (pdBrakeRatio[i] or 0) + adjustment))

        -- Lockup safety
        if pdLockupSafety then
          local prev = pdPrevWheelSpeed[i] or ws[i]
          local wheelDecelG = (prev - ws[i]) / (dtPhys * 9.81)  -- wheelSpeed is already m/s
          if wheelDecelG > pdLockupGLimit then
            pdBrakeRatio[i] = pdMinBrake
            pdLockupTriggered[i] = true
          end
        end

        -- Noise for filter diversity
        local noise = (math.random() - 0.5) * pdNoiseAmount
        cmds[wheelToBrakeMap[i]] = math.max(pdMinBrake, math.min(pdMaxBrake, pdBrakeRatio[i] + noise))

      else
        -- Not braking
        pdBrakeRatio[i] = 0
        cmds[wheelToBrakeMap[i]] = 0
      end

      pdPrevSlips[i] = pdWheelSlips[i] or 0
      pdPrevWheelSpeed[i] = ws[i]
    end

    -- Apply brake commands
    brakeCmd = cmds
    local maxBrake = math.max(cmds[1], cmds[2], cmds[3], cmds[4])
    electrics.values.brake = maxBrake
    -- applyPerWheelBrakes() will be called next physics step (already at top)
    -- but also apply immediately for this step
    applyPerWheelBrakes()
  end
end


-- =====================================================
-- onPhysicsStep(dtPhys) — ~2000Hz
-- =====================================================
-- =====================================================
-- (PPO_V2) ARMED 2kHz BRAKE SLAM — exact-speed brake onset.
-- Python arms a target via armBrakeSlam(ms). Every physics tick (0.5ms) the check
-- below compares the TRUE speed (instSpeed = obj:getVelocity():length() — the SAME
-- signal the standard brake metric uses) and latches input.brake=1 from the exact
-- tick of the crossing. Replaces the Python coast loop's ~50Hz wall-clock poll of
-- stale electrics (onset could land tenths of a mph late). The latch re-asserts
-- every tick until disarmBrakeSlam() so the pedal cannot drop between writes.
-- =====================================================
local slamArmTarget = nil
local slamFired = false

local function armBrakeSlam(target_ms)
  slamArmTarget = target_ms
  slamFired = false
  electrics.values.tel_slam_armed = 1
  electrics.values.tel_slam_fired = 0
  electrics.values.tel_slam_fire_speed = -1
end

local function disarmBrakeSlam()
  slamArmTarget = nil
  slamFired = false
  electrics.values.tel_slam_armed = 0
  electrics.values.tel_slam_fired = 0
end

-- =====================================================
-- TIRE GRIP CONTROL
-- Changes the TIRE's friction, not the ground surface: the same mechanism
-- BeamNG's own tire-damage code uses (beamstate.lua:549-557) --
-- obj:setNodeFrictionSlidingCoefs on each wheel's treadNodes. Applied evenly
-- to every wheel (this is "different tires", not split-mu).
--
-- The multiplier is ALWAYS applied against v.data.nodes -- the untouched jbeam
-- values -- never against the current coefficients, so repeated calls cannot
-- compound and restoring is exactly applyGripMultiplier(1.0).
--
-- Timing: the change is armed and fires at brake onset, so the acceleration
-- and coast-down approach always happen at stock grip (a low-grip approach
-- would spin the wheels and never reach the target speed) and only the
-- braking phase sees the new value.
-- =====================================================
local gripMultCurrent = 1.0
local gripPendingMult = nil
local gripLeadSeconds = 0
local gripFired = false

local function applyGripMultiplier(mult)
  mult = mult or 1.0
  local n = 0
  if wheels and wheels.wheels and v and v.data and v.data.nodes then
    for _, wheel in pairs(wheels.wheels) do
      if wheel.treadNodes then
        for _, nodecid in pairs(wheel.treadNodes) do
          local nd = v.data.nodes[nodecid]
          if nd and nd.frictionCoef then
            obj:setNodeFrictionSlidingCoefs(
              nodecid,
              nd.frictionCoef * mult,
              (nd.slidingFrictionCoef or nd.frictionCoef) * mult)
            n = n + 1
          end
        end
      end
    end
  end
  gripMultCurrent = mult
  electrics.values.tel_grip_mult = mult
  electrics.values.tel_grip_nodes = n
  return n
end

-- Immediate change (used by the reference runner, which has no brake-onset
-- arming step of its own to hang this off).
local function setGripMultiplier(mult)
  gripPendingMult = nil
  gripFired = false
  electrics.values.tel_grip_armed = 0
  return applyGripMultiplier(mult)
end

-- Deferred change: fires at the brake onset (same physics tick the slam
-- latches), or leadSeconds earlier if asked. The lead is predicted from the
-- current deceleration, so it is approximate; leadSeconds = 0 is exact.
local function armGripChange(mult, leadSeconds)
  gripPendingMult = mult
  gripLeadSeconds = leadSeconds or 0
  gripFired = false
  electrics.values.tel_grip_armed = 1
  electrics.values.tel_grip_fired = 0
end

local function restoreGrip()
  gripPendingMult = nil
  gripFired = false
  electrics.values.tel_grip_armed = 0
  electrics.values.tel_grip_fired = 0
  return applyGripMultiplier(1.0)
end

local function updateArmedGrip()
  if gripPendingMult == nil or gripFired or slamArmTarget == nil then return end
  local trigger
  if gripLeadSeconds <= 0 then
    trigger = instSpeed <= slamArmTarget          -- same tick as the slam
  else
    local decel = instGy                          -- current decel, m/s^2
    if decel > 0.05 then
      trigger = ((instSpeed - slamArmTarget) / decel) <= gripLeadSeconds
    else
      trigger = instSpeed <= slamArmTarget        -- coasting flat: no useful lead
    end
  end
  if trigger then
    local n = applyGripMultiplier(gripPendingMult)
    gripFired = true
    electrics.values.tel_grip_fired = 1
    electrics.values.tel_grip_fire_speed = instSpeed
    print(string.format(
      "=== TELEMETRY: tire grip x%.3f applied on %d nodes at %.4f m/s ===",
      gripPendingMult, n, instSpeed))
  end
end


local function updateArmedSlam()
  if slamArmTarget == nil then return end
  updateArmedGrip()
  if not slamFired and instSpeed <= slamArmTarget then
    slamFired = true
    electrics.values.tel_slam_fired = 1
    electrics.values.tel_slam_fire_speed = instSpeed
    print(string.format(
      "=== TELEMETRY: 2kHz brake slam FIRED at %.4f m/s (target %.4f m/s) ===",
      instSpeed, slamArmTarget))
  end
  if slamFired then
    input.brake = 1
  end
end


local function onPhysicsStep(dtPhys)
  if not initialized then return end
  if dtPhys <= 0 then return end

  -- Re-apply per-wheel brakes at physics rate (2000Hz)
  applyPerWheelBrakes()

  local vel = obj:getVelocity()
  if not vel then return end

  local vx = vel.x or 0
  local vy = vel.y or 0
  local vz = vel.z or 0
  local speed = math.sqrt(vx*vx + vy*vy + vz*vz)

  if physPrevSpeed < 0 then
    physPrevSpeed = speed
    instSpeed = speed
    return
  end

  local substepDecel = (physPrevSpeed - speed) / dtPhys

  local yawRate = 0
  pcall(function()
    yawRate = obj:getYawAngularVelocity() or 0
  end)

  instGy = substepDecel
  instYaw = yawRate
  instSpeed = speed

  -- Sample gx/gz from IMU at every physics step (2000Hz)
  local sensorX = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorX) or 0
  local sensorZ = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorZ) or 0
  instGx = sensorX
  instGz = sensorZ

  -- Poll + per-frame accumulators (extracted to helper for upvalue budget)
  updatePollAccumulators(substepDecel, sensorX, sensorZ, yawRate, dtPhys, physPrevSpeed)

  -- =====================================================
  -- FUSED SPEED ESTIMATOR at 2000Hz — integrator-floor model
  -- Consumed by threshold ABS via electrics.values.tel_fused_speed
  -- (extracted to helper for upvalue budget)
  -- =====================================================
  updateFusedSpeed(dtPhys)

  -- (PPO_V2) armed 2kHz slam — MUST run before the brake SM reads input.brake so
  -- the SM arms on the very tick the pedal latches at the target crossing.
  updateArmedSlam()

  -- =====================================================
  -- MODE 2: Brake event state machine
  -- =====================================================
  local brakeInput = input.brake or 0
  -- v3.3: use physics-rate instSpeed (computed every onPhysicsStep from obj:getVelocity())
  -- rather than electrics.values.airspeed which lags at speed_factor > 1.
  local airspeed = instSpeed

  -- v3.2: threshold lowered from 0.01 to 0.001 to play nice with Python's
  -- 0.01 brake floor. Now only resets the state machine if brakes are
  -- effectively fully released (sub-1%), preventing measurement loss when
  -- the model briefly modulates near zero during real-ABS-style control.
  if brakeInput < 0.001 then
    brakeState = "idle"
  end

  if brakeState == "idle" then
    if brakeInput > 0.05 and airspeed > brakeTargetSpeed then
      brakeState = "waiting"
    end
  elseif brakeState == "waiting" then
    if airspeed <= brakeTargetSpeed then
      brakeStartPosition = obj:getPosition()
      brakeStartTime = simTime
      brakeArcDist = 0
      brakePrevPos = brakeStartPosition
      brakeState = "measuring"
    end
  elseif brakeState == "measuring" then
    local curPos = obj:getPosition()
    if brakePrevPos then
      brakeArcDist = brakeArcDist + (curPos - brakePrevPos):length()
    end
    brakePrevPos = curPos
    if airspeed <= 1 then
      local endPosition = curPos
      local distance = (brakeStartPosition - endPosition):length()
      local avgDeceleration = -(airspeed * airspeed - brakeTargetSpeed * brakeTargetSpeed) / (2 * distance)
      local gravity = powertrain.currentGravity
      -- Arc distance is always >= chord; guard the same div-by-zero edge the
      -- chord path already assumes can't happen (a real stop always has an
      -- arc length >= its chord).
      local arcDistSafe = math.max(brakeArcDist, distance)
      local avgDecelerationArc = -(airspeed * airspeed - brakeTargetSpeed * brakeTargetSpeed) / (2 * arcDistSafe)
      print(string.format("=== BRAKE EVENT DONE: dist=%.2fm avgDecel=%.2f gravity=%.4f avgG=%.4f arcDist=%.2fm avgGArc=%.4f ===",
        distance, avgDeceleration, gravity, avgDeceleration / -gravity, arcDistSafe, avgDecelerationArc / -gravity))
      lastBrakeAvgGy = avgDeceleration
      lastBrakeAvgG = avgDeceleration / -gravity
      lastBrakeAvgGArc = avgDecelerationArc / -gravity
      lastBrakeDist = distance
      lastBrakeDistArc = arcDistSafe
      lastBrakeStartSpeed = brakeTargetSpeed
      lastBrakeDuration = simTime - brakeStartTime
      brakeStartPosition = nil
      brakePrevPos = nil
      brakeState = "idle"
    end
  end

  -- =====================================================
  -- PD CONTROLLER at 2000Hz (only when enabled)
  -- (extracted to helper for upvalue budget)
  -- =====================================================
  updatePDController(dtPhys, substepDecel, brakeInput)

  physPrevSpeed = speed
end


-- =====================================================
-- updateGFX(dtSim) — ~60Hz
-- =====================================================
local function onGraphicsStep(dtSim)
  if not initialized then
    tryInitWheels()
    if not initialized then return end
  end

  heartbeat = heartbeat + 1
  simTime = simTime + dtSim
  electrics.values.tel_heartbeat = heartbeat
  electrics.values.tel_time = simTime

  -- === WHEEL SPEEDS ===
  electrics.values.tel_wheel_count = #wheelData
  for i, wd in ipairs(wheelData) do
    local speed = 0
    pcall(function()
      speed = wd.ref.wheelSpeed or 0
    end)
    electrics.values["tel_ws_" .. i] = speed
    electrics.values["tel_wname_" .. i] = wd.name

    local absSpeed = math.abs(speed)
    if absSpeed < (accumWsMin[i] or 999999) then
      accumWsMin[i] = absSpeed
    end
    electrics.values["tel_ws_min_" .. i] = accumWsMin[i]
  end

  -- === INSTANTANEOUS TO ELECTRICS ===
  electrics.values.tel_gy_inst = instGy
  electrics.values.tel_gx_inst = instGx
  electrics.values.tel_yaw_rate_inst = instYaw

  -- === PER-60Hz-FRAME PEAK AND AVG (for particle filter) ===
  electrics.values.tel_gy_peak = gfxPeakGy
  electrics.values.tel_gy_avg60 = gfxGyCount > 0 and (gfxGySum / gfxGyCount) or 0
  gfxPeakGy = 0
  gfxGySum = 0
  gfxGyCount = 0

  -- === BRAKE MODE ===
  electrics.values.tel_brk_mode = pdEnabled and "pd2000hz" or (perWheelMode and "perwheel" or "normal")

  -- === PER-WHEEL BRAKE: write command values to electrics for readback ===
  if perWheelMode then
    electrics.values.tel_brk_fr = brakeCmd[1]
    electrics.values.tel_brk_fl = brakeCmd[2]
    electrics.values.tel_brk_rr = brakeCmd[3]
    electrics.values.tel_brk_rl = brakeCmd[4]
  end

  -- === ACTUAL BRAKE TORQUE (Nm) ===
  if initialized and #wheelData >= 4 then
    electrics.values.tel_brk_actual_fr = wheelData[3].ref.desiredBrakingTorque or 0
    electrics.values.tel_brk_actual_fl = wheelData[4].ref.desiredBrakingTorque or 0
    electrics.values.tel_brk_actual_rr = wheelData[1].ref.desiredBrakingTorque or 0
    electrics.values.tel_brk_actual_rl = wheelData[2].ref.desiredBrakingTorque or 0
  end

  -- === PD CONTROLLER STATE TO ELECTRICS ===
  if pdEnabled then
    electrics.values.pd_carSpeed = pdCarSpeed
    electrics.values.pd_absActive = pdAbsActive and 1 or 0
    electrics.values.pd_handedOff = pdHandedOff and 1 or 0
    electrics.values.pd_optimalSlip = pdOptimalSlip
    electrics.values.pd_slip_1 = pdWheelSlips[1] or 0
    electrics.values.pd_slip_2 = pdWheelSlips[2] or 0
    electrics.values.pd_slip_3 = pdWheelSlips[3] or 0
    electrics.values.pd_slip_4 = pdWheelSlips[4] or 0
    electrics.values.pd_ratio_1 = pdBrakeRatio[1] or 0
    electrics.values.pd_ratio_2 = pdBrakeRatio[2] or 0
    electrics.values.pd_ratio_3 = pdBrakeRatio[3] or 0
    electrics.values.pd_ratio_4 = pdBrakeRatio[4] or 0
  end

  -- MODE 2 results to electrics
  electrics.values.tel_brake_active = (brakeState == "measuring") and 1 or 0
  electrics.values.tel_last_brake_avg_gy = lastBrakeAvgGy
  electrics.values.tel_last_brake_avg_g = lastBrakeAvgG
  electrics.values.tel_last_brake_avg_g_arc = lastBrakeAvgGArc
  electrics.values.tel_last_brake_dist = lastBrakeDist
  electrics.values.tel_last_brake_dist_arc = lastBrakeDistArc
  electrics.values.tel_last_brake_duration = lastBrakeDuration
end


-- =====================================================
-- buildSensorData() — all data Python needs
-- =====================================================
local function buildSensorData()
  local pollAvgGy = 0
  if pollDtSum > 0 and pollSpeedStart >= 0 then
    pollAvgGy = (pollSpeedStart - instSpeed) / pollDtSum
  end

  local pn = math.max(pollFrames, 1)

  local result = {
    airspeed = electrics.values.airspeed or 0,
    inst_speed = instSpeed,   -- physics-tick-accurate speed (m/s), updated every onPhysicsStep
    ws1 = electrics.values["tel_ws_1"] or 0,
    ws2 = electrics.values["tel_ws_2"] or 0,
    ws3 = electrics.values["tel_ws_3"] or 0,
    ws4 = electrics.values["tel_ws_4"] or 0,
    gy_avg = pollAvgGy,
    gy_min = pollGySmoothedMin ~= 999999 and pollGySmoothedMin or 0,
    gy_max = pollGySmoothedMax ~= -999999 and pollGySmoothedMax or 0,
    gy_inst = instGy,
    gx_inst = instGx,
    yaw_avg = pollYawSum / pn,
    yaw_abs_avg = pollYawAbsSum / pn,
    yaw_min = pollYawMin ~= 999999 and pollYawMin or 0,
    yaw_max = pollYawMax ~= -999999 and pollYawMax or 0,
    yaw_inst = instYaw,
    poll_frames = pollFrames,
    last_brake_avg_gy = lastBrakeAvgGy,
    last_brake_avg_g = lastBrakeAvgG,
    last_brake_avg_g_arc = lastBrakeAvgGArc,
    last_brake_dist = lastBrakeDist,
    last_brake_dist_arc = lastBrakeDistArc,
    last_brake_start_speed = lastBrakeStartSpeed,
    last_brake_duration = lastBrakeDuration,
    brake_active = (brakeState == "measuring") and 1 or 0,
    tel_status = electrics.values.tel_status or "UNKNOWN",
    debug_gravity = powertrain.currentGravity or -9999,
    brk_actual_rr = (initialized and #wheelData >= 1) and (wheelData[1].ref.desiredBrakingTorque or 0) or 0,
    brk_actual_rl = (initialized and #wheelData >= 2) and (wheelData[2].ref.desiredBrakingTorque or 0) or 0,
    brk_actual_fr = (initialized and #wheelData >= 3) and (wheelData[3].ref.desiredBrakingTorque or 0) or 0,
    brk_actual_fl = (initialized and #wheelData >= 4) and (wheelData[4].ref.desiredBrakingTorque or 0) or 0,
    heading = obj:getDirection() or 0,

    -- Real-car sensors added 2026-05-02 for slip-removed PPO retrain (Rule 1 compliance)
    rpm = electrics.values.rpm or 0,
    gear = electrics.values.gearIndex or electrics.values.gear or 0,
    -- (PPO_V2) measured steering-wheel angle (deg) — SAME channel Dynamic_ABS reads
    -- (electrics.values.steering); the old normalized driver input stays as debug.
    steering = electrics.values.steering or 0,
    steering_input = electrics.values.steering_input or 0,
    input_brake = input.brake or 0,
    input_throttle = input.throttle or 0,
    gz_inst = instGz,
    -- Windowed gx/gz at 2000Hz IMU rate (for SpeedLSTM data export)
    gx_avg = (pollFrames > 0) and (pollGxSum / pollFrames) or 0,
    gx_min = pollGxSmoothedMin ~= 999999 and pollGxSmoothedMin or 0,
    gx_max = pollGxSmoothedMax ~= -999999 and pollGxSmoothedMax or 0,
    gz_avg = (pollFrames > 0) and (pollGzSum / pollFrames) or 0,
    gz_min = pollGzSmoothedMin ~= 999999 and pollGzSmoothedMin or 0,
    gz_max = pollGzSmoothedMax ~= -999999 and pollGzSmoothedMax or 0,
    -- pitch/roll: ground-truth orientation (gyro-estimate experiment REVERTED
    -- 2026-07-11 per Blake — "for now". The gyro-integrated values stay published
    -- as pitch_gyro/roll_gyro DEBUG channels only, never fed to the model).
    pitch = (function()
      local d = obj:getDirectionVector()
      local horiz = math.sqrt(d.x * d.x + d.y * d.y)
      return math.atan2(d.z, horiz)
    end)(),
    roll = (function()
      local d = obj:getDirectionVector()
      local u = obj:getDirectionVectorUp()
      local rightZ = d.x * u.y - d.y * u.x
      return math.atan2(rightZ, u.z)
    end)(),
    pitch_gyro = electrics.values.tel_imu_pitch or 0,
    roll_gyro = electrics.values.tel_imu_roll or 0,
  }

  -- Reset poll accumulator (consumed)
  pollSpeedStart = -1; pollDtSum = 0; pollFrames = 0
  pollGyRollingBuf = {}; pollGyRollingIdx = 0
  pollGySmoothedMin = 999999; pollGySmoothedMax = -999999
  pollGxRollingBuf = {}; pollGxRollingIdx = 0
  pollGxSmoothedMin = 999999; pollGxSmoothedMax = -999999
  pollGxSum = 0
  pollGzRollingBuf = {}; pollGzRollingIdx = 0
  pollGzSmoothedMin = 999999; pollGzSmoothedMax = -999999
  pollGzSum = 0
  pollYawSum = 0; pollYawAbsSum = 0
  pollYawMin = 999999; pollYawMax = -999999

  return result
end


local function exchangeData(fr, fl, rr, rl)
  setBrakes(fr, fl, rr, rl)
  electrics.values.throttle_input = 0
  electrics.values.steering_input = 0
  return jsonEncode(buildSensorData())
end

local function readAll()
  return jsonEncode(buildSensorData())
end

local function nukeControllers()
  pcall(function()
    if controller and controller.controllerCache then
      for cname, ctrl in pairs(controller.controllerCache) do
        if ctrl.updateWheelsIntermediate then
          ctrl.updateWheelsIntermediate = function() end
          print("=== TELEMETRY: Nuked " .. tostring(cname) .. " ===")
        end
      end
    end
  end)
end

local function onExtensionLoaded()
  print("=== TELEMETRY BRIDGE v3.3 LOADED (physics-rate inst_speed for speed_factor support) ===")
  electrics.values.tel_status = "LOADING_v3.2"
  electrics.values.tel_heartbeat = 0
  enablePhysicsStepHook()
  print("=== TELEMETRY: Physics step hook enabled ===")
end

local function onReset()
  print("=== TELEMETRY BRIDGE v3.1 RESET ===")
  initialized = false
  heartbeat = 0; simTime = 0
  perWheelMode = false
  brakeCmd = {0, 0, 0, 0}
  origBrakeTorque = {}; accumWsMin = {}

  physPrevSpeed = -1
  instGy = 0; instGx = 0; instGz = 0; instYaw = 0; instSpeed = 0
  gfxPeakGy = 0; gfxGySum = 0; gfxGyCount = 0

  pollSpeedStart = -1; pollDtSum = 0; pollFrames = 0
  pollGyRollingBuf = {}; pollGyRollingIdx = 0
  pollGySmoothedMin = 999999; pollGySmoothedMax = -999999
  pollGxRollingBuf = {}; pollGxRollingIdx = 0
  pollGxSmoothedMin = 999999; pollGxSmoothedMax = -999999
  pollGxSum = 0
  pollGzRollingBuf = {}; pollGzRollingIdx = 0
  pollGzSmoothedMin = 999999; pollGzSmoothedMax = -999999
  pollGzSum = 0
  pollYawSum = 0; pollYawAbsSum = 0
  pollYawMin = 999999; pollYawMax = -999999

  brakeState = "idle"
  brakeStartPosition = nil; brakeStartTime = 0
  lastBrakeAvgGy = 0; lastBrakeAvgG = 0; lastBrakeAvgGArc = 0; lastBrakeDist = 0; lastBrakeDistArc = 0; brakeArcDist = 0; brakePrevPos = nil
  lastBrakeStartSpeed = 0; lastBrakeDuration = 0

  -- Reset PD state but preserve enabled/config
  pdCarSpeed = 0
  pdBrakeRatio = {0, 0, 0, 0}
  pdPrevSlips = {0, 0, 0, 0}
  pdPrevWheelSpeed = {0, 0, 0, 0}
  pdWasBraking = false
  pdAbsActive = false; pdHandedOff = false
  pdWheelSlips = {0, 0, 0, 0}
  pdLockupTriggered = {false, false, false, false}

  -- Reset fused speed estimator (packed table, single upvalue)
  fused.decelFilter = -0.001
  fused.speed = 0
  fused.integratedSpeed = 0
  fused.prevWs = {0, 0, 0, 0}
  fused.initialized = false
  fused.lastGoodCount = 0
  fused.accSign = 1
  fused.mode = 1
  fused.lastSensorY = 0
  fused.accelTicks = 0
  fused.wheelTicks = 0
  fused.brakeWasHeld = false
end

M.onExtensionLoaded = onExtensionLoaded
M.updateGFX = onGraphicsStep
M.onPhysicsStep = onPhysicsStep
M.onReset = onReset
M.setBrakes = setBrakes
M.releaseBrakes = releaseBrakes
M.setTargetSpeed = setTargetSpeed
M.setGripMultiplier = setGripMultiplier   -- tire grip: immediate
M.armGripChange     = armGripChange       -- tire grip: at brake onset
M.restoreGrip       = restoreGrip         -- tire grip: back to stock
M.armBrakeSlam = armBrakeSlam       -- (PPO_V2) 2kHz-exact brake onset
M.disarmBrakeSlam = disarmBrakeSlam
M.resetAccum = resetAccum
M.nukeControllers = nukeControllers
M.exchangeData = exchangeData
M.readAll = readAll
M.buildSensorData = buildSensorData
-- PD controller API
M.enablePDController = enablePDController
M.disablePDController = disablePDController
M.setOptimalSlip = setOptimalSlip
M.getPDState = getPDState
-- Fused speed API (threshold ABS)
M.setDecelFilter = setDecelFilter
M.getFusedSpeed = getFusedSpeed

return M
