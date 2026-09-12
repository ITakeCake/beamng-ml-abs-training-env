local AUTO_VERSION = "V3.02_f4c85b02" -- Updated automatically by Python watcher
-- absTelemetryLogger.lua
-- Unified logger for ABS telemetry (Stock and Custom)

local M = {}

-- ABS_CodeKey: content hash of the controller code actually loaded with this
-- vehicle. AUTO_VERSION above is display-only (it can lag behind code edits
-- until a snapshot runs); the codeKey is the authoritative run<->version
-- identity the Python side links against. Computed GE-side at vehicle load
-- (vehicle Lua file IO is sandboxed), pushed back in via setCodeKey. The GE
-- side also dumps the code to telemetry/abs_code/<key>.lua the first time a
-- key is seen, so the exact code of every run survives later edits.
local codeKey = nil

local CONTROLLER_REL_PATH = "mods/unpacked/Dynamic_ABS/lua/vehicle/controller/Dynamic_ABS.lua"

local function requestCodeKey()
  local vid = tostring(objectId or 0)
  -- djb2 (h*33+b) and sdbm (h*65599+b) mod 2^32: products stay < 2^49, exact
  -- in Lua doubles, and reproduced bit-identically in version_tracker.code_key.
  local ge = [[
    pcall(function()
      local f = io.open("]] .. CONTROLLER_REL_PATH .. [[", "rb")
      if not f then return end
      local data = f:read("*a")
      f:close()
      if not data or #data == 0 then return end
      local d, s = 5381, 0
      for i = 1, #data do
        local b = string.byte(data, i)
        d = (d * 33 + b) % 4294967296
        s = (s * 65599 + b) % 4294967296
      end
      local key = string.format("%08x-%08x-%d", d, s, #data)
      pcall(function()
        if FS and not FS:directoryExists("telemetry/abs_code") then
          FS:directoryCreate("telemetry/abs_code")
        end
        local sidecarPath = "telemetry/abs_code/" .. key .. ".lua"
        local probe = io.open(sidecarPath, "rb")
        if probe then
          probe:close()
        else
          local out = io.open(sidecarPath, "wb")
          if out then out:write(data) out:close() end
        end
      end)
      local veh = be:getObjectByID(]] .. vid .. [[)
      if veh then
        veh:queueLuaCommand("extensions.absTelemetryLogger.setCodeKey('" .. key .. "')")
      end
    end)
  ]]
  pcall(function() obj:queueGameEngineLua(ge) end)
end

local function setCodeKey(key)
  codeKey = tostring(key or "")
  log('I', 'absTelemetry', 'ABS code key for this session: ' .. codeKey)
end

local isLogging = false
local logBuffer = {}
local runStartTime = 0
local frameCount = 0
local runElapsed = 0            -- seconds captured this run, accumulated at 2kHz from dtPhys (task #9 >0.5s gate)
local currentBrakeTestRunID = nil -- Brake Test's currentRunID for the run currently being captured (task #8/#9 correlation key)

local cachedControllerName = nil
local isCustomABS = false

local wheelToBrakeMap = {1, 2, 3, 4}

local function buildWheelMaps()
  wheelToBrakeMap = {}
  local ok, err = pcall(function()
    local cornerWheelData = {"FR", "FL", "RR", "RL"}
    local cornerWheels = {}
    for _, wheelName in pairs(cornerWheelData) do
      cornerWheels[wheelName] = true
    end

    local avgWheelPos = vec3(0, 0, 0)
    for _, wheel in pairs(wheels.wheels) do
      if cornerWheels[wheel.name] then
        local wheelNodePos = v.data.nodes[wheel.node1].pos
        avgWheelPos = avgWheelPos + wheelNodePos
      end
    end
    avgWheelPos = avgWheelPos / #wheels.wheels

    local refNodes = v.data.refNodes[0]
    local vectorForward = vec3(v.data.nodes[refNodes.ref].pos) - vec3(v.data.nodes[refNodes.back].pos)
    local vectorUp      = vec3(v.data.nodes[refNodes.up].pos)  - vec3(v.data.nodes[refNodes.ref].pos)
    local vectorRight   = vectorForward:cross(vectorUp)

    local foundWheelsCount = 0
    for _, wheel in pairs(wheels.wheels) do
      if cornerWheels[wheel.name] then
        local wheelNodePos = vec3(v.data.nodes[wheel.node1].pos)
        local wheelVector  = wheelNodePos - avgWheelPos
        local dotForward   = vectorForward:dot(wheelVector)
        local dotRight     = vectorRight:dot(wheelVector)

        local rotIdx = wheels.wheelRotatorIDs[wheel.name]
        if rotIdx == nil then error("wheelRotatorIDs missing for '" .. wheel.name .. "'") end
        local slot = rotIdx + 1

        if dotRight >= 0 then
          if dotForward >= 0 then
            wheelToBrakeMap[3] = slot   -- FR
          else
            wheelToBrakeMap[1] = slot   -- RR
          end
        else
          if dotForward >= 0 then
            wheelToBrakeMap[4] = slot   -- FL
          else
            wheelToBrakeMap[2] = slot   -- RL
          end
        end
        foundWheelsCount = foundWheelsCount + 1
      end
    end

    if foundWheelsCount ~= 4 or not (wheelToBrakeMap[1] and wheelToBrakeMap[2] and wheelToBrakeMap[3] and wheelToBrakeMap[4]) then
      error("Could not classify all 4 corner wheels")
    end
  end)

  if not ok then
    local f = io.open("build_wheel_error.txt", "w")
    if f then
      f:write("Error in buildWheelMaps: " .. tostring(err))
      f:close()
    end
    for i = 1, 4 do wheelToBrakeMap[i] = i end
  end
end

local pushedTelemetry = nil

local function setCustomTelemetry(name, data)
  cachedControllerName = name
  pushedTelemetry = data
  isCustomABS = true
end

-- Custom ABS controllers this logger knows how to identify. Probed by name via
-- controller.getController because controller.getControllers() does NOT exist
-- in this BeamNG build (confirmed in beamng.log: "controller or getControllers
-- is nil" on every load) — relying on it silently mislabels every pull-detected
-- controller (e.g. Blake_ABS_1F) as "StockABS".
local KNOWN_ABS_CONTROLLERS = {"DynamicABS_60hz", "DynamicABS_100hz", "DynamicABS_400hz", "DynamicABS_282", "Dynamic_ABS2", "Dynamic_ABS", "Dynamic_ABSF", "Blake_ABS_1F", "WheelOrder_ABS"}

local function getActiveABSName()
  if not controller then return nil end
  if controller.getControllers then  -- generic path, when the build has it
    local ctrls = controller.getControllers()
    if type(ctrls) == "table" then
      for k, v in pairs(ctrls) do
        if type(v) == "table" and type(v.getTelemetry) == "function" then
          return k
        end
      end
    end
  end
  if controller.getController then
    for _, name in ipairs(KNOWN_ABS_CONTROLLERS) do
      if controller.getController(name) then
        return name
      end
    end
    if controller.getController("Blake_OldABS") then
      return "ABS_1FEX"
    end
  end
  return nil
end

local function getCustomABSState(ctrlName)
  if controller and controller.getController then
    local dAbs = controller.getController(ctrlName)
    if dAbs and dAbs.getTelemetry then
      -- pcall-guarded: a buggy controller getTelemetry() must NEVER crash the vehicle
      -- (a broken telemetry read just means no extended columns for that run). This is
      -- exactly how Blake_ABS_1F's getTelemetry FFI bug took the whole vehicle down.
      local ok, t = pcall(dAbs.getTelemetry)
      -- Only usable for the extended CSV columns if it has the expected shape;
      -- the header/row field counts must stay in sync (indexing a missing
      -- slipTargets/pidOut table would error every physics step).
      if ok and type(t) == "table" and type(t.slipTargets) == "table" and type(t.pidOut) == "table" then
        return t
      end
    end
  end
  return nil
end

local function csvNum(pattern, value)
  return string.format(pattern, tonumber(value) or 0)
end

local function flushLogToDisk()
  if #logBuffer == 0 then return end
  
  local dateStr = os.date("%Y-%m-%d_%H-%M-%S")
  local controllerName = cachedControllerName or getActiveABSName() or "StockABS"
  local customState = nil
  if isCustomABS then customState = pushedTelemetry or getCustomABSState(cachedControllerName)
  elseif cachedControllerName then customState = getCustomABSState(controllerName) end

  local label = type(controllerName) == "string" and controllerName or "StockABS"
  
  -- We'll write to the mod folder so we don't hit absolute path sandbox errors, 
  -- but we can write to the user folder. 
  local dir = label == "WheelOrder_ABS" and "telemetry/calibration/" or "telemetry/"
  local filename = dir .. "DynamicABS_Run_" .. dateStr .. "_" .. label .. ".csv"
  
  -- Extract metadata
  local metadataStr = "# VEHICLE_METADATA: Car=unknown | Config=unknown | Parts=unknown"
  pcall(function()
    local car = "unknown"
    if v and v.vehicleDirectory then car = string.gsub(v.vehicleDirectory, "vehicles/", "") end
    if car:sub(-1) == "/" then car = car:sub(1, -2) end
    
    local configStr = "unknown"
    local activeParts = {}
    if v and v.config then
      if v.config.partConfigFilename then configStr = v.config.partConfigFilename end
      if v.config.parts then
        for k, partName in pairs(v.config.parts) do
          if type(partName) == "string" and (string.find(string.lower(partName), "pad") or string.find(string.lower(partName), "brake") or string.find(string.lower(partName), "tire") or string.find(string.lower(partName), "wheel")) then
            table.insert(activeParts, partName)
          end
        end
      end
    end
    metadataStr = string.format("# VEHICLE_METADATA: Car=%s | Config=%s | Parts=%s | ABS_Version=%s", car, configStr, table.concat(activeParts, ", "), AUTO_VERSION)
  end)

  -- Correlation key (task #8/#9): stamp the Brake Test Run ID that gated this capture
  -- so Python (run_data.py) can join this telemetry file back to the Brake Test result
  -- row that produced it. Appended unconditionally, even if the pcall above failed and
  -- metadataStr fell back to its "unknown" default.
  metadataStr = metadataStr .. " | BrakeTestRunID=" .. tostring(currentBrakeTestRunID or "unknown")

  -- Authoritative version identity (content hash of the loaded controller code).
  -- Appended unconditionally like BrakeTestRunID so it survives the pcall above.
  metadataStr = metadataStr .. " | ABS_CodeKey=" .. tostring(codeKey or "unknown")

  -- Create header
  local header = metadataStr .. "\nTime,CarSpeed,BrakeInput,FL_Speed,FR_Speed,RL_Speed,RR_Speed,FL_Brake,FR_Brake,RL_Brake,RR_Brake,FL_Slip,FR_Slip,RL_Slip,RR_Slip,ABS_Active,Airspeed,FL_Downforce,FR_Downforce,RL_Downforce,RR_Downforce,FL_DownforceRaw,FR_DownforceRaw,RL_DownforceRaw,RR_DownforceRaw,FL_PeakForce,FR_PeakForce,RL_PeakForce,RR_PeakForce,FL_TrueSlip,FR_TrueSlip,RL_TrueSlip,RR_TrueSlip,FL_SideSlip,FR_SideSlip,RL_SideSlip,RR_SideSlip,FL_SlipEnergy,FR_SlipEnergy,RL_SlipEnergy,RR_SlipEnergy,FL_TrueTorque,FR_TrueTorque,RL_TrueTorque,RR_TrueTorque,ThrottleInput,UserSteerInput,SteerInput,SteerAngle,FrontWheelAngle,FL_WheelAngle,FR_WheelAngle,RPM,Gear,FL_Contact,FR_Contact,RL_Contact,RR_Contact,FL_ContactDepth,FR_ContactDepth,RL_ContactDepth,RR_ContactDepth,FL_ContactMat1,FR_ContactMat1,RL_ContactMat1,RR_ContactMat1,FL_ContactMat2,FR_ContactMat2,RL_ContactMat2,RR_ContactMat2,FL_DynamicRadius,FR_DynamicRadius,RL_DynamicRadius,RR_DynamicRadius,FL_DriveTorque,FR_DriveTorque,RL_DriveTorque,RR_DriveTorque,FL_BrakeTemp,FR_BrakeTemp,RL_BrakeTemp,RR_BrakeTemp,FL_TireTemp,FR_TireTemp,RL_TireTemp,RR_TireTemp,EnvTemp,VirtualAirspeed,VirtualSensorsSpeed,Gx,Gy,Gz,YawRate,FL_FricStat,FR_FricStat,RL_FricStat,RR_FricStat,FL_FricSlid,FR_FricSlid,RL_FricSlid,RR_FricSlid,FL_BrakeEff,FR_BrakeEff,RL_BrakeEff,RR_BrakeEff,FL_DesiredBrake,FR_DesiredBrake,RL_DesiredBrake,RR_DesiredBrake"

  if customState then
    header = header .. ",FusedSpeed,NextD,FL_Target,FR_Target,RL_Target,RR_Target,FL_PID,FR_PID,RL_PID,RR_PID,TrimOffset,TrimDirection,FL_MaxBrake,FR_MaxBrake,RL_MaxBrake,RR_MaxBrake"
    header = header .. ",PID_KP,PID_KI,PID_KD,ImuSpeed,ImuClampCount,FwdVel,VLat,VVert,FL_EffectiveTarget,FR_EffectiveTarget,RL_EffectiveTarget,RR_EffectiveTarget,FL_GripD,FR_GripD,RL_GripD,RR_GripD,GripConfidentCount,LowSpeedBoost,FrontTrimOffset,FrontTrimDirection,EBD_RearScale,EBD_Active,FL_SlipError,FR_SlipError,RL_SlipError,RR_SlipError,FL_SlipIntegral,FR_SlipIntegral,RL_SlipIntegral,RR_SlipIntegral,FL_SlipDeriv,FR_SlipDeriv,RL_SlipDeriv,RR_SlipDeriv,SnapUpCount,SnapDnCount,ProbeCount"
    -- [SLIP GOVERNOR 2026-07-15] common-mode governor extended columns
    header = header .. ",TrueSlipPack,LambdaStar,GovMult,GovEvents"
    -- [SLIP GOVERNOR DIAG 2026-07-18] fit-starvation diagnostics (why lambda* frozen at seed)
    header = header .. ",GovFitN,GovF0,GovFN,GovTq"
    -- [D-PROBE 2026-07-18] boundary-probe exploration state
    header = header .. ",ExploreMult,ProbeVerdict,ProbeSatHold"
    -- [LOOP-RECON 2026-07-19] PID decomposition: ffTerm (the previously-invisible additive
    -- base), the exact slip the PID consumed (SlipUsed -- FL_Slip is the logger's own
    -- differential calc, NOT what the controller sees), P/I/D terms as summed, pre-clamp
    -- output, raw derivative, 200Hz control-tick marker, and the slip-seeker state.
    header = header .. ",CtrlTick,FL_FFTerm,FR_FFTerm,RL_FFTerm,RR_FFTerm,FL_SlipUsed,FR_SlipUsed,RL_SlipUsed,RR_SlipUsed"
    header = header .. ",FL_P_out,FR_P_out,RL_P_out,RR_P_out,FL_I_out,FR_I_out,RL_I_out,RR_I_out,FL_D_out,FR_D_out,RL_D_out,RR_D_out"
    header = header .. ",FL_PID_preclamp,FR_PID_preclamp,RL_PID_preclamp,RR_PID_preclamp"
    header = header .. ",FL_PID_overshoot,FR_PID_overshoot,RL_PID_overshoot,RR_PID_overshoot"
    header = header .. ",FL_SlipDerivRaw,FR_SlipDerivRaw,RL_SlipDerivRaw,RR_SlipDerivRaw,SeekTarget,SeekScore"
  end
  header = header .. ",ObjSpeed,RawSensorY\n"   -- truth (fwd physics vel) + raw accel fused integrates

  local f = io.open(filename, "w")
  if not f then
    -- if telemetry folder doesn't exist, try root
    filename = "DynamicABS_Run_" .. dateStr .. "_" .. label .. ".csv"
    f = io.open(filename, "w")
  end
  
  if f then
    f:write(header)
    for i=1, #logBuffer do
      f:write(logBuffer[i] .. "\n")
    end
    f:close()
    log('I', 'absTelemetry', 'Successfully saved telemetry file: ' .. filename .. ' with ' .. tostring(#logBuffer) .. ' frames of data.')
  else
    log('E', 'absTelemetry', 'CRITICAL ERROR: Failed to open file for writing: ' .. filename)
  end
  
  logBuffer = {}
  pushedTelemetry = nil
  isCustomABS = false
  cachedControllerName = nil
  currentBrakeTestRunID = nil
end

-- Task #8/#9/#18: recording lifecycle is gated entirely on the Brake Test Mod's own
-- true-2kHz measurement state (extensions.brakeTest.getMeasurementInfo(), read fresh
-- every onPhysicsStep call here) rather than this logger's own brake-input/speed
-- heuristics. This guarantees:
--   - a run is captured ONLY while a Brake Test measurement is active ("brakeState ==
--     measuring" in brakeTest.lua), so the CSV always corresponds to a real,
--     UI-initiated brake test (stock ABS included -- the gate is "was a Brake Test
--     measuring", not "is DynamicABS installed");
--   - the FULL measurement window is captured (every physics frame while active stays
--     true), instead of the old brakeInput/wheelspeed stopTimer heuristic which could
--     truncate a run to a ~0.28s fragment on noisy brake/speed signals;
--   - if BrakeTestMod isn't loaded, extensions.brakeTest is nil and nothing is ever
--     recorded, per the "braking event must go through the Brake Test UI" rule.
local function update(dtPhys)
  local bt = nil
  pcall(function()
    if extensions.brakeTest and extensions.brakeTest.getMeasurementInfo then
      bt = extensions.brakeTest.getMeasurementInfo()
    end
  end)

  local btActive        = (bt and bt.active) or false
  local btStartSpeedMph = (bt and bt.startSpeedMph) or 0
  local btRunID          = bt and bt.runID or nil

  if btActive and btStartSpeedMph > 15 then
    if not isLogging then
      isLogging = true
      logBuffer = {}
      runStartTime = obj:getSimTime()
      frameCount = 0
      runElapsed = 0
      currentBrakeTestRunID = btRunID

      if not isCustomABS then
        cachedControllerName = getActiveABSName()
        if cachedControllerName then
          isCustomABS = (getCustomABSState(cachedControllerName) ~= nil)
        end
      end
    end

    runElapsed = runElapsed + dtPhys
    frameCount = frameCount + 1
    local t = obj:getSimTime() - runStartTime

    local brakeInput = electrics.values.brake or 0

    local vel = obj:getVelocity()
    local dir = obj:getDirectionVector()
    local isReverse = vel and dir and (dir:dot(vel) < -0.1) or false
    -- Ground-truth speed for fusedSpeed accuracy: physics velocity projected onto the
    -- car's forward axis (signed, matches Airspeed's convention). Core vlua, .drive-legal
    -- (NOT the .tech AdvancedIMU). Logged as the ObjSpeed column.
    local objSpeed = (vel and dir) and dir:dot(vel) or 0
    -- RAW longitudinal accel (body-frame, gravity-cancelled) = EXACTLY what the controller
    -- integrates into fusedSpeed (Dynamic_ABS.lua latestSensorY). NOT the smoothed sensors.gy2
    -- logged as Gy. Needed to verify the onset mechanism + build an offline fusedSpeed simulator.
    local rawSensorY = (sensors and sensors.ffiSensors and sensors.ffiSensors.sensorY) or 0

    local carSpeed = electrics.values.wheelspeed or 0
    if isReverse then carSpeed = -carSpeed end
    
    local airspeed = electrics.values.airspeed or 0
    if isReverse then airspeed = -airspeed end
    
    local throttleInput = electrics.values.throttle or 0
    local userSteerInput = 0
    pcall(function() userSteerInput = input.steering or 0 end)
    local steerInput = electrics.values.steering_input or 0
    local steerAngle = electrics.values.steering or 0
    local fwa = 0
    local flWheelAngle = 0
    local frWheelAngle = 0
    local rpm = electrics.values.rpm or 0
    local gear = electrics.values.gearIndex or electrics.values.gear_A or 0
    local virtualAirspeed = electrics.values.virtualAirspeed or -1
      local virtualSensorsSpeed = -1
      local dd = controller.getController("drivingDynamics")
      if dd and dd.virtualSensors and dd.virtualSensors.virtual then
        virtualSensorsSpeed = dd.virtualSensors.virtual.speed
      elseif dd and dd.sensors and dd.sensors.virtualSensors and dd.sensors.virtualSensors.virtual then
        virtualSensorsSpeed = dd.sensors.virtualSensors.virtual.speed
      end
    local gx = sensors.gx2 or 0
    local gy = sensors.gy2 or 0
    local gz = sensors.gz2 or 0
    local yawRate = 0
    pcall(function() yawRate = obj:getYawAngularVelocity() or 0 end)
    
    local envTemp = 0
    pcall(function() envTemp = obj:getEnvTemperature() - 273.15 end)
    
    -- bff = brakeThermalEfficiency (0..1 fade fraction), dbrake = desiredBrakingTorque
    -- (raw ABS request BEFORE hydraulic delay & thermal fade). brake (=wd.brakingTorque)
    -- is the DELIVERED torque = dbrake * bff, so dbrake vs brake exposes fade directly.
    local wData = {
      RR = {speed=0, brake=0, df=0, dfRaw=0, peakForce=0, slip=0, sideSlip=0, slipEnergy=0, tq=0, contact=0, cdepth=0, mat1=-1, mat2=-1, dynRadius=0, dtq=0, btemp=0, ttemp=0, bff=1, dbrake=0},
      RL = {speed=0, brake=0, df=0, dfRaw=0, peakForce=0, slip=0, sideSlip=0, slipEnergy=0, tq=0, contact=0, cdepth=0, mat1=-1, mat2=-1, dynRadius=0, dtq=0, btemp=0, ttemp=0, bff=1, dbrake=0},
      FR = {speed=0, brake=0, df=0, dfRaw=0, peakForce=0, slip=0, sideSlip=0, slipEnergy=0, tq=0, contact=0, cdepth=0, mat1=-1, mat2=-1, dynRadius=0, dtq=0, btemp=0, ttemp=0, bff=1, dbrake=0},
      FL = {speed=0, brake=0, df=0, dfRaw=0, peakForce=0, slip=0, sideSlip=0, slipEnergy=0, tq=0, contact=0, cdepth=0, mat1=-1, mat2=-1, dynRadius=0, dtq=0, btemp=0, ttemp=0, bff=1, dbrake=0}
    }
    
    if wheels and wheels.wheels then
      for _, wd in pairs(wheels.wheels) do
        if wData[wd.name] then
          local ttemp = 0
          -- TireTemp investigation: obj:getWheelCoreTemperature(wd.wheelID) IS the correct,
          -- official API for per-wheel tire core temperature -- it's the exact call BeamNG's
          -- own engine code uses for its tire-thermal debug/GUI overlay (see
          -- lua/vehicle/wheels.lua ~line 289: "wi.tireCoreTemperature = obj:getWheelCoreTemperature(wd.wheelID)"
          -- and lua/vehicle/bdebugImpl.lua ~line 361, both confirmed against the shipped
          -- BeamNG.drive install). There is no alternate/better function to switch to here.
          -- Empirically (checked every CSV in sample_data/) this reads a constant ~15C that is
          -- always bit-identical to EnvTemp, frame after frame, run after run -- whereas
          -- FL/FR/RL/RR_BrakeTemp (read the same way, just a different getter) visibly varies
          -- 45-120C in the same runs. So the read itself is working; the simulation's tire-core
          -- heat model just isn't generating any measurable temperature rise for this
          -- vehicle/tire content under these test conditions. That's a sim/content limitation,
          -- not a bug in this logger -- flagging for a human to confirm with a different
          -- car/tire or a longer/harder braking test before assuming TireTemp is fixable here.
          pcall(function() ttemp = obj:getWheelCoreTemperature(wd.wheelID) - 273.15 end)
          local wheelAngle = 0
          if wd.name == "FL" or wd.name == "FR" then
            pcall(function()
              local steeringSign = steerInput < 0 and -1 or (steerInput > 0 and 1 or 0)
              local cosAngle = 1
              if wd.name == "FL" then
                cosAngle = obj:nodeVecPlanarCosRightForward(wd.node2, wd.node1)
              else
                cosAngle = obj:nodeVecPlanarCosRightForward(wd.node1, wd.node2)
              end
              cosAngle = math.min(1, math.max(-1, cosAngle or 1))
              wheelAngle = math.acos(cosAngle) * steeringSign
            end)
            if wd.name == "FL" then flWheelAngle = wheelAngle else frWheelAngle = wheelAngle end
          end
          
          wData[wd.name] = {
            speed = math.abs((wd.angularVelocity or 0) * (wd.radius or 0)),
            brake = wd.brakingTorque or wd.lastBrakeTorque or wd.brakeTorqueApplied or ((electrics.values.brake or 0) * (wd.brakeTorque or 0)),
            df = wd.downForce or 0,
            dfRaw = wd.downForceRaw or wd.downForce or 0,
            peakForce = wd.peakForce or 0,
            slip = wd.lastSlip or 0,
            sideSlip = wd.lastSideSlip or 0,
            slipEnergy = wd.slipEnergy or 0,
            tq = wd.brakeTorque or 0,
            contact = (((wd.contactMaterialID1 or -1) >= 0) and 1 or 0),
            cdepth = wd.contactDepth or 0,
            mat1 = wd.contactMaterialID1 or -1,
            mat2 = wd.contactMaterialID2 or -1,
            dynRadius = wd.dynamicRadius or wd.radius or 0,
            dtq = wd.propulsionTorque or 0,
            btemp = wd.brakeSurfaceTemperature or wd.brakeCoreTemperature or 0,
            ttemp = ttemp,
            -- wheels.wheels[] entries ARE the wheelRotator tables (wheels.lua:1187), so
            -- brakeThermalEfficiency/desiredBrakingTorque set there are readable here.
            bff = wd.brakeThermalEfficiency or 1,
            dbrake = wd.desiredBrakingTorque or 0
          }
        end
      end
    end
    fwa = (flWheelAngle + frWheelAngle) * 0.5
    
    local abs_speed = math.abs(carSpeed)
    local fl_slip = abs_speed > 1 and ((abs_speed - wData.FL.speed)/abs_speed) or 0
    local fr_slip = abs_speed > 1 and ((abs_speed - wData.FR.speed)/abs_speed) or 0
    local rl_slip = abs_speed > 1 and ((abs_speed - wData.RL.speed)/abs_speed) or 0
    local rr_slip = abs_speed > 1 and ((abs_speed - wData.RR.speed)/abs_speed) or 0
    
    -- customState is resolved once per frame here and reused below both for ABS_Active
    -- and for the trailing custom-controller columns (previously resolved twice; the
    -- second copy's "elseif isCustomABS and cachedControllerName" branch was unreachable
    -- dead code, since the preceding "if isCustomABS" already covers every case where
    -- isCustomABS is true).
    local customState = nil
    if isCustomABS then customState = pushedTelemetry or getCustomABSState(cachedControllerName)
    elseif cachedControllerName then customState = getCustomABSState(cachedControllerName) end

    -- ABS_Active: previously always 0 because the custom controller (Dynamic_ABS.lua)
    -- never sets electrics.values.abs / absActive -- those are stock-ABS-only flags.
    -- With a custom controller, "active" means it is actually pulling brake torque down
    -- from the driver's commanded input on at least one wheel. customState.pidOut is
    -- Dynamic_ABS's safety.lastAbsCoefs -- a per-wheel multiplier in [0,1] applied to the
    -- driver's brake torque, where 1.0 is the explicit "not intervening" passthrough
    -- value (see Dynamic_ABS.lua: absCoefs[i] = 1.0 when not braking/below MIN_SPEED).
    -- The < 0.99 threshold below mirrors Dynamic_ABS.lua's OWN internal "isAbsActive"
    -- check (search "lastAbsCoefs[j] or 1) < 0.99" in the controller's D-tracking code)
    -- so this logger stays consistent with the controller's own definition of
    -- "intervening" rather than inventing a new one. Falls back to the stock ABS
    -- electrics flags when there is no custom telemetry (e.g. stock ABS is in control).
    local absActive = 0
    if customState and customState.pidOut then
      for i = 1, 4 do
        if (customState.pidOut[i] or 1) < 0.99 then
          absActive = 1
          break
        end
      end
    else
      absActive = (electrics.values.abs == 1 or electrics.values.absActive == 1) and 1 or 0
    end

    local fields = {
      csvNum("%.4f", t), csvNum("%.2f", carSpeed), csvNum("%.2f", brakeInput),
      csvNum("%.2f", wData.FL.speed), csvNum("%.2f", wData.FR.speed), csvNum("%.2f", wData.RL.speed), csvNum("%.2f", wData.RR.speed),
      csvNum("%.0f", wData.FL.brake), csvNum("%.0f", wData.FR.brake), csvNum("%.0f", wData.RL.brake), csvNum("%.0f", wData.RR.brake),
      csvNum("%.3f", fl_slip), csvNum("%.3f", fr_slip), csvNum("%.3f", rl_slip), csvNum("%.3f", rr_slip), csvNum("%d", absActive), csvNum("%.2f", airspeed),
      csvNum("%.0f", wData.FL.df), csvNum("%.0f", wData.FR.df), csvNum("%.0f", wData.RL.df), csvNum("%.0f", wData.RR.df),
      csvNum("%.0f", wData.FL.dfRaw), csvNum("%.0f", wData.FR.dfRaw), csvNum("%.0f", wData.RL.dfRaw), csvNum("%.0f", wData.RR.dfRaw),
      csvNum("%.0f", wData.FL.peakForce), csvNum("%.0f", wData.FR.peakForce), csvNum("%.0f", wData.RL.peakForce), csvNum("%.0f", wData.RR.peakForce),
      csvNum("%.3f", wData.FL.slip), csvNum("%.3f", wData.FR.slip), csvNum("%.3f", wData.RL.slip), csvNum("%.3f", wData.RR.slip),
      csvNum("%.3f", wData.FL.sideSlip), csvNum("%.3f", wData.FR.sideSlip), csvNum("%.3f", wData.RL.sideSlip), csvNum("%.3f", wData.RR.sideSlip),
      csvNum("%.0f", wData.FL.slipEnergy), csvNum("%.0f", wData.FR.slipEnergy), csvNum("%.0f", wData.RL.slipEnergy), csvNum("%.0f", wData.RR.slipEnergy),
      csvNum("%.0f", wData.FL.tq), csvNum("%.0f", wData.FR.tq), csvNum("%.0f", wData.RL.tq), csvNum("%.0f", wData.RR.tq),
      csvNum("%.2f", throttleInput), csvNum("%.2f", userSteerInput), csvNum("%.2f", steerInput), csvNum("%.2f", steerAngle),
      csvNum("%.4f", fwa), csvNum("%.4f", flWheelAngle), csvNum("%.4f", frWheelAngle), csvNum("%.0f", rpm), csvNum("%.0f", gear),
      csvNum("%d", wData.FL.contact), csvNum("%d", wData.FR.contact), csvNum("%d", wData.RL.contact), csvNum("%d", wData.RR.contact),
      csvNum("%.3f", wData.FL.cdepth), csvNum("%.3f", wData.FR.cdepth), csvNum("%.3f", wData.RL.cdepth), csvNum("%.3f", wData.RR.cdepth),
      csvNum("%d", wData.FL.mat1), csvNum("%d", wData.FR.mat1), csvNum("%d", wData.RL.mat1), csvNum("%d", wData.RR.mat1),
      csvNum("%d", wData.FL.mat2), csvNum("%d", wData.FR.mat2), csvNum("%d", wData.RL.mat2), csvNum("%d", wData.RR.mat2),
      csvNum("%.3f", wData.FL.dynRadius), csvNum("%.3f", wData.FR.dynRadius), csvNum("%.3f", wData.RL.dynRadius), csvNum("%.3f", wData.RR.dynRadius),
      csvNum("%.0f", wData.FL.dtq), csvNum("%.0f", wData.FR.dtq), csvNum("%.0f", wData.RL.dtq), csvNum("%.0f", wData.RR.dtq),
      csvNum("%.1f", wData.FL.btemp), csvNum("%.1f", wData.FR.btemp), csvNum("%.1f", wData.RL.btemp), csvNum("%.1f", wData.RR.btemp),
      csvNum("%.1f", wData.FL.ttemp), csvNum("%.1f", wData.FR.ttemp), csvNum("%.1f", wData.RL.ttemp), csvNum("%.1f", wData.RR.ttemp), csvNum("%.1f", envTemp),
      csvNum("%.2f", virtualAirspeed), csvNum("%.2f", virtualSensorsSpeed), csvNum("%.2f", gx), csvNum("%.2f", gy), csvNum("%.2f", gz), csvNum("%.4f", yawRate),
      -- FL/FR/RL/RR_FricStat, FL/FR/RL/RR_FricSlid: intentionally left as unpopulated 0.00
      -- placeholders -- do NOT remove these columns (header/field counts must stay in sync).
      -- Investigated against the shipped BeamNG.drive install: the vehicle-side wd table in
      -- wheels.wheels[] (lua/vehicle/wheels.lua) does not expose any per-wheel friction
      -- coefficient at runtime -- it only carries contactMaterialID1/ID2 (already logged
      -- separately above as FL/FR/RL/RR_ContactMat1/2). The actual staticFrictionCoefficient /
      -- slidingFrictionCoefficient values exist only as ground-model MATERIAL properties in
      -- the GAME ENGINE Lua VM (lua/ge/extensions/core/environment.lua, "gm.staticFrictionCoefficient" /
      -- "gm.slidingFrictionCoefficient"), not in this vehicle-side VM, and are only reachable
      -- via an async obj:queueGameEngineLua() round trip -- far too slow/unsynchronized to
      -- read every physics step at 2kHz. No honest per-wheel static/sliding friction source
      -- exists in-VM, so these 8 columns are kept at 0 rather than faked.
      csvNum("%.2f", 0), csvNum("%.2f", 0), csvNum("%.2f", 0), csvNum("%.2f", 0), csvNum("%.2f", 0), csvNum("%.2f", 0), csvNum("%.2f", 0), csvNum("%.2f", 0),
      -- Brake thermal efficiency (0..1) and desired (requested) brake torque per wheel.
      -- FL_DesiredBrake vs FL_Brake (delivered) = how much the ABS asked for vs how much
      -- fade let through; FL_BrakeEff is the multiplier between them.
      csvNum("%.4f", wData.FL.bff), csvNum("%.4f", wData.FR.bff), csvNum("%.4f", wData.RL.bff), csvNum("%.4f", wData.RR.bff),
      csvNum("%.0f", wData.FL.dbrake), csvNum("%.0f", wData.FR.dbrake), csvNum("%.0f", wData.RL.dbrake), csvNum("%.0f", wData.RR.dbrake)
    }
    local line = table.concat(fields, ",")

    -- customState was already resolved above (used for ABS_Active); reused here as-is.
    if customState then
      -- Dynamic_ABS fills slipTargets/pidOut/maxBrake in LOGICAL order
      -- (1=RR, 2=RL, 3=FR, 4=FL). Remap to the physical header order FL,FR,RL,RR so
      -- each column is labeled by the correct wheel. Physical<-logical: FL=[4] FR=[3] RL=[2] RR=[1].
      line = line .. string.format(",%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.0f,%.0f,%.0f,%.0f,%.0f" ..
        ",%.4f,%.4f,%.4f,%.2f,%.0f,%.2f,%.2f,%.2f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.0f,%.3f,%.3f,%.0f,%.3f,%.0f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.0f,%.0f,%.0f",
        customState.fusedSpeed or 0,
        customState.nextD or 0,
        customState.slipTargets[4] or 0,
        customState.slipTargets[3] or 0,
        customState.slipTargets[2] or 0,
        customState.slipTargets[1] or 0,
        customState.pidOut[4] or 0,
        customState.pidOut[3] or 0,
        customState.pidOut[2] or 0,
        customState.pidOut[1] or 0,
        customState.trimOffset or 0,
        customState.trimDirection or 0,
        (customState.maxBrake and customState.maxBrake[4]) or 0,
        (customState.maxBrake and customState.maxBrake[3]) or 0,
        (customState.maxBrake and customState.maxBrake[2]) or 0,
        (customState.maxBrake and customState.maxBrake[1]) or 0,
        (customState.pidGains and customState.pidGains[1]) or 0,
        (customState.pidGains and customState.pidGains[2]) or 0,
        (customState.pidGains and customState.pidGains[3]) or 0,
        customState.imuSpeed or 0, customState.imuClamps or 0,
        customState.fwdVel or 0, customState.vLat or 0, customState.vVert or 0,
        (customState.effectiveTarget and customState.effectiveTarget[1]) or 0, (customState.effectiveTarget and customState.effectiveTarget[2]) or 0, (customState.effectiveTarget and customState.effectiveTarget[3]) or 0, (customState.effectiveTarget and customState.effectiveTarget[4]) or 0,
        (customState.gripD and customState.gripD[1]) or 0, (customState.gripD and customState.gripD[2]) or 0, (customState.gripD and customState.gripD[3]) or 0, (customState.gripD and customState.gripD[4]) or 0, customState.gripConfident or 0,
        customState.lowSpeedBoost or 1,
        customState.frontTrimOffset or 0, customState.frontTrimDirection or 0,
        customState.ebdRearScale or 1, customState.ebdActive or 0,
        (customState.slipError and customState.slipError[1]) or 0, (customState.slipError and customState.slipError[2]) or 0, (customState.slipError and customState.slipError[3]) or 0, (customState.slipError and customState.slipError[4]) or 0,
        (customState.slipIntegralState and customState.slipIntegralState[1]) or 0, (customState.slipIntegralState and customState.slipIntegralState[2]) or 0, (customState.slipIntegralState and customState.slipIntegralState[3]) or 0, (customState.slipIntegralState and customState.slipIntegralState[4]) or 0,
        (customState.slipDerivative and customState.slipDerivative[1]) or 0, (customState.slipDerivative and customState.slipDerivative[2]) or 0, (customState.slipDerivative and customState.slipDerivative[3]) or 0, (customState.slipDerivative and customState.slipDerivative[4]) or 0,
        customState.snapUpCount or 0, customState.snapDnCount or 0, customState.probeCount or 0
      )
      -- [SLIP GOVERNOR 2026-07-15] extended columns (kept ahead of the always-last
      -- ObjSpeed/RawSensorY pair; header adds these inside the same customState branch)
      line = line .. string.format(",%.4f,%.4f,%.4f,%d",
        tonumber(customState.trueSlipPack) or 0,
        tonumber(customState.lambdaStar) or 0,
        tonumber(customState.govMult) or 1,
        tonumber(customState.govEvents) or 0)
      -- [SLIP GOVERNOR DIAG 2026-07-18] fit-starvation diagnostics (order matches header)
      line = line .. string.format(",%d,%.1f,%.4f,%.1f",
        tonumber(customState.govFitN) or 0,
        tonumber(customState.govF0) or 0,
        tonumber(customState.govFN) or 0,
        tonumber(customState.govTq) or 0)
      -- [D-PROBE 2026-07-18] boundary-probe state (order matches header)
      line = line .. string.format(",%.4f,%d,%d",
        tonumber(customState.exploreMult) or 1,
        tonumber(customState.probeVerdict) or 0,
        tonumber(customState.probeSatHold) or 0)
      -- [LOOP-RECON 2026-07-19] PID decomposition (order matches header; arrays arrive
      -- from getTelemetry already remapped to FL,FR,RL,RR = [1],[2],[3],[4])
      local dgF = customState.ffTerm or {}
      local dgS = customState.slipUsed or {}
      local dgP = customState.pOut or {}
      local dgI = customState.iOut or {}
      local dgD = customState.dOut or {}
      local dgC = customState.pidPreclamp or {}
      local dgO = customState.pidOvershoot or {}
      local dgR = customState.slipDerivRaw or {}
      line = line .. string.format(",%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
        tonumber(customState.ctrlTick) or 0,
        dgF[1] or 0, dgF[2] or 0, dgF[3] or 0, dgF[4] or 0,
        dgS[1] or 0, dgS[2] or 0, dgS[3] or 0, dgS[4] or 0)
      line = line .. string.format(",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f",
        dgP[1] or 0, dgP[2] or 0, dgP[3] or 0, dgP[4] or 0,
        dgI[1] or 0, dgI[2] or 0, dgI[3] or 0, dgI[4] or 0,
        dgD[1] or 0, dgD[2] or 0, dgD[3] or 0, dgD[4] or 0)
      line = line .. string.format(",%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.5f,%.5f,%.5f,%.5f,%.4f,%.3f",
        dgC[1] or 0, dgC[2] or 0, dgC[3] or 0, dgC[4] or 0,
        dgO[1] or 0, dgO[2] or 0, dgO[3] or 0, dgO[4] or 0,
        dgR[1] or 0, dgR[2] or 0, dgR[3] or 0, dgR[4] or 0,
        tonumber(customState.seekTarget) or 0,
        tonumber(customState.seekScore) or -1)
    end

    -- ObjSpeed = forward physics-velocity ground truth; RawSensorY = raw accel fused integrates.
    -- ALWAYS the last two columns (matches header).
    line = line .. string.format(",%.4f,%.5f", objSpeed, rawSensorY)
    table.insert(logBuffer, line)

  else
    if isLogging then
      isLogging = false
      -- Task #9: only keep this capture if the Brake Test measurement window that
      -- gated it actually lasted more than 500ms; otherwise it's a fragment/junk
      -- (e.g. a brake tap that never reached "measuring" for long) -- discard the
      -- buffer instead of writing a truncated CSV to disk.
      if runElapsed > 0.5 then
        flushLogToDisk()
      else
        logBuffer = {}
        pushedTelemetry = nil
        isCustomABS = false
        cachedControllerName = nil
        currentBrakeTestRunID = nil
      end
    end
  end
end

local function onExtensionLoaded()
  log('I', 'absTelemetry', 'absTelemetryLogger extension successfully initialized and injected into vehicle!')
  buildWheelMaps()
  requestCodeKey()

  -- DIAGNOSTIC: print every controller key and whether it has getTelemetry
  if controller and controller.getControllers then
    local ctrls = controller.getControllers()
    for k, v in pairs(ctrls) do
      local hasTelem = type(v) == "table" and type(v.getTelemetry) == "function"
      log('I', 'absTelemetry', 'CTRL key="' .. tostring(k) .. '" hasTelemetry=' .. tostring(hasTelem))
    end
  else
    log('I', 'absTelemetry', 'DIAGNOSTIC: controller or getControllers is nil')
  end

  -- DIAGNOSTIC: try getController with your expected name
  if controller and controller.getController then
    local test = controller.getController("Dynamic_ABS")
    log('I', 'absTelemetry', 'getController("Dynamic_ABS") returned: ' .. type(test))
  end

  local f = io.open("telemetry_system_active.txt", "w")
  if f then
    f:write("The telemetry logger successfully loaded on: " .. os.date("%Y-%m-%d_%H-%M-%S"))
    f:close()
  end
end

M.onExtensionLoaded = onExtensionLoaded
M.onPhysicsStep = update
M.setCustomTelemetry = setCustomTelemetry
M.setCodeKey = setCodeKey

return M
