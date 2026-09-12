-- trainerRecorder.lua — 400 Hz raw-data recorder for DynamicABS_Trainer distillation
local M = {}

local recording = false
local f = nil
local tickAccum = 0
local TICK_STEP = 1 / 400
local wrFR, wrFL, wrRR, wrRL
local initialized = false
local rowCount = 0

local HEADER = "gs,ws_fr,ws_fl,ws_rr,ws_rl,"
  .. "gy_inst,gx_inst,yaw_rate_inst,slam_fired,slam_fire_speed,"
  .. "inst_speed,last_brake_dist,last_brake_avg_g,fused_speed,brake_active,"
  .. "brake_in,win_gy_avg,win_gy_min,win_gy_max,win_yaw_avg,"
  .. "rpm,gear,steer,att_pitch,att_roll,throttle_in,gz_inst,"
  .. "abs_speed,brk_nm_fr,brk_nm_fl,brk_nm_rr,brk_nm_rl,"
  .. "cmd_fr,cmd_fl,cmd_rr,cmd_rl"

local FMT = "%.4f,%.4f,%.4f,%.4f,%.4f,"
  .. "%.4f,%.4f,%.4f,%.0f,%.4f,"
  .. "%.4f,%.4f,%.6f,%.4f,%.0f,"
  .. "%.4f,%.4f,%.4f,%.4f,%.4f,"
  .. "%.0f,%.0f,%.4f,%.6f,%.6f,%.4f,%.4f,"
  .. "%.4f,%.1f,%.1f,%.1f,%.1f,"
  .. "%.6f,%.6f,%.6f,%.6f"


local function findWheels()
  wrFR, wrFL, wrRR, wrRL = nil, nil, nil, nil
  if not wheels or not wheels.wheelRotators then return false end
  for i = 0, (wheels.wheelRotatorCount or 0) - 1 do
    local w = wheels.wheelRotators[i]
    if w then
      local n = (w.name or ""):upper()
      if n == "FR" then wrFR = w
      elseif n == "FL" then wrFL = w
      elseif n == "RR" then wrRR = w
      elseif n == "RL" then wrRL = w
      end
    end
  end
  initialized = (wrFR ~= nil and wrFL ~= nil and wrRR ~= nil and wrRL ~= nil)
  return initialized
end


local function startRecording(path)
  if not initialized and not findWheels() then
    print("[trainerRecorder] cannot find all 4 corner wheels")
    return false
  end
  f = io.open(path, "w")
  if not f then
    print("[trainerRecorder] cannot open " .. tostring(path))
    return false
  end
  f:write(HEADER .. "\n")
  recording = true
  tickAccum = 0
  rowCount = 0
  enablePhysicsStepHook()
  print("[trainerRecorder] recording started: " .. path)
  return true
end


local function stopRecording()
  recording = false
  if f then
    f:close()
    f = nil
  end
  print("[trainerRecorder] stopped, " .. rowCount .. " rows")
  return rowCount
end


local function onPhysicsStep(dtPhys)
  if not recording or not f then return end
  tickAccum = tickAccum + dtPhys
  if tickAccum < TICK_STEP then return end
  tickAccum = tickAccum - TICK_STEP

  local ev = electrics.values
  local vel = obj:getVelocity()
  local gs = 0
  if vel then gs = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z) end

  f:write(string.format(FMT,
    gs,
    math.abs(wrFR.wheelSpeed or 0), math.abs(wrFL.wheelSpeed or 0),
    math.abs(wrRR.wheelSpeed or 0), math.abs(wrRL.wheelSpeed or 0),
    ev.tel_gy_inst or 0, ev.tel_gx_inst or 0, ev.tel_yaw_rate_inst or 0,
    ev.tel_slam_fired or 0, ev.tel_slam_fire_speed or 0,
    ev.tel_inst_speed or 0, ev.tel_last_brake_dist or 0, ev.tel_last_brake_avg_g or 0,
    ev.tel_fused_speed or 0, ev.tel_brake_active or 0,
    input.brake or 0,
    ev.tel_win_gy_avg or 0, ev.tel_win_gy_min or 0, ev.tel_win_gy_max or 0, ev.tel_win_yaw_avg or 0,
    ev.tel_rpm or 0, ev.tel_gear or 0, ev.tel_steer or 0,
    ev.tel_att_pitch or 0, ev.tel_att_roll or 0, ev.tel_throttle_in or 0, ev.tel_gz_inst or 0,
    ev.tel_abs_speed or 0,
    wrFR.brakingTorque or 0, wrFL.brakingTorque or 0,
    wrRR.brakingTorque or 0, wrRL.brakingTorque or 0,
    ev.abs_cmd_fr or 0, ev.abs_cmd_fl or 0, ev.abs_cmd_rr or 0, ev.abs_cmd_rl or 0
  ) .. "\n")
  rowCount = rowCount + 1
end


local function onReset()
  if recording then stopRecording() end
  initialized = false
end


M.onReset = onReset
M.onPhysicsStep = onPhysicsStep
M.startRecording = startRecording
M.stopRecording = stopRecording

return M
