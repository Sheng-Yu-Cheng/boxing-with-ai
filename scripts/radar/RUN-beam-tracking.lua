-- radarbox_run_record_startframe_beam_poll.lua
-- mmWave Studio runtime script for RadarBox closed-loop / low-frequency beam update.
--
-- This script DOES:
--   1. Initialize the beam command file polling function.
--   2. Apply the initial beam command once before StartFrame.
--   3. Start DCA1000 record to C:\temp\radar_back.bin.
--   4. Keep running and watch C:\temp\radarbox_radar_state.txt.
--   5. When state is "active", start radar frame and poll beam commands.
--   6. When state becomes "inactive", stop radar frame and wait for the next active state.
--
-- Command file format:
--   0,0,0
--   0,48,32
--   0,16,32
--   or any three integers in 0..63: tx0Code,tx1Code,tx2Code
--
-- Graceful stop:
--   Create this file while the script is running:
--       C:\temp\radarbox_stop.txt
--   The script will stop frame and exit the outer loop.
--
-- Game-controlled radar runtime state:
--   C:\temp\radarbox_radar_state.txt
--   active   -> StartFrame + beam polling loop
--   inactive -> StopFrame + idle wait loop
--
-- Note:
--   If mmWave Studio forcibly kills the script with a hard Abort button,
--   Lua cleanup may not always run. The safest stop is the stop-file method.

------------------------------------------------------------
-- USER CONSTANTS
------------------------------------------------------------

CMD_PATH = "C:\\temp\\radarbox_beam_cmd.txt"
RADAR_STATE_PATH = "C:\\temp\\radarbox_radar_state.txt"
STOP_PATH = "C:\\temp\\radarbox_stop.txt"
APPLY_LOG_PATH = "C:\\temp\\radarbox_beam_apply_log.txt"
ADC_OUTPUT_PATH = "C:\\temp\\radar_back.bin"

POLL_MS = 500
START_RECORD_TO_START_FRAME_SLEEP_MS = 1000
START_FRAME_TO_POLL_SLEEP_MS = 500

-- 0 means infinite. Set e.g. 240 for about 2 minutes at 500 ms polling.
MAX_ITER = 0

-- If true, create CMD_PATH with 0,0,0 when missing.
CREATE_DEFAULT_CMD_IF_MISSING = true
DEFAULT_CMD = "0,0,0"

------------------------------------------------------------
-- HELPERS
------------------------------------------------------------

function now_string()
    return os.date("%Y-%m-%d %H:%M:%S")
end

function trim(s)
    if s == nil then return "" end
    return (s:gsub("^%s*(.-)%s*$", "%1"))
end

function read_file(path)
    local f = io.open(path, "r")
    if f == nil then return nil end
    local s = f:read("*all")
    f:close()
    return trim(s)
end

function write_file(path, msg)
    local f = io.open(path, "w")
    if f ~= nil then
        f:write(msg)
        f:flush()
        f:close()
        return true
    end
    return false
end

function append_file(path, msg)
    local f = io.open(path, "a")
    if f ~= nil then
        f:write(msg)
        f:flush()
        f:close()
    end
end

function log_msg(msg, color)
    if color == nil then color = "yellow" end
    local line = string.format("[%s] %s", now_string(), msg)
    WriteToLog(line .. "\n", color)
    append_file(APPLY_LOG_PATH, line .. "\n")
end

function file_exists(path)
    local f = io.open(path, "r")
    if f ~= nil then
        f:close()
        return true
    end
    return false
end

function remove_file(path)
    if file_exists(path) then
        os.remove(path)
    end
end

function read_radar_state()
    local s = read_file(RADAR_STATE_PATH)
    if s == nil or s == "" then
        return "inactive"
    end
    return string.lower(trim(s))
end

function split_codes(s)
    local vals = {}
    for token in string.gmatch(s, "([^,]+)") do
        local x = tonumber(trim(token))
        if x == nil then
            return nil
        end
        vals[#vals + 1] = math.floor(x + 0.5)
    end

    if #vals ~= 3 then
        return nil
    end

    for i = 1, 3 do
        vals[i] = vals[i] % 64
        if vals[i] < 0 then vals[i] = vals[i] + 64 end
    end

    return vals
end

last_cmd = ""
frame_started = false

function apply_cmd_if_changed(force)
    local cmd = read_file(CMD_PATH)

    if cmd == nil or cmd == "" then
        if CREATE_DEFAULT_CMD_IF_MISSING then
            write_file(CMD_PATH, DEFAULT_CMD .. "\n")
            cmd = DEFAULT_CMD
            log_msg("CMD_PATH missing/empty; wrote default command " .. DEFAULT_CMD, "yellow")
        else
            return
        end
    end

    if force or cmd ~= last_cmd then
        local codes = split_codes(cmd)

        if codes ~= nil then
            log_msg(
                string.format(
                    "Apply chirp0 phase codes: TX0=%d TX1=%d TX2=%d",
                    codes[1], codes[2], codes[3]
                ),
                "yellow"
            )

            ar1.SetPerChirpPhaseShifterConfig(0, 0, codes[1], codes[2], codes[3])
            last_cmd = cmd
        else
            log_msg("Invalid command ignored: " .. tostring(cmd), "red")
        end
    end
end

function safe_stop_frame()
    if frame_started then
        log_msg("Stopping frame: ar1.StopFrame(0)", "yellow")
        ar1.StopFrame(0)
        frame_started = false
        RSTD.Sleep(1000)
    else
        log_msg("Frame was not marked started; skip StopFrame", "yellow")
    end
end

function main()
    log_msg("RadarBox record + state-controlled beam poll script started", "yellow")
    log_msg("CMD_PATH=" .. CMD_PATH, "yellow")
    log_msg("RADAR_STATE_PATH=" .. RADAR_STATE_PATH, "yellow")
    log_msg("STOP_PATH=" .. STOP_PATH, "yellow")
    log_msg("APPLY_LOG_PATH=" .. APPLY_LOG_PATH, "yellow")
    log_msg("ADC_OUTPUT_PATH=" .. ADC_OUTPUT_PATH, "yellow")
    log_msg("POLL_MS=" .. tostring(POLL_MS), "yellow")

    -- Avoid immediately stopping because of an old stop file.
    remove_file(STOP_PATH)

    -- Initialize beam update function before streaming starts.
    apply_cmd_if_changed(true)

    log_msg("Start DCA1000 record: " .. ADC_OUTPUT_PATH, "yellow")
    ar1.CaptureCardConfig_StartRecord(ADC_OUTPUT_PATH, 1)
    RSTD.Sleep(START_RECORD_TO_START_FRAME_SLEEP_MS)

    log_msg("Outer state loop entered; waiting for RADAR_STATE_PATH=active", "yellow")

    while true do
        if file_exists(STOP_PATH) then
            log_msg("STOP_PATH detected; exiting outer loop", "yellow")
            break
        end

        local state = read_radar_state()
        if state == "active" then
            apply_cmd_if_changed(true)

            if not frame_started then
                log_msg("RADAR_STATE active; Start radar frame: ar1.StartFrame()", "yellow")
                ar1.StartFrame()
                frame_started = true
                RSTD.Sleep(START_FRAME_TO_POLL_SLEEP_MS)
            end

            log_msg("Beam polling loop entered", "yellow")

            local iter = 0
            while true do
                iter = iter + 1

                if file_exists(STOP_PATH) then
                    log_msg("STOP_PATH detected inside polling loop", "yellow")
                    safe_stop_frame()
                    return
                end

                state = read_radar_state()
                if state ~= "active" then
                    log_msg("RADAR_STATE is " .. tostring(state) .. "; leaving beam polling loop", "yellow")
                    safe_stop_frame()
                    break
                end

                apply_cmd_if_changed(false)

                if MAX_ITER ~= 0 and iter >= MAX_ITER then
                    log_msg("MAX_ITER reached; stopping current active frame", "yellow")
                    safe_stop_frame()
                    break
                end

                RSTD.Sleep(POLL_MS)
            end
        else
            if frame_started then
                log_msg("RADAR_STATE inactive; stopping frame", "yellow")
                safe_stop_frame()
            end

            RSTD.Sleep(POLL_MS)
        end
    end
end

------------------------------------------------------------
-- RUN WITH CLEANUP
------------------------------------------------------------

local ok, err = xpcall(main, debug.traceback)

if not ok then
    log_msg("Lua runtime error or abort caught: " .. tostring(err), "red")
end

safe_stop_frame()

if ok then
    log_msg("RadarBox runtime script completed normally", "green")
else
    log_msg("RadarBox runtime script completed after error/abort; frame stop attempted", "red")
end
