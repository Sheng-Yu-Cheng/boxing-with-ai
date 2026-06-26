-- radarbox_beam_file_poller.lua
-- mmWave Studio Lua script for low-frequency beam control by file exchange.
--
-- It polls CMD_PATH. When the command changes, it calls:
--   ar1.SetPerChirpPhaseShifterConfig(0, 0, tx0Code, tx1Code, tx2Code)
--
-- Command file format:
--   0,0,0       center beam
--   0,48,32    approximate +30 deg under d=lambda/2 assumption
--   0,16,32    approximate -30 deg under d=lambda/2 assumption
--
-- Suggested order:
--   1. Configure radar: 3Tx simultaneous, chirp0, infinite frame.
--   2. Start Python UDP receive logger.
--   3. Start DCA1000 record / StartFrame.
--   4. Run this Lua script.
--   5. Run Python angle writer to update CMD_PATH.
--
-- Stop:
--   Stop/abort this script from mmWave Studio when done.

CMD_PATH = "C:\\temp\\radarbox_beam_cmd.txt"
APPLY_LOG_PATH = "C:\\temp\\radarbox_beam_apply_log.txt"
POLL_MS = 500

-- 0 means infinite. Set e.g. 120 for a 60 second test at 500 ms polling.
MAX_ITER = 0

function now_string()
    return os.date("%Y-%m-%d %H:%M:%S")
end

function trim(s)
    if s == nil then
        return ""
    end
    return (s:gsub("^%s*(.-)%s*$", "%1"))
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

function read_file(path)
    local f = io.open(path, "r")
    if f == nil then
        return nil
    end
    local s = f:read("*all")
    f:close()
    return trim(s)
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

log_msg("RadarBox beam file poller started", "yellow")
log_msg("CMD_PATH=" .. CMD_PATH, "yellow")
log_msg("APPLY_LOG_PATH=" .. APPLY_LOG_PATH, "yellow")
log_msg("POLL_MS=" .. tostring(POLL_MS), "yellow")

local last_cmd = ""
local iter = 0

while true do
    iter = iter + 1

    local cmd = read_file(CMD_PATH)

    if cmd ~= nil and cmd ~= "" and cmd ~= last_cmd then
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
            log_msg("Invalid command ignored: " .. cmd, "red")
        end
    end

    if MAX_ITER ~= 0 and iter >= MAX_ITER then
        log_msg("MAX_ITER reached; poller stopped", "yellow")
        break
    end

    RSTD.Sleep(POLL_MS)
end
