-- radarbox_awr2243_3tx_beam_config.lua
-- One-file mmWave Studio setup for RadarBox AWR2243BOOST + DCA1000.
--
-- Goal:
--   Configure AWR2243 as a 3Tx simultaneous + 4Rx beam-control radar.
--   chirp 0 = TX0 + TX1 + TX2 simultaneously
--   frame = chirp 0 repeated NUM_CHIRP_LOOPS times
--
-- This script DOES:
--   1. Connect AWR2243BOOST
--   2. Download BSS/MSS firmware
--   3. PowerOn + RF Enable
--   4. Configure DCA1000 Ethernet / mode / packet delay
--   5. Configure AWR2243 static settings as 3Tx + 4Rx
--   6. Enable RF LDO bypass / PA LDO disable required by 3Tx simultaneous
--   7. Enable per-chirp phase shifter
--   8. Configure LVDS data path
--   9. Configure 3Tx simultaneous profile/chirp/frame
--  10. Set initial chirp0 phase code = [0, 0, 0]
--
-- This script DOES NOT:
--   - Start DCA1000 recording
--   - Start radar frame
--   - Stop radar frame
--   - Run the beam update polling loop
--
-- Recommended runtime order:
--   1. Run this config script.
--   2. Start your Python UDP receiver / logger.
--   3. Run radarbox_run_record_startframe_beam_poll.lua.
--   4. Run Python AoA / beam writer to update C:\temp\radarbox_beam_cmd.txt.
--
-- Important data-shape note:
--   3Tx simultaneous does NOT produce 12 virtual RX channels.
--   Raw ADC is still 4Rx per chirp. Python receiver should keep tx_groups=1.

------------------------------------------------------------
-- USER CONSTANTS
------------------------------------------------------------

-- Serial / firmware
COM_PORT = 20
BAUD_RATE = 921600
CONNECT_TIMEOUT_MS = 1000
SOP_MODE = 2

DO_FULL_RESET = true
DOWNLOAD_FIRMWARE = true
DO_POWER_ON_AND_RF_ENABLE = true

BSS_FW_PATH = "C:\\ti\\mmwave_studio_03_00_00_14\\rf_eval_firmware\\AWR2243_ES1_1\\radarss\\xwr22xx_radarss.bin"
MSS_FW_PATH = "C:\\ti\\mmwave_studio_03_00_00_14\\rf_eval_firmware\\AWR2243_ES1_1\\masterss\\xwr22xx_masterss.bin"

-- DCA1000
PC_IP = "192.168.33.30"
DCA1000_IP = "192.168.33.180"
DCA1000_MAC = "12:34:56:78:90:12"
CONFIG_PORT = 4096
DATA_PORT = 4098

-- Known-good DCA mode from manual log:
-- ar1.CaptureCardConfig_Mode(1, 1, 1, 2, 3, 30)
DCA_MODE_ARG0 = 1
DCA_MODE_ARG1 = 1
DCA_MODE_ARG2 = 1
DCA_MODE_ARG3 = 2
DCA_MODE_ARG4 = 3
DCA_MODE_ARG5 = 30

-- Keep 25 us first. If UDP packet gaps appear, try 50 us.
PACKET_DELAY_US = 25

-- Static config: 3Tx simultaneous + 4Rx.
TX0_ENABLE = 1
TX1_ENABLE = 1
TX2_ENABLE = 1

RX0_ENABLE = 1
RX1_ENABLE = 1
RX2_ENABLE = 1
RX3_ENABLE = 1

-- ADC format from known-good setup.
ADC_FORMAT = 2          -- Complex 1x
ADC_BITS_OR_OUT_FMT = 1 -- 16-bit / GUI-compatible value
IQ_SWAP = 0

-- 3Tx simultaneous on AWR2243BOOST requires RF LDO bypass + PA LDO disable.
RF_LDO_BYPASS_CODE = 0x3

-- Enable per-chirp phase shifter.
-- ar1.SetMiscConfig(1, 0, 0, 0)
PER_CHIRP_PHASE_SHIFTER_ENABLE = 1
MISC_RESERVED_1 = 0
MISC_RESERVED_2 = 0
MISC_RESERVED_3 = 0

FREQ_LOW_GHZ = 77
FREQ_HIGH_GHZ = 81

LP_ADC_MODE = 0
LP_RESERVED = 0

RUN_RF_INIT = true

-- Data path / LVDS: known-good config.
DATA_PATH_CONFIG_ARG0 = 513
DATA_PATH_CONFIG_ARG1 = 1216644097
DATA_PATH_CONFIG_ARG2 = 0

LVDS_CLK_ARG0 = 1
LVDS_CLK_ARG1 = 1

LANE_FORMAT = 0
LANE1_ENABLE = 1
LANE2_ENABLE = 1
LANE3_ENABLE = 1
LANE4_ENABLE = 1
MSB_FIRST = 1
CRC_ENABLE = 0
PACKET_END_PULSE_ENABLE = 0

-- Sensor config: 3Tx beam-control preset.
PROFILE_ID = 0
START_FREQ_GHZ = 77

-- Current stable test timing:
-- Tc ~= IDLE_TIME_US + RAMP_END_TIME_US = 80 us.
-- ADC sampling time = NUM_ADC_SAMPLES / sampleRate = 25.6 us.
-- ADC ends at 6 + 25.6 = 31.6 us < 60 us ramp end.
IDLE_TIME_US = 20
ADC_START_TIME_US = 6
RAMP_END_TIME_US = 60

TX_OUT_POWER_BACKOFF_CODE = 0
PROFILE_TX_PHASE_SHIFTER = 0
FREQ_SLOPE_CONST_MHZ_PER_US = 29.982
TX_START_TIME_US = 0
NUM_ADC_SAMPLES = 256
DIG_OUT_SAMPLE_RATE_KSPS = 10000
HPF_CORNER_FREQ1 = 0
HPF_CORNER_FREQ2 = 0
RX_GAIN_CODE = 94

-- chirp0 = TX0 + TX1 + TX2 simultaneous
CHIRP_START_INDEX = 0
CHIRP_END_INDEX = 0
CHIRP_FREQ_VAR = 0
CHIRP_SLOPE_VAR = 0
CHIRP_IDLE_VAR = 0
CHIRP_ADC_START_VAR = 0
CHIRP_TX_START_VAR = 0
CHIRP_TX0_ENABLE = 1
CHIRP_TX1_ENABLE = 1
CHIRP_TX2_ENABLE = 1

-- Initial per-chirp phase code for chirp0.
-- 1 code = 5.625 degrees.
INITIAL_TX0_PHASE_CODE = 0
INITIAL_TX1_PHASE_CODE = 0
INITIAL_TX2_PHASE_CODE = 0

START_CHIRP = 0
END_CHIRP = 0
NUM_FRAMES = 0          -- 0 = continuous / infinite frame mode
NUM_CHIRP_LOOPS = 64
FRAME_PERIODICITY_MS = 20
TRIGGER_DELAY_US = 0
TRIGGER_SELECT = 1

API_SLEEP_MS = 250

------------------------------------------------------------
-- HELPERS
------------------------------------------------------------

function log(msg, color)
    if color == nil then color = "green" end
    WriteToLog(msg .. "\n", color)
end

function warn(msg)
    WriteToLog("WARNING: " .. msg .. "\n", "yellow")
end

function die(msg)
    WriteToLog("ERROR: " .. msg .. "\n", "red")
    error(msg)
end

function sleep_ms(ms)
    RSTD.Sleep(ms)
end

function sleep()
    RSTD.Sleep(API_SLEEP_MS)
end

function section(title)
    log("------------------------------------------------------------", "yellow")
    log(title, "yellow")
    log("------------------------------------------------------------", "yellow")
end

function validate_config()
    local sample_time_us = NUM_ADC_SAMPLES / DIG_OUT_SAMPLE_RATE_KSPS * 1000.0
    local adc_end_us = ADC_START_TIME_US + sample_time_us
    local chirp_time_us = IDLE_TIME_US + RAMP_END_TIME_US
    local burst_time_ms = NUM_CHIRP_LOOPS * chirp_time_us / 1000.0

    if adc_end_us >= RAMP_END_TIME_US then
        die(string.format(
            "Invalid profile timing: ADC ends at %.2f us, but ramp ends at %.2f us.",
            adc_end_us, RAMP_END_TIME_US
        ))
    end

    if burst_time_ms >= FRAME_PERIODICITY_MS then
        die(string.format(
            "Invalid frame timing: chirp burst needs %.2f ms, but frame period is %.2f ms.",
            burst_time_ms, FRAME_PERIODICITY_MS
        ))
    end
end

function print_expected_performance()
    local c = 299792458.0
    local fc_hz = START_FREQ_GHZ * 1000000000.0
    local lambda = c / fc_hz
    local tc = (IDLE_TIME_US + RAMP_END_TIME_US) / 1000000.0
    local v_max = lambda / (4.0 * tc)
    local dv = lambda / (2.0 * NUM_CHIRP_LOOPS * tc)

    -- 3Tx simultaneous raw ADC is still 4Rx, tx_groups=1.
    local bytes_per_frame = NUM_ADC_SAMPLES * 4 * 2 * 2 * NUM_CHIRP_LOOPS
    local frame_rate_hz = 1000.0 / FRAME_PERIODICITY_MS
    local data_rate_MBps = bytes_per_frame * frame_rate_hz / 1000000.0

    log("Expected radar live performance:", "green")
    log(string.format("  Tc ~= %.1f us", tc * 1000000.0), "green")
    log(string.format("  frame rate ~= %.2f Hz", frame_rate_hz), "green")
    log(string.format("  v_max ~= %.2f m/s", v_max), "green")
    log(string.format("  velocity bin ~= %.3f m/s", dv), "green")
    log(string.format("  bytes/frame = %d", bytes_per_frame), "green")
    log(string.format("  stream rate ~= %.2f MB/s", data_rate_MBps), "green")
end

------------------------------------------------------------
-- START
------------------------------------------------------------

validate_config()

section("RadarBox AWR2243 3Tx simultaneous beam config started")
log(string.format("COM_PORT=%d, PC_IP=%s, DCA1000_IP=%s", COM_PORT, PC_IP, DCA1000_IP), "yellow")
log(string.format("Preset: 3Tx simultaneous + 4Rx, %d samples, %d chirps/frame, %d ms frame, numFrames=%d",
    NUM_ADC_SAMPLES, NUM_CHIRP_LOOPS, FRAME_PERIODICITY_MS, NUM_FRAMES), "yellow")
print_expected_performance()
log("This script will NOT start DCA1000 record and will NOT start radar frame.", "yellow")

------------------------------------------------------------
-- 1. CONNECT AWR2243
------------------------------------------------------------

section("1. Connect AWR2243")

ar1.selectRadarMode(0)
sleep()

ar1.selectCascadeMode(0)
sleep()

if DO_FULL_RESET then
    log("FullReset + SOPControl(2)", "yellow")
    ar1.FullReset()
    sleep_ms(1000)

    ar1.SOPControl(SOP_MODE)
    sleep_ms(1000)
else
    warn("Skipping FullReset/SOPControl.")
end

log(string.format("Connect RS232 COM%d", COM_PORT), "yellow")
ar1.Connect(COM_PORT, BAUD_RATE, CONNECT_TIMEOUT_MS)
sleep_ms(1500)

ar1.Calling_IsConnected()
sleep()

log("Select XWR2243 / 77G", "yellow")
ar1.SelectChipVersion("AR1243")
sleep()
ar1.SelectChipVersion("AR1243")
sleep()
ar1.deviceVariantSelection("XWR2243")
sleep()
ar1.frequencyBandSelection("77G")
sleep()
ar1.SelectChipVersion("XWR2243")
sleep()

if DOWNLOAD_FIRMWARE then
    log("Download BSS firmware", "yellow")
    ar1.DownloadBSSFw(BSS_FW_PATH)
    sleep_ms(1200)
    ar1.GetBSSFwVersion()
    sleep()
    ar1.GetBSSPatchFwVersion()
    sleep()

    log("Download MSS firmware", "yellow")
    ar1.DownloadMSSFw(MSS_FW_PATH)
    sleep_ms(2000)
    ar1.GetMSSFwVersion()
    sleep()
else
    warn("Skipping firmware download.")
end

if DO_POWER_ON_AND_RF_ENABLE then
    log("PowerOn", "yellow")
    ar1.PowerOn(0, 1000, 0, 0)
    sleep_ms(2500)

    ar1.SelectChipVersion("AR1243")
    sleep()
    ar1.SelectChipVersion("XWR2243")
    sleep()

    log("RfEnable", "yellow")
    ar1.RfEnable()
    sleep_ms(1500)

    log("Firmware version check", "yellow")
    ar1.GetMSSFwVersion()
    sleep()
    ar1.GetBSSFwVersion()
    sleep()
    ar1.GetBSSPatchFwVersion()
    sleep()
else
    warn("Skipping PowerOn/RfEnable.")
end

------------------------------------------------------------
-- 2. CONFIGURE DCA1000
------------------------------------------------------------

section("2. Configure DCA1000")

ar1.GetCaptureCardDllVersion()
sleep()

ar1.SelectCaptureDevice("DCA1000")
sleep()

log(string.format("EthInit: PC=%s, DCA=%s, cfgPort=%d, dataPort=%d",
    PC_IP, DCA1000_IP, CONFIG_PORT, DATA_PORT), "yellow")

ar1.CaptureCardConfig_EthInit(
    PC_IP,
    DCA1000_IP,
    DCA1000_MAC,
    CONFIG_PORT,
    DATA_PORT
)
sleep()

log("CaptureCardConfig_Mode", "yellow")
ar1.CaptureCardConfig_Mode(
    DCA_MODE_ARG0,
    DCA_MODE_ARG1,
    DCA_MODE_ARG2,
    DCA_MODE_ARG3,
    DCA_MODE_ARG4,
    DCA_MODE_ARG5
)
sleep()

log(string.format("PacketDelay = %d us", PACKET_DELAY_US), "yellow")
ar1.CaptureCardConfig_PacketDelay(PACKET_DELAY_US)
sleep()

ar1.GetCaptureCardFPGAVersion()
sleep()

------------------------------------------------------------
-- 3. STATIC CONFIG: 3Tx + 4Rx
------------------------------------------------------------

section("3. Static config: 3Tx simultaneous + 4Rx")

log("ChanNAdcConfig: Tx0~Tx2 enabled, Rx0~Rx3 enabled, Complex1x", "yellow")
ar1.ChanNAdcConfig(
    TX0_ENABLE, TX1_ENABLE, TX2_ENABLE,
    RX0_ENABLE, RX1_ENABLE, RX2_ENABLE, RX3_ENABLE,
    ADC_FORMAT,
    ADC_BITS_OR_OUT_FMT,
    IQ_SWAP
)
sleep()

log("RfLdoBypassConfig(0x3): required for 3Tx simultaneous", "yellow")
ar1.RfLdoBypassConfig(RF_LDO_BYPASS_CODE)
sleep()

log("SetMiscConfig: enable per-chirp phase shifter", "yellow")
ar1.SetMiscConfig(
    PER_CHIRP_PHASE_SHIFTER_ENABLE,
    MISC_RESERVED_1,
    MISC_RESERVED_2,
    MISC_RESERVED_3
)
sleep()

log("LPModConfig", "yellow")
ar1.LPModConfig(LP_ADC_MODE, LP_RESERVED)
sleep()

log("CalMon frequency limits", "yellow")
ar1.SetCalMonFreqLimitConfig(FREQ_LOW_GHZ, FREQ_HIGH_GHZ)
sleep()

ar1.RfSetCalMonFreqTxPowLimitConfig(
    FREQ_LOW_GHZ, FREQ_LOW_GHZ, FREQ_LOW_GHZ,
    FREQ_HIGH_GHZ, FREQ_HIGH_GHZ, FREQ_HIGH_GHZ,
    0, 0, 0
)
sleep()

if RUN_RF_INIT then
    log("RfInit", "yellow")
    ar1.RfInit()
    sleep_ms(1000)
else
    warn("RfInit skipped.")
end

------------------------------------------------------------
-- 4. DATA CONFIG / LVDS
------------------------------------------------------------

section("4. Data config / LVDS")

log("DataPathConfig", "yellow")
ar1.DataPathConfig(
    DATA_PATH_CONFIG_ARG0,
    DATA_PATH_CONFIG_ARG1,
    DATA_PATH_CONFIG_ARG2
)
sleep()

log("LvdsClkConfig: DDR, 600 Mbps", "yellow")
ar1.LvdsClkConfig(
    LVDS_CLK_ARG0,
    LVDS_CLK_ARG1
)
sleep()

log("LVDSLaneConfig: 4 lanes enabled", "yellow")
ar1.LVDSLaneConfig(
    LANE_FORMAT,
    LANE1_ENABLE,
    LANE2_ENABLE,
    LANE3_ENABLE,
    LANE4_ENABLE,
    MSB_FIRST,
    CRC_ENABLE,
    PACKET_END_PULSE_ENABLE
)
sleep()

------------------------------------------------------------
-- 5. SENSOR CONFIG: 3Tx SIMULTANEOUS BEAM CONTROL
------------------------------------------------------------

section("5. Sensor config: 3Tx simultaneous beam-control live")

log("ProfileConfig", "yellow")
ar1.ProfileConfig(
    PROFILE_ID,
    START_FREQ_GHZ,
    IDLE_TIME_US,
    ADC_START_TIME_US,
    RAMP_END_TIME_US,
    TX_OUT_POWER_BACKOFF_CODE,
    TX_OUT_POWER_BACKOFF_CODE,
    TX_OUT_POWER_BACKOFF_CODE,
    PROFILE_TX_PHASE_SHIFTER,
    PROFILE_TX_PHASE_SHIFTER,
    PROFILE_TX_PHASE_SHIFTER,
    FREQ_SLOPE_CONST_MHZ_PER_US,
    TX_START_TIME_US,
    NUM_ADC_SAMPLES,
    DIG_OUT_SAMPLE_RATE_KSPS,
    HPF_CORNER_FREQ1,
    HPF_CORNER_FREQ2,
    RX_GAIN_CODE
)
sleep()

log("ChirpConfig: chirp 0 -> Tx0 + Tx1 + Tx2 simultaneous", "yellow")
ar1.ChirpConfig(
    CHIRP_START_INDEX,
    CHIRP_END_INDEX,
    PROFILE_ID,
    CHIRP_FREQ_VAR,
    CHIRP_SLOPE_VAR,
    CHIRP_IDLE_VAR,
    CHIRP_ADC_START_VAR,
    CHIRP_TX0_ENABLE,
    CHIRP_TX1_ENABLE,
    CHIRP_TX2_ENABLE
)
sleep()

log("Disable test source", "yellow")
ar1.DisableTestSource(0)
sleep()

log("FrameConfig: infinite frame, chirp0 only, 64 loops, 20 ms", "yellow")
ar1.FrameConfig(
    START_CHIRP,
    END_CHIRP,
    NUM_FRAMES,
    NUM_CHIRP_LOOPS,
    FRAME_PERIODICITY_MS,
    TRIGGER_DELAY_US,
    TRIGGER_SELECT
)
sleep()

log("Initial SetPerChirpPhaseShifterConfig: chirp0 phase [0,0,0]", "yellow")
ar1.SetPerChirpPhaseShifterConfig(
    START_CHIRP,
    END_CHIRP,
    INITIAL_TX0_PHASE_CODE,
    INITIAL_TX1_PHASE_CODE,
    INITIAL_TX2_PHASE_CODE
)
sleep()

------------------------------------------------------------
-- SUMMARY
------------------------------------------------------------

section("Config finished")
log("RadarBox 3Tx simultaneous beam-control config is ready.", "green")
log("This script did NOT start DCA1000 record and did NOT start radar frame.", "green")
log("Next: start Python UDP receiver, then run radarbox_run_record_startframe_beam_poll.lua.", "green")
print_expected_performance()
log("If UDP packet loss occurs, try PACKET_DELAY_US = 50.", "yellow")
log("Python receiver must use tx_groups=1 because 3Tx simultaneous raw ADC is still 4Rx.", "yellow")
log("============================================================", "green")
