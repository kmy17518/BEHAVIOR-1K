// Optional MANUS Integrated SDK acquisition sidecar for dex_teleop.
//
// Powered by MANUS. The proprietary MANUS SDK is supplied separately and is
// deliberately not vendored into this repository.
// This software contains source code provided by Manus Technology Group B.V.

#include <ManusSDK.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <unistd.h>

#include "manus_lifecycle.h"

namespace
{

using dex_teleop::manus::ConnectionEpochGuard;

constexpr const char* kProtocol = "dex_teleop.manus";
constexpr uint32_t kProtocolVersion = 2;
constexpr const char* kCoreWorldFrame = "manus_core_world_y_up_rh_z_to_viewer_m";

enum class BridgeMode
{
    Integrated,
    Remote,
};

#if defined(DEX_TELEOP_MANUS_LINK_MODE_REMOTE)
constexpr BridgeMode kLinkedMode = BridgeMode::Remote;
#elif defined(DEX_TELEOP_MANUS_LINK_MODE_INTEGRATED)
constexpr BridgeMode kLinkedMode = BridgeMode::Integrated;
#else
#error "Build with an explicit DEX_TELEOP_MANUS_LINK_MODE_* definition"
#endif

std::atomic<bool> g_stop_requested{ false };

void request_stop(int)
{
    g_stop_requested.store(true);
}

std::string json_escape(const std::string& value)
{
    std::ostringstream output;
    for (const unsigned char character : value)
    {
        switch (character)
        {
        case '"':
            output << "\\\"";
            break;
        case '\\':
            output << "\\\\";
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            if (character < 0x20)
            {
                output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<unsigned int>(character) << std::dec;
            }
            else
            {
                output << character;
            }
        }
    }
    return output.str();
}

std::string side_name(Side side)
{
    switch (side)
    {
    case Side_Left:
        return "left";
    case Side_Right:
        return "right";
    default:
        return "invalid";
    }
}

std::string mode_name(BridgeMode mode)
{
    return mode == BridgeMode::Remote ? "remote" : "integrated";
}

std::string mode_display_name(BridgeMode mode)
{
    return mode == BridgeMode::Remote ? "Remote" : "Integrated";
}

std::string hand_motion_name(HandMotion motion)
{
    switch (motion)
    {
    case HandMotion_None:
        return "none";
    case HandMotion_IMU:
        return "imu";
    case HandMotion_Tracker:
        return "tracker";
    case HandMotion_Tracker_RotationOnly:
        return "tracker_rotation_only";
    case HandMotion_Auto:
        return "auto";
    default:
        return "unknown";
    }
}

std::string tracker_type_name(TrackerType type)
{
    switch (type)
    {
    case TrackerType_LeftHand:
        return "left_hand";
    case TrackerType_RightHand:
        return "right_hand";
    case TrackerType_Head:
        return "head";
    case TrackerType_Waist:
        return "waist";
    case TrackerType_LeftFoot:
        return "left_foot";
    case TrackerType_RightFoot:
        return "right_foot";
    case TrackerType_Controller:
        return "controller";
    case TrackerType_Camera:
        return "camera";
    default:
        return "unknown";
    }
}

std::string tracking_quality_name(TrackingQuality quality)
{
    switch (quality)
    {
    case TrackingQuality_Untrackable:
        return "untrackable";
    case TrackingQuality_BadTracking:
        return "bad";
    case TrackingQuality_Trackable:
        return "trackable";
    default:
        return "unknown";
    }
}

std::string tracker_system_name(TrackerSystemType type)
{
    switch (type)
    {
    case TrackerSystemType_OpenVR:
        return "openvr";
    case TrackerSystemType_OpenXR:
        return "openxr";
    case TrackerSystemType_ART:
        return "art";
    case TrackerSystemType_Optitrack:
        return "optitrack";
    case TrackerSystemType_Vicon:
        return "vicon";
    case TrackerSystemType_Antilatency:
        return "antilatency";
    default:
        return "unknown";
    }
}

std::string version_string(const Version& version)
{
    std::ostringstream output;
    output << version.major << '.' << version.minor << '.' << version.patch;
    return output.str();
}

std::string lower_copy(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return value;
}

std::string chain_name(ChainType chain)
{
    switch (chain)
    {
    case ChainType_Hand:
        return "hand";
    case ChainType_FingerThumb:
        return "thumb";
    case ChainType_FingerIndex:
        return "index";
    case ChainType_FingerMiddle:
        return "middle";
    case ChainType_FingerRing:
        return "ring";
    case ChainType_FingerPinky:
        return "little";
    default:
        return "invalid";
    }
}

std::string joint_name(FingerJointType joint)
{
    switch (joint)
    {
    case FingerJointType_Metacarpal:
        return "metacarpal";
    case FingerJointType_Proximal:
        return "proximal";
    case FingerJointType_Intermediate:
        return "intermediate";
    case FingerJointType_Distal:
        return "distal";
    case FingerJointType_Tip:
        return "tip";
    default:
        return "invalid";
    }
}

uint64_t monotonic_now_ns()
{
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

struct Options
{
    int protocol_fd = STDOUT_FILENO;
    BridgeMode mode = BridgeMode::Integrated;
    uint32_t connect_timeout_seconds = 15;
    uint32_t glove_timeout_seconds = 15;
    uint32_t reconnect_timeout_seconds = 15;
    uint32_t discovery_wait_seconds = 1;
    std::string settings_dir;
    std::string log_dir;
    std::string left_calibration;
    std::string right_calibration;
    std::string core_host;
    Side required_side = Side_Right;
    HandMotion hand_motion = HandMotion_Auto;
    bool loopback_only = false;
    bool tracker_diagnostics = false;
    bool tracker_diagnostics_explicit = false;
    bool print_protocol_version = false;
};

std::string home_directory()
{
    const char* home = std::getenv("HOME");
    return home == nullptr ? "." : home;
}

std::string sdk_directory_path(std::string path)
{
    std::replace(path.begin(), path.end(), '\\', '/');
    if (!path.empty() && path.back() != '/')
    {
        path.push_back('/');
    }
    return path;
}

uint32_t parse_timeout(const char* value, const char* option)
{
    try
    {
        const unsigned long parsed = std::stoul(value);
        if (parsed > 3600)
        {
            throw std::out_of_range("timeout too large");
        }
        return static_cast<uint32_t>(parsed);
    }
    catch (const std::exception&)
    {
        throw std::runtime_error(std::string(option) + " requires an integer in [0, 3600]");
    }
}

BridgeMode parse_mode(const char* value)
{
    const std::string mode = lower_copy(value);
    if (mode == "integrated")
    {
        return BridgeMode::Integrated;
    }
    if (mode == "remote")
    {
        return BridgeMode::Remote;
    }
    throw std::runtime_error("--mode requires integrated or remote");
}

HandMotion parse_hand_motion(const char* value)
{
    const std::string motion = lower_copy(value);
    if (motion == "none")
    {
        return HandMotion_None;
    }
    if (motion == "imu")
    {
        return HandMotion_IMU;
    }
    if (motion == "tracker")
    {
        return HandMotion_Tracker;
    }
    if (motion == "tracker_rotation_only")
    {
        return HandMotion_Tracker_RotationOnly;
    }
    if (motion == "auto")
    {
        return HandMotion_Auto;
    }
    throw std::runtime_error(
        "--hand-motion requires auto, tracker, tracker_rotation_only, imu, or none");
}

Side parse_required_side(const char* value)
{
    const std::string side = lower_copy(value);
    if (side == "left")
    {
        return Side_Left;
    }
    if (side == "right")
    {
        return Side_Right;
    }
    throw std::runtime_error("--required-hand requires left or right");
}

Options parse_options(int argc, char** argv)
{
    Options options;
    options.settings_dir = home_directory() + "/.config/manus/";
    options.log_dir = home_directory() + "/.local/state/dex_teleop/manus/";

    for (int index = 1; index < argc; ++index)
    {
        const std::string argument = argv[index];
        auto require_value = [&](const char* name) -> const char* {
            if (++index >= argc)
            {
                throw std::runtime_error(std::string(name) + " requires a value");
            }
            return argv[index];
        };

        if (argument == "--protocol-fd")
        {
            options.protocol_fd = static_cast<int>(parse_timeout(require_value("--protocol-fd"), "--protocol-fd"));
        }
        else if (argument == "--mode")
        {
            options.mode = parse_mode(require_value("--mode"));
        }
        else if (argument == "--connect-timeout")
        {
            options.connect_timeout_seconds = parse_timeout(require_value("--connect-timeout"), "--connect-timeout");
        }
        else if (argument == "--glove-timeout")
        {
            options.glove_timeout_seconds = parse_timeout(require_value("--glove-timeout"), "--glove-timeout");
        }
        else if (argument == "--reconnect-timeout")
        {
            options.reconnect_timeout_seconds =
                parse_timeout(require_value("--reconnect-timeout"), "--reconnect-timeout");
        }
        else if (argument == "--discovery-wait")
        {
            options.discovery_wait_seconds =
                parse_timeout(require_value("--discovery-wait"), "--discovery-wait");
        }
        else if (argument == "--core-host")
        {
            options.core_host = require_value("--core-host");
        }
        else if (argument == "--required-hand")
        {
            options.required_side = parse_required_side(require_value("--required-hand"));
        }
        else if (argument == "--loopback-only")
        {
            options.loopback_only = true;
        }
        else if (argument == "--hand-motion")
        {
            options.hand_motion = parse_hand_motion(require_value("--hand-motion"));
        }
        else if (argument == "--tracker-diagnostics")
        {
            options.tracker_diagnostics = true;
            options.tracker_diagnostics_explicit = true;
        }
        else if (argument == "--no-tracker-diagnostics")
        {
            options.tracker_diagnostics = false;
            options.tracker_diagnostics_explicit = true;
        }
        else if (argument == "--settings-dir")
        {
            options.settings_dir = require_value("--settings-dir");
        }
        else if (argument == "--log-dir")
        {
            options.log_dir = require_value("--log-dir");
        }
        else if (argument == "--left-calibration")
        {
            options.left_calibration = require_value("--left-calibration");
        }
        else if (argument == "--right-calibration")
        {
            options.right_calibration = require_value("--right-calibration");
        }
        else if (argument == "--protocol-version")
        {
            options.print_protocol_version = true;
        }
        else if (argument == "--help" || argument == "-h")
        {
            std::cout << "Usage: manus_bridge [options]\n"
                         "  --protocol-fd FD         dedicated JSON-lines output descriptor\n"
                         "  --mode MODE              integrated (default) or remote\n"
                         "  --connect-timeout SEC    host discovery/connect timeout\n"
                         "  --glove-timeout SEC      timeout waiting for a valid glove topology\n"
                         "  --reconnect-timeout SEC  Remote reconnect deadline (0 disables)\n"
                         "  --discovery-wait SEC     seconds per vendor host-discovery request\n"
                         "  --core-host VALUE        exact Core IP/name or sorted zero-based index\n"
                         "  --required-hand SIDE     hand that must requalify before streaming\n"
                         "  --loopback-only          restrict Remote discovery to this host\n"
                         "  --hand-motion MODE       auto, tracker, tracker_rotation_only, imu, none\n"
                         "  --[no-]tracker-diagnostics emit optional tracker stream diagnostics\n"
                         "  --settings-dir PATH      Integrated Core settings directory\n"
                         "  --log-dir PATH           Integrated Core log directory\n"
                         "  --left-calibration PATH  optional left .mcal file\n"
                         "  --right-calibration PATH optional right .mcal file\n"
                         "  --protocol-version       print the protocol version without device access\n";
            std::exit(0);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }
    if (!options.tracker_diagnostics_explicit)
    {
        options.tracker_diagnostics = options.mode == BridgeMode::Remote;
    }
    if (options.discovery_wait_seconds == 0)
    {
        throw std::runtime_error("--discovery-wait must be at least 1 second");
    }
    if (!options.print_protocol_version && options.mode != kLinkedMode)
    {
        throw std::runtime_error(
            "Requested --mode " + mode_name(options.mode) + " but this executable was linked for " +
            mode_name(kLinkedMode) +
            "; build the matching variant with build_manus_bridge.sh --mode " + mode_name(options.mode));
    }
    if (options.mode == BridgeMode::Integrated && !options.core_host.empty())
    {
        throw std::runtime_error("--core-host is only valid in --mode remote");
    }
    options.settings_dir = sdk_directory_path(std::move(options.settings_dir));
    options.log_dir = sdk_directory_path(std::move(options.log_dir));
    return options;
}

std::vector<unsigned char> read_binary_file(const std::string& path)
{
    if (path.empty())
    {
        return {};
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        throw std::runtime_error("Could not open MANUS calibration file: " + path);
    }
    const std::streamsize size = input.tellg();
    if (size <= 0 || static_cast<uint64_t>(size) > UINT32_MAX)
    {
        throw std::runtime_error("Invalid MANUS calibration file size: " + path);
    }
    std::vector<unsigned char> bytes(static_cast<size_t>(size));
    input.seekg(0, std::ios::beg);
    if (!input.read(reinterpret_cast<char*>(bytes.data()), size))
    {
        throw std::runtime_error("Could not read MANUS calibration file: " + path);
    }
    return bytes;
}

SDKReturnCode raw_skeleton_node_count(uint32_t glove_id, uint32_t& node_count)
{
#if defined(__aarch64__) || defined(__arm__) || defined(_M_ARM64) || defined(_M_ARM)
    return CoreSdk_GetRawSkeletonNodeCount(glove_id, &node_count);
#else
    return CoreSdk_GetRawSkeletonNodeCount(glove_id, node_count);
#endif
}

class Bridge
{
public:
    explicit Bridge(Options options)
        : m_options(std::move(options)), m_left_calibration(read_binary_file(m_options.left_calibration)),
          m_right_calibration(read_binary_file(m_options.right_calibration))
    {
    }

    ~Bridge()
    {
        shutdown();
    }

    int run()
    {
        m_connection_generation.store(1);
        emit_status(
            "starting",
            "Powered by MANUS; initializing SDK " + mode_name(m_options.mode) +
                " (linked variant " + mode_name(kLinkedMode) + ")");
        if (!initialize_sdk())
        {
            return m_terminal_exit_code.load() == 0 ? 10 : m_terminal_exit_code.load();
        }

        if (!connect())
        {
            return 14;
        }
        const auto metadata_deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(m_options.connect_timeout_seconds);
        if (!check_versions(metadata_deadline))
        {
            return 14;
        }
        if (!configure_hand_motion())
        {
            return 15;
        }
        emit_connected_status(false);

        const auto qualification_deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(m_options.glove_timeout_seconds);
        if (!wait_for_stream_qualification(qualification_deadline, false))
        {
            return m_terminal_exit_code.load() == 0 ? 15 : m_terminal_exit_code.load();
        }
        while (!g_stop_requested.load())
        {
            std::unique_lock<std::mutex> lock(m_state_mutex);
            m_state_changed.wait_for(lock, std::chrono::milliseconds(250), [&] {
                return m_disconnect_pending || g_stop_requested.load();
            });
            const bool reconnect = m_disconnect_pending;
            m_disconnect_pending = false;
            lock.unlock();
            if (!reconnect || g_stop_requested.load())
            {
                continue;
            }
            if (m_options.mode != BridgeMode::Remote || m_options.reconnect_timeout_seconds == 0)
            {
                m_terminal_exit_code.store(17);
                fail(
                    "host_disconnected",
                    "MANUS " + mode_display_name(m_options.mode) +
                        " host disconnected; automatic reconnect is disabled for this mode",
                    std::nullopt,
                    17);
                g_stop_requested.store(true);
                break;
            }
            if (!restart_and_reconnect())
            {
                m_terminal_exit_code.store(17);
                g_stop_requested.store(true);
                break;
            }
        }
        emit_status("stopping", "Stop requested");
        return m_terminal_exit_code.load();
    }

private:
    struct GloveState
    {
        Side side = Side_Invalid;
        std::vector<NodeInfo> topology;
        bool calibration_attempted = false;
    };

    static Bridge* s_instance;

    void emit(const std::string& payload)
    {
        const std::string line = payload + "\n";
        std::lock_guard<std::mutex> lock(m_output_mutex);
        size_t offset = 0;
        while (offset < line.size())
        {
            const ssize_t written = ::write(m_options.protocol_fd, line.data() + offset, line.size() - offset);
            if (written > 0)
            {
                offset += static_cast<size_t>(written);
                continue;
            }
            if (written < 0 && errno == EINTR)
            {
                continue;
            }
            g_stop_requested.store(true);
            return;
        }
    }

    std::string envelope(const char* type) const
    {
        std::ostringstream output;
        output << "{\"protocol\":\"" << kProtocol << "\",\"version\":" << kProtocolVersion
               << ",\"type\":\"" << type << "\",\"mode\":\"" << mode_name(m_options.mode)
               << "\",\"linked_mode\":\"" << mode_name(kLinkedMode)
               << "\",\"connection_generation\":" << m_connection_generation.load();
        return output.str();
    }

    void emit_status(const std::string& state, const std::string& message)
    {
        std::ostringstream output;
        output << envelope("status") << ",\"state\":\"" << json_escape(state) << "\",\"message\":\""
               << json_escape(message) << "\"}";
        emit(output.str());
    }

    int fail(const std::string& code, const std::string& message, std::optional<SDKReturnCode> sdk_code, int exit_code)
    {
        std::ostringstream output;
        output << envelope("error") << ",\"code\":\"" << json_escape(code) << "\",\"message\":\""
               << json_escape(message) << "\"";
        if (sdk_code.has_value())
        {
            output << ",\"sdk_return_code\":" << static_cast<int>(*sdk_code);
        }
        output << "}";
        emit(output.str());
        std::cerr << "[manus_bridge] " << message;
        if (sdk_code.has_value())
        {
            std::cerr << " (SDK code " << static_cast<int>(*sdk_code) << ")";
        }
        std::cerr << std::endl;
        return exit_code;
    }

    bool initialize_sdk()
    {
        const SDKReturnCode initialize_result = m_options.mode == BridgeMode::Remote
                                                    ? CoreSdk_InitializeCore()
                                                    : CoreSdk_InitializeIntegrated();
        if (initialize_result != SDKReturnCode_Success)
        {
            const std::string initialize_call =
                m_options.mode == BridgeMode::Remote ? "CoreSdk_InitializeCore" : "CoreSdk_InitializeIntegrated";
            fail(
                "sdk_initialization_failed",
                initialize_call + " failed for the explicitly linked " + mode_name(kLinkedMode) + " library",
                initialize_result,
                10);
            return false;
        }
        m_sdk_initialized = true;

        // The v3.1.1 vendor Remote client proceeds directly from
        // CoreSdk_InitializeCore to callback/coordinate registration.
        // Settings and log locations configure embedded Core Integrated and
        // return FunctionNotAvailable from the Remote library.
        if (m_options.mode == BridgeMode::Integrated)
        {
            std::filesystem::create_directories(m_options.settings_dir);
            std::filesystem::create_directories(m_options.log_dir);
            if (const SDKReturnCode result = CoreSdk_SetSessionType(SessionType_CoreSDK);
                result != SDKReturnCode_Success)
            {
                fail("session_configuration_failed", "CoreSdk_SetSessionType failed", result, 11);
                return false;
            }
            if (const SDKReturnCode result = CoreSdk_SetSettingsLocation(m_options.settings_dir.c_str());
                result != SDKReturnCode_Success)
            {
                fail("settings_configuration_failed", "CoreSdk_SetSettingsLocation failed", result, 11);
                return false;
            }
            if (const SDKReturnCode result = CoreSdk_SetLogLocation(m_options.log_dir.c_str());
                result != SDKReturnCode_Success)
            {
                fail("log_configuration_failed", "CoreSdk_SetLogLocation failed", result, 11);
                return false;
            }
        }

        s_instance = this;
        if (!register_callbacks())
        {
            return false;
        }

        CoordinateSystemVUH coordinates;
        CoordinateSystemVUH_Init(&coordinates);
        coordinates.handedness = Side_Right;
        coordinates.up = AxisPolarity_PositiveY;
        coordinates.view = AxisView_ZToViewer;
        coordinates.unitScale = 1.0f;
        if (const SDKReturnCode result = CoreSdk_InitializeCoordinateSystemWithVUH(coordinates, true);
            result != SDKReturnCode_Success)
        {
            fail("coordinate_configuration_failed", "Could not configure Y-up right-handed meters", result, 13);
            return false;
        }
        return true;
    }

    bool register_callbacks()
    {
        std::vector<std::pair<const char*, SDKReturnCode>> results = {
            { "raw skeleton", CoreSdk_RegisterCallbackForRawSkeletonStream(on_raw_skeleton) },
            { "landscape", CoreSdk_RegisterCallbackForLandscapeStream(on_landscape) },
            { "connect", CoreSdk_RegisterCallbackForOnConnect(on_connect) },
            { "disconnect", CoreSdk_RegisterCallbackForOnDisconnect(on_disconnect) },
        };
        if (m_options.tracker_diagnostics)
        {
            results.emplace_back("tracker", CoreSdk_RegisterCallbackForTrackerStream(on_tracker));
        }
        for (const auto& [name, result] : results)
        {
            if (result != SDKReturnCode_Success)
            {
                fail("callback_registration_failed", std::string("Could not register ") + name + " callback", result, 12);
                return false;
            }
        }
        m_callbacks_registered = true;
        return true;
    }

    void unregister_callbacks()
    {
        if (!m_callbacks_registered)
        {
            return;
        }
        CoreSdk_RegisterCallbackForRawSkeletonStream(nullptr);
        CoreSdk_RegisterCallbackForLandscapeStream(nullptr);
        CoreSdk_RegisterCallbackForOnConnect(nullptr);
        CoreSdk_RegisterCallbackForOnDisconnect(nullptr);
        if (m_options.tracker_diagnostics)
        {
            CoreSdk_RegisterCallbackForTrackerStream(nullptr);
        }
        m_callbacks_registered = false;
    }

    static bool host_less(const ManusHost& left, const ManusHost& right)
    {
        const std::string left_name = lower_copy(left.hostName);
        const std::string right_name = lower_copy(right.hostName);
        if (left_name != right_name)
        {
            return left_name < right_name;
        }
        const std::string left_ip = left.ipAddress;
        const std::string right_ip = right.ipAddress;
        if (left_ip != right_ip)
        {
            return left_ip < right_ip;
        }
        if (left.manusCoreVersion.major != right.manusCoreVersion.major)
        {
            return left.manusCoreVersion.major < right.manusCoreVersion.major;
        }
        if (left.manusCoreVersion.minor != right.manusCoreVersion.minor)
        {
            return left.manusCoreVersion.minor < right.manusCoreVersion.minor;
        }
        return left.manusCoreVersion.patch < right.manusCoreVersion.patch;
    }

    void emit_hosts(const std::vector<ManusHost>& hosts)
    {
        const bool loopback_only =
            m_options.mode == BridgeMode::Integrated || m_options.loopback_only;
        std::ostringstream output;
        output << envelope("hosts") << ",\"selector\":\"" << json_escape(m_options.core_host)
               << "\",\"loopback_only\":" << (loopback_only ? "true" : "false")
               << ",\"hosts\":[";
        for (size_t index = 0; index < hosts.size(); ++index)
        {
            if (index != 0)
            {
                output << ',';
            }
            const ManusHost& host = hosts[index];
            output << "{\"index\":" << index << ",\"name\":\"" << json_escape(host.hostName)
                   << "\",\"ip\":\"" << json_escape(host.ipAddress) << "\",\"version\":\""
                   << json_escape(version_string(host.manusCoreVersion)) << "\"}";
        }
        output << "]}";
        emit(output.str());
    }

    std::optional<size_t> select_host(const std::vector<ManusHost>& hosts)
    {
        if (hosts.empty())
        {
            return std::nullopt;
        }
        if (m_options.core_host.empty())
        {
            if (m_options.mode == BridgeMode::Remote && hosts.size() != 1)
            {
                fail(
                    "core_host_ambiguous",
                    "Remote discovery found " + std::to_string(hosts.size()) +
                        " Core hosts; select one with --core-host by exact IP, exact name, or sorted index",
                    std::nullopt,
                    14);
                m_terminal_exit_code.store(14);
                return std::nullopt;
            }
            return 0;
        }

        const std::string selector = m_options.core_host;
        const std::string folded_selector = lower_copy(selector);
        std::vector<size_t> matches;
        for (size_t index = 0; index < hosts.size(); ++index)
        {
            if (selector == hosts[index].ipAddress || folded_selector == lower_copy(hosts[index].hostName))
            {
                matches.push_back(index);
            }
        }
        if (matches.size() == 1)
        {
            return matches.front();
        }
        if (matches.size() > 1)
        {
            fail(
                "core_host_ambiguous",
                "Requested Core host name " + selector +
                    " matched more than one discovered Core; use an exact IP or sorted index",
                std::nullopt,
                14);
            m_terminal_exit_code.store(14);
            return std::nullopt;
        }

        const bool all_digits = !selector.empty() &&
                                std::all_of(selector.begin(), selector.end(), [](unsigned char value) {
                                    return std::isdigit(value) != 0;
                                });
        if (all_digits)
        {
            try
            {
                const size_t index = static_cast<size_t>(std::stoul(selector));
                if (index < hosts.size())
                {
                    return index;
                }
            }
            catch (const std::exception&)
            {
            }
            fail(
                "core_host_not_found",
                "Requested sorted Core host index " + selector + " is outside the discovered host list",
                std::nullopt,
                14);
            m_terminal_exit_code.store(14);
            return std::nullopt;
        }

        fail(
            "core_host_not_found",
            "Requested Core host " + selector +
                " did not match any discovered Core host; use an exact IP, exact name, or sorted index",
            std::nullopt,
            14);
        m_terminal_exit_code.store(14);
        return std::nullopt;
    }

    bool wait_for_connect_callback(
        std::chrono::steady_clock::time_point deadline,
        uint64_t generation)
    {
        std::unique_lock<std::mutex> lock(m_state_mutex);
        return m_state_changed.wait_until(lock, deadline, [&] {
            return g_stop_requested.load() ||
                   ConnectionEpochGuard::callback_is_current(
                       generation,
                       m_connection_generation.load(),
                       m_connect_callback_generation,
                       m_restarting_sdk);
        }) &&
               !g_stop_requested.load() &&
               ConnectionEpochGuard::callback_is_current(
                   generation,
                   m_connection_generation.load(),
                   m_connect_callback_generation,
                   m_restarting_sdk);
    }

    bool connect()
    {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(m_options.connect_timeout_seconds);
        const uint64_t generation = m_connection_generation.load();
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            m_connect_callback_generation = 0;
        }
        if (m_options.mode == BridgeMode::Integrated)
        {
            ManusHost empty_host;
            ManusHost_Init(&empty_host);
            const SDKReturnCode result = CoreSdk_ConnectToHost(empty_host);
            if (result == SDKReturnCode_Success &&
                wait_for_connect_callback(deadline, generation))
            {
                return true;
            }
            fail(
                "host_connection_failed",
                "MANUS Integrated did not deliver its on-connect callback before the " +
                    std::to_string(m_options.connect_timeout_seconds) +
                    " second timeout",
                result,
                14);
            return false;
        }

        const bool loopback_only =
            m_options.loopback_only;
        SDKReturnCode last_result = SDKReturnCode_NotConnected;
        do
        {
            last_result = CoreSdk_LookForHosts(m_options.discovery_wait_seconds, loopback_only);
            if (last_result == SDKReturnCode_Success)
            {
                uint32_t count = 0;
                last_result = CoreSdk_GetNumberOfAvailableHostsFound(&count);
                if (last_result == SDKReturnCode_Success && count > 0)
                {
                    std::vector<ManusHost> hosts(count);
                    last_result = CoreSdk_GetAvailableHostsFound(hosts.data(), count);
                    if (last_result == SDKReturnCode_Success)
                    {
                        std::sort(hosts.begin(), hosts.end(), host_less);
                        emit_hosts(hosts);
                        const std::optional<size_t> selected_index = select_host(hosts);
                        if (m_terminal_exit_code.load() != 0)
                        {
                            return false;
                        }
                        if (selected_index.has_value())
                        {
                            m_selected_host = hosts[*selected_index];
                            last_result = CoreSdk_ConnectToHost(*m_selected_host);
                            if (last_result == SDKReturnCode_Success &&
                                wait_for_connect_callback(deadline, generation))
                            {
                                return true;
                            }
                        }
                    }
                }
            }
            if (g_stop_requested.load())
            {
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        } while (std::chrono::steady_clock::now() < deadline);

        const std::string selector =
            m_options.core_host.empty() ? "automatic single-host selection" : m_options.core_host;
        fail(
            "host_connection_failed",
            "Could not discover and connect to a MANUS " + mode_name(m_options.mode) +
                " host before the " + std::to_string(m_options.connect_timeout_seconds) +
                " second timeout (selector: " + selector +
                "). Verify Core mode, host/network discovery, firewall policy, and matching SDK/Core versions; "
                "no network port is assumed by dex_teleop",
            last_result,
            14);
        return false;
    }

    bool check_versions(std::chrono::steady_clock::time_point deadline)
    {
        SDKReturnCode last_result = SDKReturnCode_NotConnected;
        do
        {
            ManusVersion sdk_version{};
            ManusVersion core_version{};
            bool compatible = false;
            last_result =
                CoreSdk_GetVersionsAndCheckCompatibility(&sdk_version, &core_version, &compatible);
            uint32_t session_id = 0;
            const SDKReturnCode session_result = CoreSdk_GetSessionId(&session_id);
            const bool versions_ready =
                sdk_version.versionInfo[0] != '\0' && core_version.versionInfo[0] != '\0';
            if (last_result == SDKReturnCode_Success && versions_ready && !compatible)
            {
                fail(
                    "version_incompatible",
                    "MANUS SDK " + std::string(sdk_version.versionInfo) +
                        " is incompatible with Core " + core_version.versionInfo +
                        "; install matching SDK/Core major-minor releases",
                    std::nullopt,
                    14);
                return false;
            }
            const bool session_ready =
                m_options.mode == BridgeMode::Integrated ||
                (session_result == SDKReturnCode_Success && session_id != 0);
            if (last_result == SDKReturnCode_Success && versions_ready &&
                compatible && session_ready)
            {
                std::lock_guard<std::mutex> lock(m_state_mutex);
                m_sdk_version = sdk_version.versionInfo;
                m_core_version = core_version.versionInfo;
                m_versions_compatible = true;
                m_session_id = session_id;
                m_versions_qualified = true;
                return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        } while (!g_stop_requested.load() && std::chrono::steady_clock::now() < deadline);

        if (m_options.mode == BridgeMode::Integrated &&
            last_result == SDKReturnCode_Success)
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            // v3.1.1 Integrated can return empty versions and an unusable
            // compatibility bit even after its on-connect callback. Integrated
            // packages its matching Core implementation in the linked library;
            // preserve the missing metadata explicitly instead of inventing a
            // version or failing a valid local lifecycle.
            m_sdk_version = "unreported-by-integrated-sdk";
            m_core_version = m_landscape_core_version.empty()
                                 ? "unreported-by-integrated-sdk"
                                 : m_landscape_core_version;
            m_versions_compatible = true;
            m_versions_qualified = true;
            return true;
        }
        fail(
            "version_metadata_not_ready",
            "MANUS on-connect callback arrived, but non-empty compatible SDK/Core versions "
            "and required session metadata were not ready before the connection deadline",
            last_result,
            14);
        return false;
    }

    bool configure_hand_motion()
    {
        const SDKReturnCode result = CoreSdk_SetRawSkeletonHandMotion(m_options.hand_motion);
        if (result != SDKReturnCode_Success)
        {
            fail(
                "hand_motion_configuration_failed",
                "CoreSdk_SetRawSkeletonHandMotion(" + hand_motion_name(m_options.hand_motion) + ") failed",
                result,
                15);
            return false;
        }
        HandMotion actual = HandMotion_None;
        const SDKReturnCode read_result = CoreSdk_GetRawSkeletonHandMotion(&actual);
        if (read_result != SDKReturnCode_Success || actual != m_options.hand_motion)
        {
            fail(
                "hand_motion_verification_failed",
                "Core did not confirm requested raw-skeleton hand motion " +
                    hand_motion_name(m_options.hand_motion),
                read_result,
                15);
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            m_hand_motion_qualified = true;
        }
        return true;
    }

    std::string connection_fields_locked() const
    {
        std::ostringstream output;
        output << ",\"core_host_name\":\""
               << json_escape(m_selected_host.has_value() ? m_selected_host->hostName : "")
               << "\",\"core_host_ip\":\""
               << json_escape(m_selected_host.has_value() ? m_selected_host->ipAddress : "")
               << "\",\"sdk_version\":\"" << json_escape(m_sdk_version)
               << "\",\"core_version\":\"" << json_escape(m_core_version)
               << "\",\"versions_compatible\":" << (m_versions_compatible ? "true" : "false")
               << ",\"session_id\":" << m_session_id << ",\"hand_motion\":\""
               << hand_motion_name(m_options.hand_motion) << "\",\"core_world_frame\":\""
               << kCoreWorldFrame << "\"";
        return output.str();
    }

    std::string connection_fields()
    {
        std::lock_guard<std::mutex> lock(m_state_mutex);
        return connection_fields_locked();
    }

    void emit_connected_status(bool reconnected)
    {
        std::ostringstream output;
        output << envelope("status") << ",\"state\":\""
               << (reconnected ? "reconnected" : "sdk_connected") << "\",\"message\":\"MANUS "
               << mode_name(m_options.mode) << " host connected\"" << connection_fields() << "}";
        emit(output.str());
    }

    bool wait_for_stream_qualification(
        std::chrono::steady_clock::time_point deadline,
        bool reconnecting)
    {
        std::unique_lock<std::mutex> lock(m_state_mutex);
        const uint64_t generation = m_connection_generation.load();
        const bool qualified = m_state_changed.wait_until(lock, deadline, [&] {
            return g_stop_requested.load() || m_disconnect_pending ||
                   (m_connection_generation.load() == generation &&
                    ConnectionEpochGuard::qualification_complete(
                        m_versions_qualified,
                        m_hand_motion_qualified,
                        m_license_qualified,
                        m_required_topology_qualified,
                        m_required_frame_seen));
        });
        if (qualified && !g_stop_requested.load() && !m_disconnect_pending &&
            m_connection_generation.load() == generation &&
            ConnectionEpochGuard::qualification_complete(
                m_versions_qualified,
                m_hand_motion_qualified,
                m_license_qualified,
                m_required_topology_qualified,
                m_required_frame_seen))
        {
            std::ostringstream output;
            if (reconnecting)
            {
                output << envelope("status")
                       << ",\"state\":\"reconnected\",\"message\":\"MANUS Remote "
                          "connection and required-hand stream requalified\""
                       << connection_fields_locked() << "}";
            }
            else
            {
                output << envelope("status")
                       << ",\"state\":\"topology_ready\",\"message\":\"MANUS required-hand "
                          "license, topology, and raw stream qualified\""
                       << connection_fields_locked() << "}";
            }
            emit(output.str());
            m_streaming_enabled = true;
            m_recovering = false;
            return true;
        }
        if (g_stop_requested.load() || m_disconnect_pending)
        {
            return false;
        }
        const int exit_code = reconnecting ? 17 : 15;
        m_terminal_exit_code.store(exit_code);
        lock.unlock();
        fail(
            reconnecting ? "reconnect_stream_not_qualified" : "no_glove",
            "MANUS connection did not requalify license, required-hand topology, "
            "and a required-hand raw frame before the bounded deadline",
            std::nullopt,
            exit_code);
        return false;
    }

    void reset_qualification_locked()
    {
        m_topologies.clear();
        m_tracker_systems.clear();
        m_last_landscape_payload.clear();
        m_sdk_version.clear();
        m_core_version.clear();
        m_landscape_core_version.clear();
        m_session_id = 0;
        m_versions_compatible = false;
        m_versions_qualified = false;
        m_hand_motion_qualified = false;
        m_license_qualified = false;
        m_required_topology_qualified = false;
        m_required_frame_seen = false;
        m_streaming_enabled = false;
        m_connect_callback_generation = 0;
        m_license_error_emitted = false;
    }

    bool restart_and_reconnect()
    {
        ManusHost selected_host;
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            if (!m_selected_host.has_value())
            {
                fail(
                    "reconnect_failed",
                    "Cannot reconnect because no deterministic Core host was selected",
                    std::nullopt,
                    17);
                return false;
            }
            selected_host = *m_selected_host;
            m_restarting_sdk = true;
            reset_qualification_locked();
        }
        emit_status(
            "reconnecting",
            "Restarting the MANUS SDK before reconnecting to locked Core host " +
                std::string(selected_host.hostName) + " (" + selected_host.ipAddress + ")");
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::seconds(m_options.reconnect_timeout_seconds);

        unregister_callbacks();
        const SDKReturnCode shutdown_result = CoreSdk_ShutDown();
        m_sdk_initialized = false;
        if (shutdown_result != SDKReturnCode_Success)
        {
            fail(
                "sdk_restart_failed",
                "CoreSdk_ShutDown failed while preparing Remote reconnect",
                shutdown_result,
                17);
            return false;
        }
        if (!initialize_sdk())
        {
            fail(
                "sdk_restart_failed",
                "MANUS SDK reinitialization failed during Remote reconnect",
                std::nullopt,
                17);
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            m_restarting_sdk = false;
            m_selected_host = selected_host;
        }

        const uint64_t generation = m_connection_generation.load();
        SDKReturnCode connect_result = SDKReturnCode_NotConnected;
        bool connected = false;
        do
        {
            connect_result = CoreSdk_ConnectToHost(selected_host);
            if (connect_result == SDKReturnCode_Success)
            {
                connected = wait_for_connect_callback(deadline, generation);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        } while (!g_stop_requested.load() &&
                 std::chrono::steady_clock::now() < deadline);
        if (!connected)
        {
            fail(
                "reconnect_failed",
                "Restarted MANUS SDK did not reconnect to the locked Core host "
                "before the bounded deadline",
                connect_result,
                17);
            return false;
        }
        if (!check_versions(deadline) || !configure_hand_motion() ||
            !wait_for_stream_qualification(deadline, true))
        {
            return false;
        }
        return true;
    }

    bool apply_calibration(uint32_t glove_id, GloveState& state)
    {
        if (state.calibration_attempted)
        {
            return true;
        }
        state.calibration_attempted = true;
        const std::vector<unsigned char>& calibration =
            state.side == Side_Left ? m_left_calibration : m_right_calibration;
        if (calibration.empty())
        {
            return true;
        }
        SetGloveCalibrationReturnCode result = SetGloveCalibrationReturnCode_Error;
        const SDKReturnCode sdk_result = CoreSdk_SetGloveCalibration(
            glove_id, const_cast<unsigned char*>(calibration.data()), static_cast<uint32_t>(calibration.size()), &result);
        if (sdk_result != SDKReturnCode_Success || result != SetGloveCalibrationReturnCode_Success)
        {
            std::ostringstream message;
            message << "Calibration rejected for " << side_name(state.side) << " glove " << glove_id
                    << " (result " << static_cast<int>(result) << ")";
            fail("calibration_failed", message.str(), sdk_result, 0);
            return false;
        }
        return true;
    }

    void emit_landscape(const Landscape& landscape)
    {
        std::ostringstream output;
        output << envelope("landscape") << ",\"core_version\":\""
               << json_escape(version_string(landscape.settings.manusCoreVersion))
               << "\",\"license_sdk\":" << (landscape.settings.license.sdk ? "true" : "false")
               << ",\"license_integrated\":"
               << (landscape.settings.license.integrated ? "true" : "false") << ",\"trackers\":[";
        for (uint32_t index = 0; index < landscape.trackers.trackerCount; ++index)
        {
            if (index != 0)
            {
                output << ',';
            }
            const TrackerLandscapeData& tracker = landscape.trackers.trackers[index];
            output << "{\"id\":\"" << json_escape(tracker.id) << "\",\"type\":\""
                   << tracker_type_name(tracker.type) << "\",\"system\":\""
                   << tracker_system_name(tracker.systemType) << "\",\"user_id\":" << tracker.user
                   << ",\"is_hmd\":" << (tracker.isHMD ? "true" : "false")
                   << ",\"manufacturer\":\"" << json_escape(tracker.manufacturer)
                   << "\",\"product_name\":\"" << json_escape(tracker.productName) << "\"}";
        }
        output << "]}";
        const std::string payload = output.str();
        if (payload != m_last_landscape_payload)
        {
            m_last_landscape_payload = payload;
            emit(payload);
        }
    }

    void handle_landscape(const Landscape* landscape)
    {
        if (landscape == nullptr)
        {
            return;
        }
        const uint64_t generation = m_connection_generation.load();
        std::unique_lock<std::mutex> lock(m_state_mutex);
        if (!ConnectionEpochGuard::callback_is_current(
                generation,
                m_connection_generation.load(),
                m_connect_callback_generation,
                m_restarting_sdk))
        {
            return;
        }
        m_landscape_core_version = version_string(landscape->settings.manusCoreVersion);
        emit_landscape(*landscape);
        std::unordered_map<std::string, std::string> tracker_systems;
        for (uint32_t index = 0; index < landscape->trackers.trackerCount; ++index)
        {
            const TrackerLandscapeData& tracker = landscape->trackers.trackers[index];
            tracker_systems.emplace(tracker.id, tracker_system_name(tracker.systemType));
        }
        m_tracker_systems = std::move(tracker_systems);
        const DeviceLandscape& devices = landscape->gloveDevices;
        bool have_supported_glove = false;
        for (uint32_t index = 0; index < devices.gloveCount; ++index)
        {
            const Side side = devices.gloves[index].side;
            if (side == Side_Left || side == Side_Right)
            {
                have_supported_glove = true;
                break;
            }
        }
        // Core can publish an initial empty Landscape before device
        // enumeration completes. A false feature bit in that snapshot is not
        // license evidence; reject only after a supported glove is present.
        const bool license_available = m_options.mode == BridgeMode::Remote
                                           ? landscape->settings.license.sdk
                                           : landscape->settings.license.integrated;
        if (have_supported_glove && !license_available)
        {
            const bool should_emit = !m_license_error_emitted;
            m_license_error_emitted = true;
            if (should_emit)
            {
                m_terminal_exit_code.store(16);
                fail(
                    m_options.mode == BridgeMode::Remote ? "sdk_license_unavailable"
                                                        : "integrated_license_unavailable",
                    "The connected MANUS license does not enable the required " +
                        std::string(m_options.mode == BridgeMode::Remote ? "SDK" : "Integrated SDK") +
                        " feature",
                    std::nullopt,
                    16);
                g_stop_requested.store(true);
                m_state_changed.notify_all();
            }
            return;
        }
        if (have_supported_glove && license_available)
        {
            m_license_qualified = true;
        }
        std::vector<uint32_t> present;
        for (uint32_t index = 0; index < devices.gloveCount; ++index)
        {
            const GloveLandscapeData& glove = devices.gloves[index];
            if (glove.side != Side_Left && glove.side != Side_Right)
            {
                continue;
            }
            present.push_back(glove.id);

            uint32_t node_count = 0;
            if (raw_skeleton_node_count(glove.id, node_count) != SDKReturnCode_Success || node_count == 0)
            {
                continue;
            }
            std::vector<NodeInfo> topology(node_count);
            if (CoreSdk_GetRawSkeletonNodeInfoArray(glove.id, topology.data(), node_count) != SDKReturnCode_Success)
            {
                continue;
            }

            bool should_emit = false;
            GloveState& state = m_topologies[glove.id];
            state.side = glove.side;
            if (!apply_calibration(glove.id, state))
            {
                m_terminal_exit_code.store(16);
                g_stop_requested.store(true);
                m_state_changed.notify_all();
                return;
            }
            if (state.topology.size() != topology.size() ||
                !std::equal(topology.begin(), topology.end(), state.topology.begin(), [](const NodeInfo& left, const NodeInfo& right) {
                    return left.nodeId == right.nodeId && left.parentId == right.parentId &&
                           left.chainType == right.chainType && left.side == right.side &&
                           left.fingerJointType == right.fingerJointType;
                }))
            {
                state.topology = topology;
                should_emit = true;
            }
            if (should_emit)
            {
                emit_topology(glove.id, glove.side, topology);
            }
            if (glove.side == m_options.required_side)
            {
                m_required_topology_qualified = true;
            }
        }

        std::vector<std::pair<uint32_t, Side>> removed;
        for (auto iterator = m_topologies.begin(); iterator != m_topologies.end();)
        {
            if (std::find(present.begin(), present.end(), iterator->first) == present.end())
            {
                removed.emplace_back(iterator->first, iterator->second.side);
                iterator = m_topologies.erase(iterator);
            }
            else
            {
                ++iterator;
            }
        }
        for (const auto& [glove_id, side] : removed)
        {
            emit_glove_removed(glove_id, side);
            if (side == m_options.required_side)
            {
                m_required_topology_qualified = false;
                m_required_frame_seen = false;
                m_streaming_enabled = false;
            }
        }
        m_state_changed.notify_all();
    }

    void emit_glove_removed(uint32_t glove_id, Side side)
    {
        std::ostringstream output;
        output << envelope("glove_removed") << ",\"glove_id\":" << glove_id << ",\"side\":\""
               << side_name(side) << "\",\"reason\":\"landscape_absent\"}";
        emit(output.str());
    }

    void emit_topology(uint32_t glove_id, Side side, const std::vector<NodeInfo>& topology)
    {
        std::ostringstream output;
        output << envelope("topology") << ",\"glove_id\":" << glove_id << ",\"side\":\"" << side_name(side)
               << "\",\"transform_space\":\"sdk_global\",\"nodes\":[";
        for (size_t index = 0; index < topology.size(); ++index)
        {
            if (index != 0)
            {
                output << ',';
            }
            const NodeInfo& node = topology[index];
            output << "{\"id\":" << node.nodeId << ",\"parent_id\":" << node.parentId << ",\"side\":\""
                   << side_name(node.side) << "\",\"chain\":\"" << chain_name(node.chainType)
                   << "\",\"joint\":\"" << joint_name(node.fingerJointType) << "\"}";
        }
        output << "]}";
        emit(output.str());
    }

    void handle_raw_skeleton(const SkeletonStreamInfo* stream)
    {
        if (stream == nullptr)
        {
            return;
        }
        const uint64_t generation = m_connection_generation.load();
        const uint64_t callback_monotonic_ns = monotonic_now_ns();
        for (uint32_t index = 0; index < stream->skeletonsCount; ++index)
        {
            RawSkeletonInfo info{};
            if (CoreSdk_GetRawSkeletonInfo(index, &info) != SDKReturnCode_Success || info.nodesCount == 0)
            {
                continue;
            }
            std::vector<SkeletonNode> nodes(info.nodesCount);
            if (CoreSdk_GetRawSkeletonData(index, nodes.data(), info.nodesCount) != SDKReturnCode_Success)
            {
                continue;
            }

            std::unique_lock<std::mutex> lock(m_state_mutex);
            if (!ConnectionEpochGuard::callback_is_current(
                    generation,
                    m_connection_generation.load(),
                    m_connect_callback_generation,
                    m_restarting_sdk))
            {
                continue;
            }
            const auto state = m_topologies.find(info.gloveId);
            if (state == m_topologies.end())
            {
                continue;
            }
            const Side side = state->second.side;
            const size_t topology_size = state->second.topology.size();
            std::optional<uint32_t> wrist_node_id;
            for (const NodeInfo& node : state->second.topology)
            {
                if (node.chainType == ChainType_Hand)
                {
                    wrist_node_id = node.nodeId;
                    break;
                }
            }
            if (nodes.size() != topology_size || !wrist_node_id.has_value())
            {
                continue;
            }
            if (!ConnectionEpochGuard::sample_may_publish(
                    generation,
                    m_connection_generation.load(),
                    m_connect_callback_generation,
                    m_restarting_sdk,
                    m_streaming_enabled))
            {
                if (side == m_options.required_side &&
                    m_versions_qualified && m_hand_motion_qualified &&
                    m_license_qualified && m_required_topology_qualified)
                {
                    m_required_frame_seen = true;
                    m_state_changed.notify_all();
                }
                continue;
            }
            emit_sample(
                info.gloveId,
                side,
                stream->publishTime.time,
                callback_monotonic_ns,
                *wrist_node_id,
                nodes);
        }
    }

    void emit_sample(
        uint32_t glove_id,
        Side side,
        uint64_t manus_publish_time,
        uint64_t callback_monotonic_ns,
        uint32_t wrist_node_id,
        const std::vector<SkeletonNode>& nodes)
    {
        for (const SkeletonNode& node : nodes)
        {
            const ManusTransform& transform = node.transform;
            const float values[] = { transform.position.x, transform.position.y, transform.position.z,
                                     transform.rotation.x, transform.rotation.y, transform.rotation.z,
                                     transform.rotation.w };
            if (!std::all_of(std::begin(values), std::end(values), [](float value) { return std::isfinite(value); }))
            {
                return;
            }
        }

        const uint64_t sequence = m_sequence.fetch_add(1);
        std::ostringstream output;
        output << std::setprecision(9) << envelope("articulation") << ",\"sequence\":" << sequence
               << ",\"glove_id\":" << glove_id << ",\"side\":\"" << side_name(side)
               << "\",\"manus_publish_time\":" << manus_publish_time << ",\"capture_monotonic_ns\":"
               << callback_monotonic_ns << ",\"callback_capture_monotonic_ns\":"
               << callback_monotonic_ns << ",\"wrist_node_id\":" << wrist_node_id
               << connection_fields_locked() << ",\"nodes\":[";
        for (size_t index = 0; index < nodes.size(); ++index)
        {
            if (index != 0)
            {
                output << ',';
            }
            const SkeletonNode& node = nodes[index];
            const ManusTransform& transform = node.transform;
            output << "{\"id\":" << node.id << ",\"position\":[" << transform.position.x << ','
                   << transform.position.y << ',' << transform.position.z << "],\"orientation_xyzw\":["
                   << transform.rotation.x << ',' << transform.rotation.y << ',' << transform.rotation.z << ','
                   << transform.rotation.w << "]}";
        }
        output << "]}";
        emit(output.str());
    }

    void handle_tracker(const TrackerStreamInfo* stream)
    {
        if (stream == nullptr)
        {
            return;
        }
        const uint64_t generation = m_connection_generation.load();
        const uint64_t callback_monotonic_ns = monotonic_now_ns();
        std::vector<TrackerData> trackers;
        trackers.reserve(stream->trackerCount);
        for (uint32_t index = 0; index < stream->trackerCount; ++index)
        {
            TrackerData tracker{};
            if (CoreSdk_GetTrackerData(index, &tracker) == SDKReturnCode_Success)
            {
                trackers.push_back(tracker);
            }
        }

        std::unique_lock<std::mutex> lock(m_state_mutex);
        if (!ConnectionEpochGuard::callback_is_current(
                generation,
                m_connection_generation.load(),
                m_connect_callback_generation,
                m_restarting_sdk))
        {
            return;
        }
        std::ostringstream output;
        output << std::setprecision(9) << envelope("trackers") << ",\"sequence\":"
               << m_tracker_sequence.fetch_add(1) << ",\"manus_publish_time\":"
               << stream->publishTime.time << ",\"capture_monotonic_ns\":"
               << callback_monotonic_ns << connection_fields_locked() << ",\"trackers\":[";
        bool first = true;
        for (const TrackerData& tracker : trackers)
        {
            if (!first)
            {
                output << ',';
            }
            first = false;
            const float values[] = {
                tracker.position.x,
                tracker.position.y,
                tracker.position.z,
                tracker.rotation.x,
                tracker.rotation.y,
                tracker.rotation.z,
                tracker.rotation.w,
            };
            const bool pose_valid =
                std::all_of(std::begin(values), std::end(values), [](float value) {
                    return std::isfinite(value);
                });
            const std::string tracker_id = tracker.trackerId.id;
            const auto system = m_tracker_systems.find(tracker_id);
            output << "{\"id\":\"" << json_escape(tracker_id) << "\",\"user_id\":"
                   << tracker.userId << ",\"is_hmd\":" << (tracker.isHmd ? "true" : "false")
                   << ",\"type\":\"" << tracker_type_name(tracker.trackerType)
                   << "\",\"quality\":\"" << tracking_quality_name(tracker.quality)
                   << "\",\"tracking_system\":\""
                   << json_escape(
                          system == m_tracker_systems.end() ? "unknown" : system->second)
                   << "\",\"last_update_time\":" << tracker.lastUpdateTime.time
                   << ",\"pose_valid\":" << (pose_valid ? "true" : "false");
            if (pose_valid)
            {
                output << ",\"position\":[" << tracker.position.x << ',' << tracker.position.y << ','
                       << tracker.position.z << "],\"orientation_xyzw\":[" << tracker.rotation.x << ','
                       << tracker.rotation.y << ',' << tracker.rotation.z << ',' << tracker.rotation.w << ']';
            }
            output << '}';
        }
        output << "]}";
        emit(output.str());
    }

    void handle_connect(const ManusHost* host)
    {
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            if (m_stopping)
            {
                return;
            }
            if (host != nullptr && (host->hostName[0] != '\0' || host->ipAddress[0] != '\0'))
            {
                m_selected_host = *host;
            }
            m_connected.store(true);
            m_connect_callback_generation = m_connection_generation.load();
        }
        m_state_changed.notify_all();
    }

    void handle_disconnect()
    {
        std::unique_lock<std::mutex> lock(m_state_mutex);
        if (m_stopping || m_restarting_sdk || m_recovering)
        {
            return;
        }
        const bool recoverable =
            m_options.mode == BridgeMode::Remote &&
            m_options.reconnect_timeout_seconds > 0 && !g_stop_requested.load();
        const uint64_t previous_generation = m_connection_generation.load();
        const uint32_t previous_session_id = m_session_id;
        m_connection_generation.store(
            ConnectionEpochGuard::next_generation(previous_generation));
        m_recovering = true;
        m_streaming_enabled = false;
        m_required_frame_seen = false;
        m_required_topology_qualified = false;
        m_license_qualified = false;
        m_topologies.clear();
        m_tracker_systems.clear();
        m_last_landscape_payload.clear();
        m_connected.store(false);
        m_disconnect_pending = !g_stop_requested.load();
        std::ostringstream output;
        output << envelope("status") << ",\"state\":\"disconnected\",\"message\":\"MANUS "
               << mode_display_name(m_options.mode)
               << " host disconnected\",\"recoverable\":"
               << (recoverable ? "true" : "false")
               << ",\"previous_connection_generation\":" << previous_generation
               << ",\"previous_session_id\":" << previous_session_id
               << connection_fields_locked() << "}";
        emit(output.str());
        lock.unlock();
        m_state_changed.notify_all();
    }

    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(m_state_mutex);
            m_stopping = true;
            m_streaming_enabled = false;
        }
        unregister_callbacks();
        s_instance = nullptr;
        if (m_connected.load())
        {
            CoreSdk_Disconnect();
            m_connected.store(false);
        }
        if (m_sdk_initialized)
        {
            CoreSdk_ShutDown();
            m_sdk_initialized = false;
        }
    }

    static void on_landscape(const Landscape* const landscape)
    {
        if (s_instance != nullptr)
        {
            s_instance->handle_landscape(landscape);
        }
    }

    static void on_raw_skeleton(const SkeletonStreamInfo* const stream)
    {
        if (s_instance != nullptr)
        {
            s_instance->handle_raw_skeleton(stream);
        }
    }

    static void on_tracker(const TrackerStreamInfo* const stream)
    {
        if (s_instance != nullptr)
        {
            s_instance->handle_tracker(stream);
        }
    }

    static void on_connect(const ManusHost* const host)
    {
        if (s_instance != nullptr)
        {
            s_instance->handle_connect(host);
        }
    }

    static void on_disconnect(const ManusHost* const)
    {
        if (s_instance != nullptr)
        {
            s_instance->handle_disconnect();
        }
    }

    Options m_options;
    std::vector<unsigned char> m_left_calibration;
    std::vector<unsigned char> m_right_calibration;
    std::mutex m_output_mutex;
    std::mutex m_state_mutex;
    std::condition_variable m_state_changed;
    std::unordered_map<uint32_t, GloveState> m_topologies;
    std::unordered_map<std::string, std::string> m_tracker_systems;
    std::optional<ManusHost> m_selected_host;
    std::string m_sdk_version;
    std::string m_core_version;
    std::string m_landscape_core_version;
    std::string m_last_landscape_payload;
    uint32_t m_session_id = 0;
    bool m_versions_compatible = false;
    bool m_versions_qualified = false;
    bool m_hand_motion_qualified = false;
    bool m_license_qualified = false;
    bool m_required_topology_qualified = false;
    bool m_required_frame_seen = false;
    bool m_streaming_enabled = false;
    bool m_recovering = false;
    bool m_restarting_sdk = false;
    bool m_stopping = false;
    uint64_t m_connect_callback_generation = 0;
    std::atomic<uint64_t> m_connection_generation{ 0 };
    std::atomic<uint64_t> m_sequence{ 0 };
    std::atomic<uint64_t> m_tracker_sequence{ 0 };
    std::atomic<int> m_terminal_exit_code{ 0 };
    bool m_license_error_emitted = false;
    bool m_disconnect_pending = false;
    bool m_sdk_initialized = false;
    bool m_callbacks_registered = false;
    std::atomic<bool> m_connected{ false };
};

Bridge* Bridge::s_instance = nullptr;

} // namespace

int main(int argc, char** argv)
{
    try
    {
        Options options = parse_options(argc, argv);
        if (options.print_protocol_version)
        {
            std::cout << kProtocol << " " << kProtocolVersion << std::endl;
            return 0;
        }
        std::signal(SIGINT, request_stop);
        std::signal(SIGTERM, request_stop);
        std::signal(SIGPIPE, request_stop);
        Bridge bridge(std::move(options));
        return bridge.run();
    }
    catch (const std::exception& error)
    {
        std::cerr << "[manus_bridge] " << error.what() << std::endl;
        return 2;
    }
}
