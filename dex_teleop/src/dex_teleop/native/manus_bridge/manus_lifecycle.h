#pragma once

#include <cstdint>

namespace dex_teleop::manus
{

struct ConnectionEpochGuard
{
    static constexpr uint64_t next_generation(uint64_t current) noexcept
    {
        return current + 1;
    }

    static constexpr bool callback_is_current(
        uint64_t captured_generation,
        uint64_t current_generation,
        uint64_t connect_callback_generation,
        bool restarting_sdk) noexcept
    {
        return !restarting_sdk &&
               captured_generation == current_generation &&
               connect_callback_generation == current_generation;
    }

    static constexpr bool qualification_complete(
        bool versions,
        bool hand_motion,
        bool license,
        bool required_topology,
        bool required_frame) noexcept
    {
        return versions && hand_motion && license && required_topology &&
               required_frame;
    }

    static constexpr bool sample_may_publish(
        uint64_t captured_generation,
        uint64_t current_generation,
        uint64_t connect_callback_generation,
        bool restarting_sdk,
        bool streaming_enabled) noexcept
    {
        return streaming_enabled &&
               callback_is_current(
                   captured_generation,
                   current_generation,
                   connect_callback_generation,
                   restarting_sdk);
    }
};

}  // namespace dex_teleop::manus
