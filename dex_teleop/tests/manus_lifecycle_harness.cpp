#include "manus_lifecycle.h"

#include <cassert>
#include <cstdint>

using dex_teleop::manus::ConnectionEpochGuard;

int main()
{
    const uint64_t initial = 1;
    assert(!ConnectionEpochGuard::callback_is_current(initial, initial, 0, false));
    assert(ConnectionEpochGuard::callback_is_current(initial, initial, initial, false));

    assert(!ConnectionEpochGuard::qualification_complete(true, true, true, true, false));
    assert(!ConnectionEpochGuard::qualification_complete(true, true, false, true, true));
    assert(ConnectionEpochGuard::qualification_complete(true, true, true, true, true));

    assert(!ConnectionEpochGuard::sample_may_publish(
        initial, initial, initial, false, false));
    assert(ConnectionEpochGuard::sample_may_publish(
        initial, initial, initial, false, true));

    const uint64_t reconnected = ConnectionEpochGuard::next_generation(initial);
    assert(reconnected > initial);
    assert(!ConnectionEpochGuard::sample_may_publish(
        initial, reconnected, reconnected, false, true));
    assert(!ConnectionEpochGuard::sample_may_publish(
        reconnected, reconnected, reconnected, true, true));
    assert(ConnectionEpochGuard::sample_may_publish(
        reconnected, reconnected, reconnected, false, true));
    return 0;
}
