#ifndef _SIMULATION_BOX_ENUM_HPP_
#define _SIMULATION_BOX_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @brief InitVelocities enum forward declaration
 */
enum class InitVelocities : std::uint8_t;

#define INIT_VELOCITIES_LIST(X) \
    X(FALSE)                    \
    X(TRUE)                     \
    X(FORCE)

MSTD_ENUM(InitVelocities, std::uint8_t, INIT_VELOCITIES_LIST)

#undef INIT_VELOCITIES_LIST

#endif   // _SIMULATION_BOX_ENUM_HPP_
