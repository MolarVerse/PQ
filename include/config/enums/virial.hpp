#ifndef _VIRIAL_ENUM_HPP_
#define _VIRIAL_ENUM_HPP_

#include <cstdint>   // for uint8_t
#include <mstd/enum.hpp>

/**
 * @brief Enum class for different types of virial
 */
enum class VirialType : std::uint8_t;

#define VIRIAL_TYPE_LIST(X) \
    X(ATOMIC)               \
    X(MOLECULAR)

MSTD_ENUM(VirialType, std::uint8_t, VIRIAL_TYPE_LIST)

#undef VIRIAL_TYPE_LIST

#endif   // _VIRIAL_ENUM_HPP_
