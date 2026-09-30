#ifndef _GENERAL_ENUM_HPP_
#define _GENERAL_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @enum FPType
 *
 * @brief enum class to store the floating point type
 *
 */
enum class FPType : std::uint8_t;

#define FP_TYPE_LIST(X) \
    X(FLOAT)            \
    X(DOUBLE)

MSTD_ENUM(FPType, std::uint8_t, FP_TYPE_LIST)

#undef FP_TYPE_LIST

#endif   // _GENERAL_ENUM_HPP_
