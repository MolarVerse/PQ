#ifndef _HESSIAN_ENUM_HPP_
#define _HESSIAN_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @brief Enum class for different Hessian builder types.
 *
 */
enum class HessianBuilderType : std::uint8_t;

#define HESSIAN_BUILDER_TYPE_LIST(X) \
    X(CENTRAL)                       \
    X(FORWARD)                       \
    X(FIVE_POINT)                    \
    X(ANALYTIC)

MSTD_ENUM(HessianBuilderType, std::uint8_t, HESSIAN_BUILDER_TYPE_LIST)

#endif   // _HESSIAN_ENUM_HPP_
