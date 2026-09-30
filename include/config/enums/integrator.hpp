#ifndef _INTEGRATOR_ENUM_HPP_
#define _INTEGRATOR_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>
#include <mstd/enum/enum_string.hpp>

/**
 * @brief Enum class for different types of integrators
 */
enum class IntegratorType : std::uint8_t;

#define INTEGRATOR_TYPE_LIST(X) \
    X(NONE)                     \
    X(VELOCITY_VERLET)

MSTD_ENUM(IntegratorType, std::uint8_t, INTEGRATOR_TYPE_LIST)

namespace mstd
{
    /**
     * @brief Provides string aliases for IntegratorType enum values
     */
    template <>
    struct EnumAliases<IntegratorType>
    {
        static constexpr auto value = makeAliases<IntegratorType>(
            {{"v_verlet", IntegratorType::VELOCITY_VERLET}}
        );
    };
}   // namespace mstd

#undef INTEGRATOR_TYPE_LIST

#endif   // _INTEGRATOR_ENUM_HPP_
