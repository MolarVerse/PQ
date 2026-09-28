#ifndef _POTENTIAL_ENUM_HPP_
#define _POTENTIAL_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

#include "base.hpp"

/**
 * @enum Enumeration for different types of force fields.
 *
 */
enum class ForceFieldType : std::uint8_t;

#define FF_TYPE_LIST(X) \
    X(OFF)              \
    X(ON)               \
    X(BONDED)

MSTD_ENUM(ForceFieldType, std::uint8_t, FF_TYPE_LIST)

#undef FF_TYPE_LIST

/**
 * @enum Enumeration for different types of non-coulomb interactions.
 *
 */
enum class NonCoulombType : std::uint8_t;

#define NON_COULOMB_TYPE_LIST(X) \
    X(NONE)                      \
    X(LJ)                        \
    X(LJ_9_12)                   \
    X(BUCKINGHAM)                \
    X(MORSE)                     \
    X(GUFF)

MSTD_ENUM(NonCoulombType, std::uint8_t, NON_COULOMB_TYPE_LIST)

/**
 * @brief Input alias for settings::NonCoulombType
 */
template <>
struct InputAlias<NonCoulombType>
{
    static constexpr std::array<std::pair<std::string_view, NonCoulombType>, 1>
        value = {{{"buck", NonCoulombType::BUCKINGHAM}}};
};

#undef NON_COULOMB_TYPE_LIST

/**
 * @brief Enumeration for different types of long-range Coulomb interaction
 * methods.
 *
 */
enum class CoulombLongRangeType : std::uint8_t;

#define COULOMB_LONG_RANGE_TYPE_LIST(X) \
    X(SHIFTED)                          \
    X(REACTION_FIELD)                   \
    X(WOLF)

MSTD_ENUM(CoulombLongRangeType, std::uint8_t, COULOMB_LONG_RANGE_TYPE_LIST)

/**
 * @brief Input alias for settings::CoulombLongRangeType
 */
template <>
struct InputAlias<CoulombLongRangeType>
{
    static constexpr std::
        array<std::pair<std::string_view, CoulombLongRangeType>, 1>
            value = {{{"none", CoulombLongRangeType::SHIFTED}}};
};

#undef COULOMB_LONG_RANGE_TYPE_LIST

#endif   // _POTENTIAL_ENUM_HPP_
