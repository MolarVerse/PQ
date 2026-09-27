#ifndef _ENUM_BASE_HPP_
#define _ENUM_BASE_HPP_

#include <array>
#include <string_view>

/**
 * @enum VirialType
 *
 * @brief enum class to store the type of the virial
 *
 */
template <typename T>
struct InputAlias
{
    static constexpr std::array<std::pair<std::string_view, T>, 0> value{};
};

#endif   // _ENUM_BASE_HPP_
