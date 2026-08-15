#ifndef _STRONG_TYPES_HPP_
#define _STRONG_TYPES_HPP_

#include <cstddef>
#include <mstd/types.hpp>

template <typename Tag>
using StrongSizeT = mstd::StrongType<
    size_t,
    Tag,
    mstd::StrongTypeTrait::ORDERED | mstd::StrongTypeTrait::HASHABLE>;

// clang-format off
struct AtomNumberTag{};
using AtomNumber = StrongSizeT<struct AtomNumberTag>;
// clang-format on

struct BondIdTag
{
    static std::string toString(const size_t &value)
    {
        return std::format("BondId({})", value);
    }
};
using BondId = StrongSizeT<struct BondIdTag>;

struct AngleIdTag
{
    static std::string toString(const size_t &value)
    {
        return std::format("AngleId({})", value);
    }
};
using AngleId = StrongSizeT<struct AngleIdTag>;

struct DihedralIdTag
{
    static std::string toString(const size_t &value)
    {
        return std::format("DihedralId({})", value);
    }
};
using DihedralId = StrongSizeT<struct DihedralIdTag>;

#endif   // _STRONG_TYPES_HPP_
