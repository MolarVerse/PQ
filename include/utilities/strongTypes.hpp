#ifndef _STRONG_TYPES_HPP_
#define _STRONG_TYPES_HPP_

#include <cstddef>
#include <mstd/types.hpp>

template <typename Tag>
using StrongSizeT = mstd::StrongType<
    size_t,
    Tag,
    mstd::StrongTypeTrait::ORDERED | mstd::StrongTypeTrait::HASHABLE>;

struct AtomNumberTag
{
};

using AtomNumber = StrongSizeT<struct AtomNumberTag>;

#endif   // _STRONG_TYPES_HPP_
