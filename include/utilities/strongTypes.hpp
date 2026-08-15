/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

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
