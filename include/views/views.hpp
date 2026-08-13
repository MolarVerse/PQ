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

#ifndef __VIEWS_HPP__
#define __VIEWS_HPP__

#include <utility>

#include "filterAdaptor.hpp"      // IWYU pragma: export
#include "filterView.hpp"         // IWYU pragma: export
#include "transformAdaptor.hpp"   // IWYU pragma: export
#include "transformView.hpp"      // IWYU pragma: export

namespace pqviews
{
    /**
     * @brief A range-based view that applies an adaptor to a range
     *
     * @tparam Range The type of the range to be adapted
     * @tparam Adaptor The type of the adaptor to be applied
     * @param r The range to be adapted
     * @param a The adaptor to be applied
     * @return auto A view of the adapted range
     */
    template <typename Range, typename Adaptor>
    auto operator|(Range&& r, const Adaptor& a)
    {
        return a(std::forward<Range>(r));
    }

}   // namespace pqviews

#endif   // __VIEWS_HPP__