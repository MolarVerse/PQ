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

#ifndef __FILTER_ADAPTOR_HPP__
#define __FILTER_ADAPTOR_HPP__

#include <type_traits>

#include "filterView.hpp"

namespace pqviews
{
    /**
     * @brief FilterAdaptor is a functor that creates a FilterView
     *
     * @tparam Pred Predicate type used for filtering
     */
    template <typename Pred>
    struct FilterAdaptor
    {
        Pred _pred;

        template <typename Range>
        auto operator()(Range&& range) const
        {
            if constexpr (std::is_lvalue_reference_v<Range>)
                // For lvalue references, store a reference to avoid copying
                return FilterView<Range, Pred>{
                    std::forward<Range>(range),
                    _pred
                };
            else
                // For rvalue references, decay
                return FilterView<std::decay_t<Range>, Pred>{
                    std::forward<Range>(range),
                    _pred
                };
        }
    };

    /**
     * @brief Creates a FilterAdaptor with the given predicate
     *
     * @tparam Pred Predicate type used for filtering
     * @param pred Predicate to be used for filtering
     * @return FilterAdaptor<Pred>
     */
    template <typename Pred>
    auto filter(Pred pred)
    {
        return FilterAdaptor<Pred>{pred};
    }

}   // namespace pqviews

#endif   // __FILTER_ADAPTOR_HPP__