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

#ifndef __FILTER_VIEW_HPP__
#define __FILTER_VIEW_HPP__

#include <utility>

#include "filterIterator.hpp"
#include "sentinel.hpp"

namespace pqviews
{

    /**
     * @brief FilterView is a range-based view that filters elements based on a
     * predicate
     *
     * @tparam Range The range type to be filtered
     * @tparam Pred  The predicate type used for filtering
     */
    template <typename Range, typename Pred>
    class FilterView
    {
       private:
        using begin_t   = decltype(std::begin(std::declval<Range&>()));
        using end_t     = decltype(std::end(std::declval<Range&>()));
        using c_begin_t = decltype(std::begin(std::declval<const Range&>()));
        using c_end_t   = decltype(std::end(std::declval<const Range&>()));

        Range _range;
        Pred  _pred;

       public:
        using iterator       = FilterIterator<begin_t, Pred>;
        using const_iterator = FilterIterator<c_begin_t, Pred>;
        using sentinel       = Sentinel<end_t>;
        using const_sentinel = Sentinel<c_end_t>;

        FilterView(Range r, Pred p);

        iterator begin();
        sentinel end();

        const_iterator begin() const;
        const_sentinel end() const;
    };

    /**
     * @brief Constructor for FilterView
     *
     * @param r The range to be filtered
     * @param p The predicate to be used for filtering
     */
    template <typename Range, typename Pred>
    FilterView<Range, Pred>::FilterView(Range r, Pred p)
        : _range(std::forward<Range>(r)), _pred(std::move(p))
    {
    }

    /**
     * @brief returns an iterator to the beginning of the filtered range
     *
     * @return iterator
     */
    template <typename Range, typename Pred>
    FilterView<Range, Pred>::iterator FilterView<Range, Pred>::begin()
    {
        return iterator{std::begin(_range), std::end(_range), &_pred};
    }

    /**
     * @brief returns a sentinel to the end of the filtered range
     *
     * @return sentinel
     */
    template <typename Range, typename Pred>
    FilterView<Range, Pred>::sentinel FilterView<Range, Pred>::end()
    {
        return sentinel{std::end(_range)};
    }

    /**
     * @brief returns a const iterator to the beginning of the filtered
     * range
     *
     * @return const_iterator
     */
    template <typename Range, typename Pred>
    FilterView<Range, Pred>::const_iterator FilterView<Range, Pred>::begin(
    ) const
    {
        return const_iterator{std::begin(_range), std::end(_range), &_pred};
    }

    /**
     * @brief returns a const sentinel to the end of the filtered range
     *
     * @return const_sentinel
     */
    template <typename Range, typename Pred>
    FilterView<Range, Pred>::const_sentinel FilterView<Range, Pred>::end() const
    {
        return const_sentinel{std::end(_range)};
    }

}   // namespace pqviews

#endif   // __FILTER_VIEW_HPP__