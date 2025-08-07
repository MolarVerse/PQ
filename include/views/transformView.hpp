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

#ifndef __TRANSFORM_VIEW_HPP__
#define __TRANSFORM_VIEW_HPP__

#include <utility>

#include "transformIterator.hpp"
#include "transformSentinel.hpp"

namespace pqviews
{

    /**
     * @brief TransformView is a range-based view that transforms elements using
     * a function
     *
     * @tparam Range The range type to be transformed
     * @tparam Func  The function type used for transformation
     */
    template <typename Range, typename Func>
    class TransformView
    {
       private:
        using begin_t   = decltype(std::begin(std::declval<Range&>()));
        using end_t     = decltype(std::end(std::declval<Range&>()));
        using c_begin_t = decltype(std::begin(std::declval<const Range&>()));
        using c_end_t   = decltype(std::end(std::declval<const Range&>()));

        Range _range;
        Func  _func;

       public:
        using iterator       = TransformIterator<begin_t, Func>;
        using sentinel       = TransformSentinel<end_t>;
        using const_iterator = TransformIterator<c_begin_t, Func>;
        using const_sentinel = TransformSentinel<c_end_t>;

        TransformView(Range r, Func f);

        iterator begin();
        sentinel end();

        const_iterator begin() const;
        const_sentinel end() const;
    };

    /**
     * @brief Constructor for TransformView
     *
     * @param r The range to be transformed
     * @param f The function to be used for transformation
     */
    template <typename Range, typename Func>
    TransformView<Range, Func>::TransformView(Range r, Func f)
        : _range(std::move(r)), _func(std::move(f))
    {
    }

    /**
     * @brief returns an iterator to the beginning of the transformed range
     *
     * @return iterator
     */
    template <typename Range, typename Func>
    typename TransformView<Range, Func>::iterator TransformView<Range, Func>::
        begin()
    {
        return iterator{std::begin(_range), &_func};
    }

    /**
     * @brief returns a sentinel to the end of the transformed range
     *
     * @return sentinel
     */
    template <typename Range, typename Func>
    typename TransformView<Range, Func>::sentinel TransformView<Range, Func>::
        end()
    {
        return sentinel{std::end(_range)};
    }

    /**
     * @brief returns a const iterator to the beginning of the transformed
     * range
     *
     * @return const_iterator
     */
    template <typename Range, typename Func>
    typename TransformView<Range, Func>::const_iterator TransformView<
        Range,
        Func>::begin() const
    {
        return const_iterator{std::begin(_range), &_func};
    }

    /**
     * @brief returns a const sentinel to the end of the transformed range
     *
     * @return const_sentinel
     */
    template <typename Range, typename Func>
    typename TransformView<Range, Func>::const_sentinel TransformView<
        Range,
        Func>::end() const
    {
        return const_sentinel{std::end(_range)};
    }

}   // namespace pqviews

#endif   // __TRANSFORM_VIEW_HPP__
