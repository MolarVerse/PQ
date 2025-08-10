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

#ifndef __RANGE_SENTINEL_HPP__
#define __RANGE_SENTINEL_HPP__

#include <iterator>   // for std::input_iterator_tag

namespace pqviews
{
    /**
     * @brief RangeSentinel is a sentinel for RangeIterator
     *
     * @tparam Iter The iterator type to be used with the sentinel
     */
    template <typename Iter>
    class RangeSentinel
    {
       private:
        Iter _end;

       public:
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

        // make it default constructible, copyable, and movable
        RangeSentinel()                                = default;
        RangeSentinel(const RangeSentinel&)            = default;
        RangeSentinel(RangeSentinel&&)                 = default;
        RangeSentinel& operator=(const RangeSentinel&) = default;
        RangeSentinel& operator=(RangeSentinel&&)      = default;

        // Required by std::sentinel_for
        explicit RangeSentinel(Iter end) : _end(end) {}

        /**
         * @brief checks if the iterator is equal to the sentinel
         *
         * @param it the RangeIterator to compare with
         * @return true if they are equal, false otherwise
         */
        template <typename Iterator>
        friend bool operator==(const Iterator& it, const RangeSentinel& s)
        {
            return it.current() == s._end;
        }

        /**
         * @brief checks if the sentinel is equal to the iterator
         *
         * @param it the RangeIterator to compare with
         * @return true if they are equal, false otherwise
         */
        template <typename Iterator>
        friend bool operator!=(const Iterator& it, const RangeSentinel& s)
        {
            return !(it == s);
        }

        /**
         * @brief calculates the distance between the sentinel and the iterator
         *
         * @param s the RangeSentinel to compare with
         * @param it the RangeIterator to compare with
         * @return difference_type the distance between the two
         */
        template <typename Iterator>
        friend difference_type operator-(
            const RangeSentinel& s,
            const Iterator&      it
        )
        {
            return std::distance(it.current(), s._end);
        }

        /**
         * @brief calculates the distance between the iterator and the sentinel
         *
         * @param it the RangeIterator to compare with
         * @param s the RangeSentinel to compare with
         * @return difference_type the distance between the two
         */
        template <typename Iterator>
        friend difference_type operator-(
            const Iterator&      it,
            const RangeSentinel& s
        )
        {
            return -std::distance(it.current(), s._end);
        }
    };

}   // namespace pqviews

#endif   // __RANGE_SENTINEL_HPP__