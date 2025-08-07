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

#ifndef __TRANSFORM_SENTINEL_HPP__
#define __TRANSFORM_SENTINEL_HPP__

#include <cstddef>   // for std::ptrdiff_t
#include <iterator>

namespace pqviews
{
    /**
     * @brief TransformSentinel is a sentinel for TransformIterator
     *
     * @tparam Iter The iterator type to be transformed
     */
    template <typename Iter>
    class TransformSentinel
    {
       private:
        Iter _end;

       public:
        // Use std::ptrdiff_t as a default when iterator_traits are not
        // available
        using difference_type = std::ptrdiff_t;

        // make it default constructible, copyable, and movable
        TransformSentinel()                                    = default;
        TransformSentinel(const TransformSentinel&)            = default;
        TransformSentinel(TransformSentinel&&)                 = default;
        TransformSentinel& operator=(const TransformSentinel&) = default;
        TransformSentinel& operator=(TransformSentinel&&)      = default;

        // Required by std::sentinel_for
        explicit TransformSentinel(Iter end) : _end(end) {}

        /**
         * @brief checks if the iterator is equal to the sentinel
         *
         * @param it the TransformIterator to compare with
         * @return true if they are equal, false otherwise
         */
        template <typename Iterator>
        friend bool operator==(const Iterator& it, const TransformSentinel& s)
        {
            return it.current() == s._end;
        }

        /**
         * @brief checks if the sentinel is equal to the iterator
         *
         * @param it the TransformIterator to compare with
         * @return true if they are equal, false otherwise
         */
        template <typename Iterator>
        friend bool operator!=(const Iterator& it, const TransformSentinel& s)
        {
            return !(it == s);
        }

        /**
         * @brief calculates the distance between the sentinel and the iterator
         *
         * @param s the TransformSentinel to compare with
         * @param it the TransformIterator to compare with
         * @return difference_type the distance between the two
         */
        template <typename Iterator>
        friend difference_type operator-(
            const TransformSentinel& s,
            const Iterator&          it
        )
        {
            return std::distance(it.current(), s._end);
        }

        /**
         * @brief calculates the distance between the iterator and the sentinel
         *
         * @param it the TransformIterator to compare with
         * @param s the TransformSentinel to compare with
         * @return difference_type the distance between the two
         */
        template <typename Iterator>
        friend difference_type operator-(
            const Iterator&          it,
            const TransformSentinel& s
        )
        {
            return -std::distance(it.current(), s._end);
        }
    };

}   // namespace pqviews

#endif   // __TRANSFORM_SENTINEL_HPP__
