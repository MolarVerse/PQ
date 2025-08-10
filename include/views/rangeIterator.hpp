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

#ifndef __RANGE_ITERATOR_HPP__
#define __RANGE_ITERATOR_HPP__

#include <iterator>

namespace pqviews
{
    /**
     * @brief RangeIterator is a common iterator Base for iterators working on
     * ranges
     *
     * @tparam Iter The iterator type to be used
     */
    template <typename Iter>
    class RangeIterator
    {
       protected:
        Iter _current;

       public:
        using iterator_category = std::input_iterator_tag;

        // make it default constructible, copyable, and movable
        RangeIterator()                                = default;
        RangeIterator(const RangeIterator&)            = default;
        RangeIterator(RangeIterator&&)                 = default;
        RangeIterator& operator=(const RangeIterator&) = default;
        RangeIterator& operator=(RangeIterator&&)      = default;

        RangeIterator(Iter current) : _current(current) {}

        RangeIterator operator++(int);

        bool operator==(const RangeIterator& other) const;
        bool operator!=(const RangeIterator& other) const;

        Iter current() const;
    };

    /**
     * @brief checks if two RangeIterators are equal
     *
     * @param other the other RangeIterator to compare with
     * @return true if they are equal, false otherwise
     */
    template <typename Iter>
    bool RangeIterator<Iter>::operator==(const RangeIterator& other) const
    {
        return _current == other._current;
    }

    /**
     * @brief checks if two RangeIterators are not equal
     *
     * @param other the other RangeIterator to compare with
     * @return true if they are not equal, false otherwise
     */
    template <typename Iter>
    bool RangeIterator<Iter>::operator!=(const RangeIterator& other) const
    {
        return !(*this == other);
    }

    /**
     * @brief returns the current iterator
     *
     * @return Iter
     */
    template <typename Iter>
    Iter RangeIterator<Iter>::current() const
    {
        return _current;
    }

    /**
     * @brief post-increment the iterator to the next element in the
     * filtered range
     *
     * @return FilterIterator
     */
    template <typename Iter>
    RangeIterator<Iter> RangeIterator<Iter>::operator++(int)
    {
        RangeIterator temp = *this;
        ++(*this);
        return temp;
    }

}   // namespace pqviews

#endif   // __RANGE_ITERATOR_HPP__
