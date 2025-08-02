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

#ifndef __ITERATOR_HPP__
#define __ITERATOR_HPP__

#include <iterator>   // clang-format on

namespace pqviews
{
    /**
     * @brief FilterIterator is an iterator that filters elements based on a
     * predicate
     *
     * @tparam Iter The iterator type to be filtered
     * @tparam Pred The predicate type used for filtering
     */
    template <typename Iter, typename Pred>
    class FilterIterator
    {
       public:
        using iterator_category = std::input_iterator_tag;
        using value_type = typename std::iterator_traits<Iter>::value_type;
        using pointer    = typename std::iterator_traits<Iter>::pointer;
        using reference  = typename std::iterator_traits<Iter>::reference;
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

        // make it default constructible, copyable, and movable
        FilterIterator()                                 = default;
        FilterIterator(const FilterIterator&)            = default;
        FilterIterator(FilterIterator&&)                 = default;
        FilterIterator& operator=(const FilterIterator&) = default;
        FilterIterator& operator=(FilterIterator&&)      = default;

        FilterIterator(Iter current, Iter end, const Pred* pred)
            : _current(current), _end(end), _pred(pred)
        {
            satisfy();
        }

        /**
         * @brief reference to the current element in the filtered range
         *
         * @return reference
         */
        reference operator*() const { return *_current; }

        /**
         * @brief pointer to the current element in the filtered range
         *
         * @return pointer
         */
        pointer operator->() const { return std::addressof(*_current); }

        /**
         * @brief increment the iterator to the next element in the filtered
         * range
         *
         * @return FilterIterator&
         */
        FilterIterator& operator++()
        {
            ++_current;
            satisfy();
            return *this;
        }

        /**
         * @brief post-increment the iterator to the next element in the
         * filtered range
         *
         * @return FilterIterator
         */
        FilterIterator operator++(int)
        {
            FilterIterator temp = *this;
            ++(*this);
            return temp;
        }

        /**
         * @brief checks if two FilterIterators are equal
         *
         * @param other the other FilterIterator to compare with
         * @return true if they are equal, false otherwise
         */
        bool operator==(const FilterIterator& other) const
        {
            return _current == other._current;
        }

        /**
         * @brief checks if two FilterIterators are not equal
         *
         * @param other the other FilterIterator to compare with
         * @return true if they are not equal, false otherwise
         */
        bool operator!=(const FilterIterator& other) const
        {
            return !(*this == other);
        }

        /**
         * @brief returns the current iterator
         *
         * @return Iter
         */
        Iter current() const { return _current; }

       private:
        /**
         * @brief satisfies the predicate for the current iterator
         *
         * This function advances the iterator until it finds an element that
         * satisfies the predicate or reaches the end of the range.
         */
        void satisfy()
        {
            while (_current != _end && !_pred->operator()(*_current))
            {
                ++_current;
            }
        }

        Iter        _current;
        Iter        _end;
        const Pred* _pred;
    };

    /**
     * @brief FilterSentinel is a sentinel for FilterIterator
     *
     * @tparam Iter The iterator type to be filtered
     */
    template <typename Iter>
    class FilterSentinel
    {
       private:
        Iter _end;

       public:
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

        // make it default constructible, copyable, and movable
        FilterSentinel()                                 = default;
        FilterSentinel(const FilterSentinel&)            = default;
        FilterSentinel(FilterSentinel&&)                 = default;
        FilterSentinel& operator=(const FilterSentinel&) = default;
        FilterSentinel& operator=(FilterSentinel&&)      = default;

        // Required by std::sentinel_for
        explicit FilterSentinel(Iter end) : _end(end) {}

        /**
         * @brief checks if the iterator is equal to the sentinel
         *
         * @param it the FilterIterator to compare with
         * @return true if they are equal, false otherwise
         */
        template <typename Pred>
        friend bool operator==(
            const FilterIterator<Iter, Pred>& it,
            const FilterSentinel&             s
        )
        {
            return it.current() == s._end;
        }

        /**
         * @brief checks if the sentinel is equal to the iterator
         *
         * @param it the FilterIterator to compare with
         * @return true if they are equal, false otherwise
         */
        template <typename Iterator>
        friend bool operator!=(const Iterator& it, const FilterSentinel& s)
        {
            return !(it == s);
        }

        /**
         * @brief calculates the distance between the sentinel and the iterator
         *
         * @param s the FilterSentinel to compare with
         * @param it the FilterIterator to compare with
         * @return difference_type the distance between the two
         */
        template <typename Iterator>
        friend difference_type operator-(
            const FilterSentinel& s,
            const Iterator&       it
        )
        {
            return std::distance(it.current(), s._end);
        }

        /**
         * @brief calculates the distance between the iterator and the sentinel
         *
         * @param it the FilterIterator to compare with
         * @param s the FilterSentinel to compare with
         * @return difference_type the distance between the two
         */
        template <typename Iterator>
        friend difference_type operator-(
            const Iterator&       it,
            const FilterSentinel& s
        )
        {
            return -std::distance(it.current(), s._end);
        }
    };

}   // namespace pqviews

#endif   // __ITERATOR_HPP__