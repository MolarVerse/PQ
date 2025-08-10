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

#ifndef __FILTER_ITERATOR_HPP__
#define __FILTER_ITERATOR_HPP__

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
       private:
        void satisfy();

        Iter        _current;
        Iter        _end;
        const Pred* _pred;

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

        FilterIterator(Iter current, Iter end, const Pred* pred);

        reference       operator*() const;
        pointer         operator->() const;
        FilterIterator& operator++();
        FilterIterator  operator++(int);

        bool operator==(const FilterIterator& other) const;
        bool operator!=(const FilterIterator& other) const;

        Iter current() const;
    };

    /**
     * @brief satisfies the predicate for the current iterator
     *
     * This function advances the iterator until it finds an element that
     * satisfies the predicate or reaches the end of the range.
     */
    template <typename Iter, typename Pred>
    void FilterIterator<Iter, Pred>::satisfy()
    {
        while (_current != _end && !_pred->operator()(*_current))
        {
            ++_current;
        }
    }

    /**
     * @brief Constructor for FilterIterator
     *
     * @param current The current iterator position
     * @param end The end iterator position
     * @param pred The predicate used for filtering
     * @details This constructor initializes the FilterIterator with the current
     * and end iterators, and applies the predicate to find the first valid
     * element.
     **/
    template <typename Iter, typename Pred>
    FilterIterator<Iter, Pred>::FilterIterator(
        Iter        current,
        Iter        end,
        const Pred* pred
    )
        : _current(current), _end(end), _pred(pred)
    {
        satisfy();
    }

    /**
     * @brief reference to the current element in the filtered range
     *
     * @return reference
     */
    template <typename Iter, typename Pred>
    typename FilterIterator<Iter, Pred>::reference FilterIterator<Iter, Pred>::
    operator*() const
    {
        return *_current;
    }

    /**
     * @brief pointer to the current element in the filtered range
     *
     * @return pointer
     */
    template <typename Iter, typename Pred>
    typename FilterIterator<Iter, Pred>::pointer FilterIterator<Iter, Pred>::
    operator->() const
    {
        return std::addressof(*_current);
    }

    /**
     * @brief increment the iterator to the next element in the filtered
     * range
     *
     * @return FilterIterator&
     */
    template <typename Iter, typename Pred>
    FilterIterator<Iter, Pred>& FilterIterator<Iter, Pred>::operator++()
    {
        ++_current;
        satisfy();
        return *this;
    }

    /**
     * @brief checks if two FilterIterators are equal
     *
     * @param other the other FilterIterator to compare with
     * @return true if they are equal, false otherwise
     */
    template <typename Iter, typename Pred>
    bool FilterIterator<Iter, Pred>::operator==(const FilterIterator& other
    ) const
    {
        return _current == other._current;
    }

    /**
     * @brief checks if two FilterIterators are not equal
     *
     * @param other the other FilterIterator to compare with
     * @return true if they are not equal, false otherwise
     */
    template <typename Iter, typename Pred>
    bool FilterIterator<Iter, Pred>::operator!=(const FilterIterator& other
    ) const
    {
        return !(*this == other);
    }

    /**
     * @brief returns the current iterator
     *
     * @return Iter
     */
    template <typename Iter, typename Pred>
    Iter FilterIterator<Iter, Pred>::current() const
    {
        return _current;
    }

}   // namespace pqviews

#endif   // __FILTER_ITERATOR_HPP__