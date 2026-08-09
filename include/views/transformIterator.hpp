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

#ifndef __TRANSFORM_ITERATOR_HPP__
#define __TRANSFORM_ITERATOR_HPP__

#include <iterator>

namespace pqviews
{
    /**
     * @brief TransformIterator is an iterator that applies a transformation
     * function to each element
     *
     * @tparam Iter The iterator type to be transformed
     * @tparam Func The transformation function type
     */
    template <typename Iter, typename Func>
    class TransformIterator
    {
       private:
        Iter        _current;
        const Func* _func;

       public:
        using iterator_category = std::input_iterator_tag;
        using value_type =
            std::decay_t<decltype(std::declval<Func>()(*std::declval<Iter>()))>;
        using pointer   = value_type*;
        using reference = value_type;
        using difference_type =
            typename std::iterator_traits<Iter>::difference_type;

        // make it default constructible, copyable, and movable
        TransformIterator()                                    = default;
        TransformIterator(const TransformIterator&)            = default;
        TransformIterator(TransformIterator&&)                 = default;
        TransformIterator& operator=(const TransformIterator&) = default;
        TransformIterator& operator=(TransformIterator&&)      = default;

        TransformIterator(Iter current, const Func* func);

        reference          operator*() const;
        TransformIterator& operator++();
        TransformIterator  operator++(int);

        bool operator==(const TransformIterator& other) const;
        bool operator!=(const TransformIterator& other) const;

        Iter current() const;
    };

    /**
     * @brief Constructor for TransformIterator
     *
     * @param current The current iterator position
     * @param func The transformation function to apply
     */
    template <typename Iter, typename Func>
    TransformIterator<Iter, Func>::TransformIterator(
        Iter        current,
        const Func* func
    )
        : _current(current), _func(func)
    {
    }

    /**
     * @brief applies transformation and returns the result
     *
     * @return reference (actually value_type due to transformation)
     */
    template <typename Iter, typename Func>
    typename TransformIterator<Iter, Func>::reference TransformIterator<
        Iter,
        Func>::operator*() const
    {
        return _func->operator()(*_current);
    }

    /**
     * @brief increment the iterator to the next element
     *
     * @return TransformIterator&
     */
    template <typename Iter, typename Func>
    TransformIterator<Iter, Func>& TransformIterator<Iter, Func>::operator++()
    {
        ++_current;
        return *this;
    }

    /**
     * @brief post-increment the iterator to the next element
     *
     * @return TransformIterator
     */
    template <typename Iter, typename Func>
    TransformIterator<Iter, Func> TransformIterator<Iter, Func>::operator++(int)
    {
        TransformIterator temp = *this;
        ++(*this);
        return temp;
    }

    /**
     * @brief checks if two TransformIterators are equal
     *
     * @param other the other TransformIterator to compare with
     * @return true if they are equal, false otherwise
     */
    template <typename Iter, typename Func>
    bool TransformIterator<Iter, Func>::operator==(
        const TransformIterator& other
    ) const
    {
        return _current == other._current;
    }

    /**
     * @brief checks if two TransformIterators are not equal
     *
     * @param other the other TransformIterator to compare with
     * @return true if they are not equal, false otherwise
     */
    template <typename Iter, typename Func>
    bool TransformIterator<Iter, Func>::operator!=(
        const TransformIterator& other
    ) const
    {
        return !(*this == other);
    }

    /**
     * @brief returns the current iterator
     *
     * @return Iter
     */
    template <typename Iter, typename Func>
    Iter TransformIterator<Iter, Func>::current() const
    {
        return _current;
    }

}   // namespace pqviews

#endif   // __TRANSFORM_ITERATOR_HPP__
