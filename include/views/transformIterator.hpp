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

#include <functional>

#include "rangeIterator.hpp"

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
    class TransformIterator : public RangeIterator<Iter>
    {
       private:
        const Func* _func;

       public:
        using base_ref   = std::iter_reference_t<Iter>;
        using reference  = std::invoke_result_t<const Func&, base_ref>;
        using value_type = std::remove_cvref_t<reference>;
        using pointer    = value_type*;

        // make it default constructible, copyable, and movable
        TransformIterator()                                    = default;
        TransformIterator(const TransformIterator&)            = default;
        TransformIterator(TransformIterator&&)                 = default;
        TransformIterator& operator=(const TransformIterator&) = default;
        TransformIterator& operator=(TransformIterator&&)      = default;

        TransformIterator(Iter current, const Func* func);

        reference operator*() const;
        // pointer            operator->() const;
        TransformIterator& operator++();
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
        : RangeIterator<Iter>(current), _func(func)
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
        return std::invoke(*_func, *this->_current);
    }

    /**
     * @brief returns a pointer to the transformed value
     *
     * @return pointer to the transformed value
     */
    // template <typename Iter, typename Func>
    // typename TransformIterator<Iter, Func>::pointer TransformIterator<
    //     Iter,
    //     Func>::operator->() const
    // {
    //     return std::addressof(std::invoke(*_func, *this->_current));
    // }

    /**
     * @brief increment the iterator to the next element
     *
     * @return TransformIterator&
     */
    template <typename Iter, typename Func>
    TransformIterator<Iter, Func>& TransformIterator<Iter, Func>::operator++()
    {
        ++(this->_current);
        return *this;
    }

}   // namespace pqviews

#endif   // __TRANSFORM_ITERATOR_HPP__
