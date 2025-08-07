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

#ifndef __TRANSFORM_ADAPTOR_HPP__
#define __TRANSFORM_ADAPTOR_HPP__

#include "transformView.hpp"

namespace pqviews
{
    /**
     * @brief TransformAdaptor is a functor that creates a TransformView
     *
     * @tparam Func Function type used for transformation
     */
    template <typename Func>
    struct TransformAdaptor
    {
        Func _func;

        template <typename Range>
        auto operator()(Range&& range) const
        {
            return TransformView<std::decay_t<Range>, Func>{
                std::forward<Range>(range),
                _func
            };
        }
    };

    /**
     * @brief Creates a TransformAdaptor with the given transformation function
     *
     * @tparam Func Function type used for transformation
     * @param func Function to be used for transformation
     * @return TransformAdaptor<Func>
     */
    template <typename Func>
    auto transform(Func func)
    {
        return TransformAdaptor<Func>{func};
    }

}   // namespace pqviews

#endif   // __TRANSFORM_ADAPTOR_HPP__
