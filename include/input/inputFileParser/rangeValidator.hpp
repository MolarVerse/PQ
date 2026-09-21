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

#ifndef _RANGE_VALIDATOR_HPP_
#define _RANGE_VALIDATOR_HPP_

#include "keyValidatorBase.hpp"

namespace input
{
    template <typename T>
    class RangeValidator : public KeyValidator<T>
    {
       private:
        std::optional<T> _min;
        std::optional<T> _max;

       public:
        RangeValidator(
            const std::optional<T> &min,
            const std::optional<T> &max
        );

        [[nodiscard]]
        bool validate(const T &value) const override;

        [[nodiscard]]
        std::string errorMessage() const override;
    };
}   // namespace input

#ifndef _RANGE_VALIDATOR_TPP_
#include "rangeValidator.tpp"
#endif

#endif   // _RANGE_VALIDATOR_HPP_
