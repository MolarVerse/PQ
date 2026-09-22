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

#include <string>

#include "keyValidatorBase.hpp"

namespace input
{
    /**
     * @brief Specifies whether a value should be greater than or greater than
     * or equal to another value
     *
     */
    enum class Greater : std::uint8_t
    {
        GE,
        GT,
    };

    /**
     * @brief Specifies whether a value should be less than or less than
     * or equal to another value
     *
     */
    enum class Less : std::uint8_t
    {
        LE,
        LT,
    };

    template <typename T, Greater G = Greater::GE, Less L = Less::LE>
    class RangeValidator : public KeyValidator<T>
    {
       private:
        std::optional<T> _min;
        std::optional<T> _max;

        bool _allowNaN = false;
        bool _allowInf = false;

        bool _nanCheckFailed = false;
        bool _infCheckFailed = false;

       public:
        RangeValidator(
            const std::optional<T> &min,
            const std::optional<T> &max
        );

        [[nodiscard]]
        bool validate(const T &value) override;

        [[nodiscard]]
        std::string errorMessage() override;

       private:
        [[nodiscard]]
        constexpr bool _isGreater(const T &value) const
        requires(G == Greater::GE);
        [[nodiscard]]
        constexpr bool _isGreater(const T &value) const
        requires(G == Greater::GT);

        [[nodiscard]]
        constexpr bool _isLess(const T &value) const
        requires(L == Less::LE);
        [[nodiscard]]
        constexpr bool _isLess(const T &value) const
        requires(L == Less::LT);

        [[nodiscard]]
        std::string _isGreaterMsg() const
        requires(G == Greater::GE);
        [[nodiscard]]
        std::string _isGreaterMsg() const
        requires(G == Greater::GT);

        [[nodiscard]]
        std::string _isLessMsg() const
        requires(L == Less::LE);
        [[nodiscard]]
        std::string _isLessMsg() const
        requires(L == Less::LT);
    };
}   // namespace input

#ifndef _RANGE_VALIDATOR_TPP_
#include "rangeValidator.tpp"
#endif

#endif   // _RANGE_VALIDATOR_HPP_
