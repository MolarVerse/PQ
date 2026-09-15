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

#ifndef _BASE_EXCEPTION_HPP_
#define _BASE_EXCEPTION_HPP_

#include "color.hpp"
#include "exceptionTypes.hpp"

namespace exc
{
    /**
     * @class PQException
     *
     * @brief Base class for all custom exceptions in the application
     *
     * This class serves as a base for all custom exceptions in the application.
     * It inherits from std::exception and provides a common interface for
     * exception handling.
     */
    class PQException : public std::exception
    {
       private:
        std::string           _message;
        std::optional<size_t> _lineNumber;

       public:
        explicit PQException(const std::string_view message);
        explicit PQException(
            const std::string_view message,
            std::optional<size_t>  lineNumber
        );

        void setLineNumber(const size_t lineNumber) noexcept;

        [[nodiscard]]
        const std::string &getMessage() const noexcept;

        [[nodiscard]]
        std::optional<size_t> getLineNumber() const noexcept;
    };

    /**
     * @class BaseException
     *
     * @brief Base class for custom exceptions
     *
     * This class serves as a base for all custom exceptions in the application.
     * It provides common functionality such as message handling, line number
     * tracking, and colorful output for exception messages.
     *
     * @tparam Color The color code for the exception message output
     * @tparam Type  The type of exception being thrown
     */
    template <
        Color::Code   colorCode     = Color::FG_RED,
        ExceptionType exceptionType = ExceptionType::Undefined>
    class BaseException : public PQException
    {
       private:
        static constexpr Color::Code   _color = colorCode;
        static constexpr ExceptionType _type  = exceptionType;

       public:
        explicit BaseException(
            const std::string_view message,
            std::optional<size_t>  lineNumber
        );
        explicit BaseException(const std::string_view message);

        [[nodiscard]]
        const char *what() const noexcept override;

        static void colorfulOutput(
            const Color::Code      color,
            const std::string_view message
        );
    };
}   // namespace exc

#ifndef _BASE_EXCEPTION_TPP_
#include "baseException.tpp"
#endif   // _BASE_EXCEPTION_TPP_

#endif   // _BASE_EXCEPTION_HPP_
