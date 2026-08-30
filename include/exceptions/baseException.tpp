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

#ifndef _BASE_EXCEPTION_TPP_
#define _BASE_EXCEPTION_TPP_

#include <iostream>

#include "baseException.hpp"

namespace exc
{
    /**
     * @brief Construct a new Custom Exception:: Custom Exception object
     *
     * @param message
     */
    template <Color::Code colorCode, ExceptionType exceptionType>
    BaseException<colorCode, exceptionType>::BaseException(
        const std::string_view message,
        std::optional<size_t>  lineNumber
    )
        : PQException(message, lineNumber)
    {
    }

    /**
     * @brief Construct a new Custom Exception:: Custom Exception object
     *
     * @param message
     */
    template <Color::Code colorCode, ExceptionType exceptionType>
    BaseException<colorCode, exceptionType>::BaseException(
        const std::string_view message
    )
        : PQException(message, std::nullopt)
    {
    }

    /**
     * @brief Prints the exceptionMsg type in color.
     *
     * @param color
     * @param exceptionMsg
     */
    template <Color::Code colorCode, ExceptionType exceptionType>
    void BaseException<colorCode, exceptionType>::colorfulOutput(
        const Color::Code      color,
        const std::string_view exceptionMsg
    )
    {
        const Color::Modifier modifier(color);
        const Color::Modifier def(Color::FG_DEFAULT);

        std::cout << modifier << exceptionMsg << def << '\n' << std::flush;
    }

    /**
     * @brief Construct a new Custom Exception:: Custom Exception object
     *
     * @return const char*
     */
    template <Color::Code colorCode, ExceptionType exceptionType>
    const char *BaseException<colorCode, exceptionType>::what() const noexcept
    {
        if (exceptionType != ExceptionType::Undefined)
        {
            colorfulOutput(
                colorCode,
                ExceptionTypeMeta::toString(exceptionType)
            );
        }

        return PQException::getMessage().c_str();
    }
}   // namespace exc

#endif   // _BASE_EXCEPTION_TPP_
