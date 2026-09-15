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

#include "baseException.hpp"

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
    PQException::PQException(const std::string_view message)
        : _message(message), _lineNumber(std::nullopt)
    {
    }

    /**
     * @brief Constructor for PQException with message and line number
     *
     * @param message The exception message
     * @param lineNumber The line number where the exception occurred (optional)
     */
    PQException::PQException(
        const std::string_view message,
        std::optional<size_t>  lineNumber
    )
        : _message(message), _lineNumber(lineNumber)
    {
    }

    /**
     * @brief Set the line number for the exception
     *
     * @param lineNumber The line number to set
     */
    void PQException::setLineNumber(const size_t lineNumber) noexcept
    {
        // TODO: Consider whether to allow overwriting the line number or not.
        // Currently, it only sets the line number if it hasn't been set before.
        // This is a very bad code smell here
        if (!_lineNumber.has_value())
            _lineNumber = lineNumber;
    }

    /**
     * @brief Get the exception message
     *
     * @return const std::string& The exception message
     */
    const std::string &PQException::getMessage() const noexcept
    {
        return _message;
    }

    /**
     * @brief Get the line number where the exception occurred
     *
     * @return std::optional<size_t> The line number (if set)
     */
    std::optional<size_t> PQException::getLineNumber() const noexcept
    {
        return _lineNumber;
    }

}   // namespace exc
