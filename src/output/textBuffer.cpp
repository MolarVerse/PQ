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

#include "textBuffer.hpp"

#include <charconv>
#include <cstddef>
#include <string_view>

namespace out
{
    /**
     * @brief format an integer into the scratch buffer
     */
    std::string_view TextBuffer::_toChars(std::size_t value)
    {
        const auto result = std::to_chars(
            _scratch.data(),
            _scratch.data() + SCRATCH_SIZE,
            value
        );

        return {_scratch.data(), result.ptr};
    }

    /**
     * @brief format a double into the scratch buffer
     */
    std::string_view TextBuffer::_toChars(
        double            value,
        std::chars_format format,
        int               precision
    )
    {
        const auto result = std::to_chars(
            _scratch.data(),
            _scratch.data() + SCRATCH_SIZE,
            value,
            format,
            precision
        );

        return {_scratch.data(), result.ptr};
    }

    /**
     * @brief format a double with the fewest digits that read back as the
     * same value into the scratch buffer
     */
    std::string_view TextBuffer::_toChars(
        double            value,
        std::chars_format format
    )
    {
        const auto result = std::to_chars(
            _scratch.data(),
            _scratch.data() + SCRATCH_SIZE,
            value,
            format
        );

        return {_scratch.data(), result.ptr};
    }

    /**
     * @brief append spaces so that a field of the given length fills width
     */
    void TextBuffer::_pad(std::size_t length, std::size_t width)
    {
        if (length < width)
            _data.append(width - length, ' ');
    }

    /**
     * @brief append one character
     */
    TextBuffer &TextBuffer::put(char character)
    {
        _data.push_back(character);
        return *this;
    }

    /**
     * @brief append text as it is
     */
    TextBuffer &TextBuffer::append(std::string_view text)
    {
        _data.append(text);
        return *this;
    }

    /**
     * @brief append text left aligned, like "{:<width}"
     */
    TextBuffer &TextBuffer::left(std::string_view text, std::size_t width)
    {
        _data.append(text);
        _pad(text.size(), width);
        return *this;
    }

    /**
     * @brief append an integer left aligned, like "{:<width}"
     */
    TextBuffer &TextBuffer::left(std::size_t value, std::size_t width)
    {
        return left(_toChars(value), width);
    }

    /**
     * @brief append a double right aligned, like "{:width.precisionf}"
     */
    TextBuffer &TextBuffer::fixed(
        double      value,
        std::size_t width,
        int         precision
    )
    {
        const auto text = _toChars(value, std::chars_format::fixed, precision);

        _pad(text.size(), width);
        _data.append(text);
        return *this;
    }

    /**
     * @brief append a double right aligned, like "{:width.precisione}"
     */
    TextBuffer &TextBuffer::scientific(
        double      value,
        std::size_t width,
        int         precision
    )
    {
        const auto text =
            _toChars(value, std::chars_format::scientific, precision);

        _pad(text.size(), width);
        _data.append(text);
        return *this;
    }

    /**
     * @brief append a double right aligned in fixed notation with the fewest
     * digits that read back as exactly the same value
     */
    TextBuffer &TextBuffer::exactFixed(double value, std::size_t width)
    {
        const auto text = _toChars(value, std::chars_format::fixed);

        _pad(text.size(), width);
        _data.append(text);
        return *this;
    }

    /**
     * @brief append a double right aligned in scientific notation with the
     * fewest digits that read back as exactly the same value
     */
    TextBuffer &TextBuffer::exactScientific(double value, std::size_t width)
    {
        const auto text = _toChars(value, std::chars_format::scientific);

        _pad(text.size(), width);
        _data.append(text);
        return *this;
    }

}   // namespace out
