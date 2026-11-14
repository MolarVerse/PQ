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

#ifndef OUTPUT__TEXT_BUFFER_HPP
#define OUTPUT__TEXT_BUFFER_HPP

#include <array>
#include <charconv>
#include <cstddef>
#include <string>
#include <string_view>

namespace out
{
    /**
     * @brief reusable text buffer that formats numbers with std::to_chars
     *
     * @details gives the same bytes as std::format with "{:<w}", "{:w.pf}"
     * and "{:w.pe}", but keeps its memory between frames
     */
    class TextBuffer
    {
       public:
        // fits any double in fixed notation (~310 digits) plus precision
        static constexpr std::size_t SCRATCH_SIZE = 512;

       private:
        std::string                    _data;
        std::array<char, SCRATCH_SIZE> _scratch{};

        void             _pad(std::size_t length, std::size_t width);
        std::string_view _toChars(std::size_t value);
        std::string_view _toChars(
            double value,
            std::chars_format,
            int precision
        );

       public:
        void clear() noexcept { _data.clear(); }

        [[nodiscard]] std::string_view view() const noexcept { return _data; }

        TextBuffer &put(char character);
        TextBuffer &append(std::string_view text);
        TextBuffer &left(std::string_view text, std::size_t width);
        TextBuffer &left(std::size_t value, std::size_t width);
        TextBuffer &fixed(double value, std::size_t width, int precision);
        TextBuffer &scientific(double value, std::size_t width, int precision);
    };

}   // namespace out

#endif   // OUTPUT__TEXT_BUFFER_HPP
