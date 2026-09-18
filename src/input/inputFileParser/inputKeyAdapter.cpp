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

#include "inputKeyAdapter.hpp"

namespace input
{
    /**
     * @brief bridges an InputKeyBase onto the existing
     * InputFileParser::ParseFunc / addKeyword mechanism
     *
     * @details this is the entire integration surface between the new
     * InputKey<T>/InputRegistry design and InputFileParser/InputFileReader
     * -- neither of those classes needs to change. InputFileParser::ParseFunc
     * and InputKeyBase::parse are already structurally identical, so this
     * is a thin forwarding wrapper, not new parsing logic.
     *
     * @details deliberately a plain capturing lambda, not std::bind_front:
     * this project's parser infrastructure has hit a libc++ _Callable
     * trait incompatibility with std::bind_front elsewhere, so binding
     * utilities in this area use a plain lambda or the project's own
     * bindMember helper instead.
     *
     * @param key the InputKey<T> (type-erased via InputKeyBase) to adapt.
     * Must outlive every call through the returned ParseFunc -- the
     * registry that owns the key is expected to live for the lifetime of
     * the InputFileParser subclass that registers it, exactly as
     * InputKey<T> objects already do today.
     *
     * @return InputFileParser::ParseFunc
     */
    [[nodiscard]]
    InputFileParser::ParseFunc adapt(InputKeyBase &key)
    {
        return [&key](
                   const std::vector<std::string> &lineElements,
                   const std::size_t               lineNumber
               ) { key.parse(lineElements, lineNumber); };
    }
}   // namespace input
