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
     * @brief Adapts an InputKeyBase into an InputFileParser::ParseFunc.
     *
     * Provides integration between the InputKey<T>/InputRegistry design
     * and the existing InputFileParser::ParseFunc callback mechanism,
     * allowing InputKeyBase::parse (type-erased key handler) to be
     * registered with InputFileParser::addKeyword without modifying
     * InputFileParser or InputFileReader.
     *
     * Creates a callback that forwards InputFileParser::ParseFunc calls
     * to the corresponding InputKeyBase::parse method. Both signatures
     * accept (const std::vector<std::string>&, size_t lineNumber) and
     * are structurally identical, making this a simple forwarding wrapper.
     *
     * @param key The InputKeyBase instance to adapt. Typically obtained
     * from an InputKey<T> that has been type-erased. Must remain valid
     * for every invocation of the returned ParseFunc. The registry owning
     * @p key is expected to outlive the InputFileParser instance that
     * registers the adapted callback.
     *
     * @return InputFileParser::ParseFunc that delegates parse requests to
     * @p key. The returned callable captures a reference to @p key and
     * forwards (lineElements, lineNumber) arguments to key.parse().
     *
     * @note Implemented as a capturing lambda rather than std::bind_front
     * to avoid libc++ _Callable trait incompatibility issues encountered
     * elsewhere in the parser infrastructure.
     */
    [[nodiscard]]
    InputFileParser::ParseFunc adapt(InputKeyBase &key)
    {
        return [&key](
                   const std::vector<std::string> &lineElements,
                   size_t                          lineNumber
               ) { key.parse(lineElements, lineNumber); };
    }

    /**
     * @brief Adapts a DeprecatedInputKey into an InputFileParser::ParseFunc.
     *
     * Provides integration between the DeprecatedInputKey and the existing
     * InputFileParser::ParseFunc callback mechanism, allowing
     * DeprecatedInputKey::parse to be registered with
     * InputFileParser::addKeyword without modifying InputFileParser or
     * InputFileReader.
     *
     * Creates a callback that forwards InputFileParser::ParseFunc calls
     * to the corresponding DeprecatedInputKey::parse method. Both signatures
     * accept (const std::vector<std::string>&, size_t lineNumber) and
     * are structurally identical, making this a simple forwarding wrapper.
     *
     * @param deprecatedKey The DeprecatedInputKey instance to adapt. Copied
     * into the returned callable, since callers typically construct it as a
     * local temporary that would otherwise not outlive the registration.
     *
     * @return InputFileParser::ParseFunc that delegates parse requests to
     * a copy of @p deprecatedKey and forwards (lineElements, lineNumber)
     * arguments to deprecatedKey.parse().
     */
    [[nodiscard]]
    InputFileParser::ParseFunc adapt(const DeprecatedInputKey &deprecatedKey)
    {
        return [deprecatedKey](
                   const std::vector<std::string> & /*lineElements*/,
                   size_t lineNumber
               ) { deprecatedKey.deprecated(lineNumber); };
    }
}   // namespace input
