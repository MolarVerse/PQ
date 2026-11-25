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

#include "metaDataSection.hpp"

#include <mstd/exceptions.hpp>

#include "enums/rstMetaData.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "settings.hpp"
#include "stringUtilities.hpp"

namespace input::restartFile
{
    /**
     * @brief Returns the keyword identifying the metadata section in the
     * restart file.
     *
     * @return The keyword string "META_DATA".
     */
    std::string MetaDataSection::keyword()
    {
        return utilities::toLowerCopy(std::string(RestartFileMetaDataKeyWord));
    }

    /**
     * @brief Processes the metadata section of the restart file.
     *
     * @param lineElements The elements of the current line.
     * @param engine The engine object (unused).
     * @param lineNumber The current line number in the restart file.
     *
     * @throws exc::RstFileException if the metadata section is malformed or
     * contains unrecognized or duplicate keys.
     */
    void MetaDataSection::process(
        std::vector<std::string> &lineElements,
        size_t                   &lineNumber,
        Settings                 &settings,
        molsys::SimulationBox & /* simulationBox */
    )
    {
        if (lineElements.empty() ||
            utilities::toLowerAndReplaceDashesCopy(lineElements[0]) !=
                keyword())
        {
            throw exc::RstFileException(
                "Expected keyword " + keyword() + " at line " +
                std::to_string(lineNumber)
            );
        }

        // for now put a single constraint on the number of elements expected in
        // the metadata line
        if (lineElements.size() != 3)
        {
            throw exc::RstFileException(
                "Expected 3 elements for " + keyword() + " at line " +
                std::to_string(lineNumber)
            );
        }

        const auto key =
            utilities::toLowerAndReplaceDashesCopy(lineElements[1]);

        const auto enumOptValue =
            RestartFileMetaDataMeta::from_stringCaseInsensitive(key);

        if (!enumOptValue.has_value())
        {
            throw exc::RstFileException(
                "Unrecognized metadata key '" + lineElements[1] + "' at line " +
                std::to_string(lineNumber)
            );
        }

        const auto enumValue = enumOptValue.value();

        if (_parsedKeywords.contains(enumValue))
        {
            throw exc::RstFileException(
                "Duplicate metadata key '" + lineElements[1] + "' at line " +
                std::to_string(lineNumber)
            );
        }

        _parsedKeywords.insert(enumValue);

        switch (enumValue)
        {
            case RestartFileMetaData::Version:
                _setVersion(lineElements[2], lineNumber, settings);
                break;
        }
    }

    /**
     * @brief Parses the version string and stores it as restart version.
     *
     * @param version The version string, e.g. "v1.2.3".
     * @param lineNumber The current line number in the restart file.
     *
     * @throws exc::RstFileException if the version string is not a valid
     * version.
     */
    void MetaDataSection::_setVersion(
        const std::string &version,
        size_t             lineNumber,
        Settings          &settings
    )
    {
        try
        {
            settings.generalSettings.setRestartVersionByUser(
                SemVer::from_string(version)
            );
        }
        catch (const mstd::InvalidArgument &)
        {
            throw exc::RstFileException(
                "Invalid version '" + version + "' at line " +
                std::to_string(lineNumber) +
                ", expected format 'vMAJOR.MINOR.PATCH'"
            );
        }
    }

}   // namespace input::restartFile
