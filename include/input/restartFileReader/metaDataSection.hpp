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

#ifndef INPUT__RESTART_FILE_READER__META_DATA_SECTION_HPP
#define INPUT__RESTART_FILE_READER__META_DATA_SECTION_HPP

#include <mstd/types.hpp>
#include <unordered_set>

#include "enums/rstMetaData.hpp"
#include "restartFileSection.hpp"

class GeneralSettings;

namespace input::restartFile
{
    /**
     * @brief Represents the metadata section of a restart file, providing
     * access to general settings and related information.
     *
     * This section typically contains metadata such as simulation parameters,
     * version information, and other global settings that are necessary for
     * correctly interpreting the restart file.
     *
     */
    class MetaDataSection : public RestartFileSection
    {
        mstd::Ref<GeneralSettings>              _generalSettings;
        std::unordered_set<RestartFileMetaData> _parsedKeywords;

       public:
        explicit MetaDataSection(GeneralSettings &generalSettings);

        [[nodiscard]]
        std::string keyword() override;

        void process(
            std::vector<std::string> &lineElements,
            engine::Engine           &engine,
            size_t                   &lineNumber
        ) override;

       private:
        void _setVersion(const std::string &version, size_t lineNumber);
    };
}   // namespace input::restartFile

#endif   // INPUT__RESTART_FILE_READER__META_DATA_SECTION_HPP
