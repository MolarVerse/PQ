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

#include "integratorInputParser.hpp"

#include <format>   // for format

#include "exceptions.hpp"   // for InputFileException, customException
#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "references.hpp"         // for ReferencesOutput
#include "referencesOutput.hpp"   // for ReferencesOutput
#include "settings.hpp"           // for Settings

namespace input
{

    /**
     * @brief Construct a new Input File Parser Integrator:: Input File Parser
     * Integrator object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) integrator "<string>"
     */
    IntegratorInputParser::IntegratorInputParser() { addIntegratorKey(); }

    /**
     * @brief Add the integrator key to the input file parser
     *
     * @details This function adds the "integrator" key to the input file parser
     * along with its metadata.
     */
    void IntegratorInputParser::addIntegratorKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "integrator",
            .title = "Integrator Type",
            .description =
                "Specifies the integrator type to be used in the simulation",
        };

        const auto setValue = [](settings::IntegratorType integratorType)
        {
            // TODO: remove this via general setup
            if (!settings::Settings::isMDJobType())
            {
                throw exc::InputFileException(
                    std::format(
                        "Integrator is only supported for MD simulations!"
                    )
                );
            }

            settings::Settings::setIntegratorType(integratorType);
            references::ReferencesOutput::addReferenceFile(
                references::VELOCITY_VERLET_FILE
            );
        };

        auto& key = _getRegistry().registerKey(
            KeyRegistry<settings::IntegratorType>{
                .metadata     = metaData,
                .defaultValue = settings::IntegratorType::VELOCITY_VERLET,
                .onSet        = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }
}   // namespace input
