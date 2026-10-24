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

#include "simulationBoxInputParser.hpp"

#include "defaults.hpp"
#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "potentialSettings.hpp"
#include "rangeValidator.hpp"
#include "simulationBoxSettings.hpp"

namespace input
{

    /**
     * @brief Construct a new Input File Parser Simulation Box:: Input File
     * Parser Simulation Box object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) rcoulomb "<double>" 2)
     * density
     * "<double>"
     *
     */
    SimulationBoxInputParser::SimulationBoxInputParser()
    {
        addCoulombRadiusKey();
        addNonCoulombRadiusKey();
        addDensityKey();
        addInitializeVelocitiesKey();
    }

    /**
     * @brief adds the coulomb radius key to the key registry
     */
    void SimulationBoxInputParser::addCoulombRadiusKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "rcoulomb",
            .title = "Coulomb cutoff radius",
            .description =
                "Specifies the cutoff radius for Coulomb interactions"
        };

        const auto setValue = [](double value)
        { settings::PotentialSettings::setCoulombRadiusCutOff(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata     = metaData,
                .defaultValue = defaults::COULOMB_CUT_OFF_DEFAULT,
                .onSet        = setValue,
                .validators   = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief adds the non-coulomb radius key to the key registry
     */
    void SimulationBoxInputParser::addNonCoulombRadiusKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "rnoncoulomb",
            .title = "Non-Coulomb cutoff radius",
            .description =
                "Specifies the cutoff radius for Non-Coulomb interactions"
        };

        const auto setValue = [](double value)
        { settings::PotentialSettings::setNonCoulombRadiusCutOff(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief adds the density key to the key registry
     */
    void SimulationBoxInputParser::addDensityKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "density",
            .title       = "Density of the simulation box",
            .description = "Specifies the density of the simulation box"
        };

        const auto setValue = [](double value)
        {
            settings::SimulationBoxSettings::setDensity(value);
            settings::SimulationBoxSettings::setDensitySet(true);
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief adds the initialize velocities key to the key registry
     */
    void SimulationBoxInputParser::addInitializeVelocitiesKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "init_velocities",
            .title = "Initialize velocities",
            .description =
                "Specifies if velocities should be initialized with "
                "Maxwell-Boltzmann distribution"
        };

        const auto setValue = [](InitVelocities value)
        { settings::SimulationBoxSettings::setInitializeVelocities(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<InitVelocities>{
                .metadata     = metaData,
                .defaultValue = InitVelocities::FALSE,
                .onSet        = setValue
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

}   // namespace input
