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

#include "infoOutput.hpp"

#include <format>   // for format

#include "constraintSettings.hpp"   // for ConstraintSettings
#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "manostatSettings.hpp"     // for ManostatSettings
#include "physicalData.hpp"         // for PhysicalData
#include "settings.hpp"             // for Settings
#include "thermostatSettings.hpp"   // for ThermostatSettings

namespace out
{

    /**
     * @brief write info file
     *
     * @details
     * - Coulomb and Non-Coulomb energies contain the intra and inter energies.
     * - Bond, Angle, Dihedral and Improper energies are only available if the
     * force field is active.
     * - qm energy is only available if qm is active.
     * - coulomb and non-coulomb energies are only available if mm is active.
     * - volume and density are only available if manostat is active.
     * - nose hoover momentum and friction energies are only available if nose
     * hoover thermostat is active.
     *
     * @param simulationTime
     * @param physicalData the physical data of the system
     */
    void InfoOutput::write(
        double                            simulationTime,
        const physicalData::PhysicalData &physicalData
    )
    {
        _fp.close();

        _fp.open(_fileName);

        _writeHeader();

        if (settings::Settings::isMDJobType())
            _writeLeft(simulationTime, "SIMULATION-TIME", "ps");
        else
            _writeLeftInteger(simulationTime, "EFFECTIVE STEPS", "-");

        _writeRight(physicalData.getTemperature(), "TEMPERATURE", "K");

        _writeLeft(physicalData.getPressure(), "PRESSURE", "bar");
        _writeRight(physicalData.getTotalEnergy(), "E(TOT)", "kcal/mol");

        if (settings::Settings::isQMActivated())
        {
            _writeLeft(physicalData.getQMEnergy(), "E(QM)", "kcal/mol");
            _writeRight(physicalData.getNumberOfQMAtoms(), "N(QM-ATOMS)", "-");
        }

        _writeLeft(physicalData.getKineticEnergy(), "E(KIN)", "kcal/mol");
        _writeRight(physicalData.getIntraEnergy(), "E(INTRA)", "kcal/mol");

        if (settings::Settings::isMMActivated())
        {
            _writeLeft(physicalData.getCoulombEnergy(), "E(COUL)", "kcal/mol");
            _writeRight(
                physicalData.getNonCoulombEnergy(),
                "E(NON-COUL)",
                "kcal/mol"
            );
        }

        if (settings::ForceFieldSettings::isActive())
        {
            _writeLeft(physicalData.getBondEnergy(), "E(BOND)", "kcal/mol");
            _writeRight(physicalData.getAngleEnergy(), "E(ANGLE)", "kcal/mol");
            _writeLeft(
                physicalData.getDihedralEnergy(),
                "E(DIHEDRAL)",
                "kcal/mol"
            );
            _writeRight(
                physicalData.getImproperEnergy(),
                "E(IMPROPER)",
                "kcal/mol"
            );
        }

        if (settings::Settings::isHybridJobtype())
        {
            _writeLeft(
                physicalData.getNumberOfSmoothingMolecules(),
                "N(SM-MOL)",
                "-"
            );
            _writeRight();
        }

        if (settings::ManostatSettings::getManostatType() != ManostatType::NONE)
        {
            _writeLeft(physicalData.getVolume(), "VOLUME", "A^3");
            _writeRight(physicalData.getDensity(), "DENSITY", "g/cm^3");

            if (settings::ManostatSettings::getFixedAxis() != FixedAxis::NONE)
            {
                _writeLeft(
                    physicalData.getCoupledPressure(),
                    "P(COUPLED)",
                    "bar"
                );
                _writeRight();
            }
        }

        if (settings::ThermostatSettings::getThermostatType() ==
            ThermostatType::NOSE_HOOVER)
        {
            _writeLeft(
                physicalData.getNoseHooverMomentumEnergy(),
                "E(NH-MOMENTUM)",
                "kcal/mol"
            );
            _writeRight(
                physicalData.getNoseHooverFrictionEnergy(),
                "E(NH-FRICTION)",
                "kcal/mol"
            );
        }

        if (settings::ConstraintSettings::isDistanceConstraintsActivated())
        {
            _writeLeft(
                physicalData.getLowerDistanceConstraints(),
                "LOWER-DIST-CONSTR",
                "kcal/mol"
            );
            _writeRight(
                physicalData.getUpperDistanceConstraints(),
                "UPPER-DIST-CONSTR",
                "kcal/mol"
            );
        }

        _writeLeftScientific(
            norm(physicalData.getMomentum()),
            "MOMENTUM",
            "amuA/fs"
        );
        _writeRight(physicalData.getLoopTime(), "LOOPTIME", "s");

        _fp << std::format("{:-^89}", "") << "\n\n";

        _fp.flush();
    }

    /**
     * @brief write header of info file
     *
     */
    void InfoOutput::_writeHeader()
    {
        _fp << std::format("{:-^89}", "") << '\n';

        _fp << '|' << std::format("{:^87}", "PQ info file") << '|' << '\n';

        _fp << std::format("{:-^89}", "") << '\n';
    }

    /**
     * @brief write left column of info file
     *
     * @param value
     * @param name
     * @param unit
     */
    void InfoOutput::_writeLeft(
        double                  value,
        const std::string_view &name,
        const std::string_view &unit
    )
    {
        _fp << std::format("|   {:<15} {:15.5f} {:<8} ", name, value, unit);
    }

    /**
     * @brief write left column of info file
     *
     * @param value
     * @param name
     * @param unit
     */
    void InfoOutput::_writeLeftInteger(
        double                  value,
        const std::string_view &name,
        const std::string_view &unit
    )
    {
        _fp << std::format(
            "|   {:<15} {:15d} {:<8} ",
            name,
            static_cast<int>(value),
            unit
        );
    }

    /**
     * @brief write left column of info file
     *
     * @param value
     * @param name
     * @param unit
     */
    void InfoOutput::_writeLeftScientific(
        double                  value,
        const std::string_view &name,
        const std::string_view &unit
    )
    {
        _fp << std::format("|   {:<15} {:15.1e} {:<8} ", name, value, unit);
    }

    /**
     * @brief write std::right column of info file
     *
     * @param value
     * @param name
     * @param unit
     */
    void InfoOutput::_writeRight(
        double                  value,
        const std::string_view &name,
        const std::string_view &unit
    )
    {
        _fp << std::format("{:<15} {:15.5f} {:<8}   |\n", name, value, unit);
    }

    /**
     * @brief write empty right column to info file
     *
     */
    void InfoOutput::_writeRight()
    {
        _fp << "                                           |\n";
    }

}   // namespace out
