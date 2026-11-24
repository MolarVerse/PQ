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

#include "energyColumns.hpp"

#include <format>

#include "constraintSettings.hpp"
#include "forceFieldSettings.hpp"
#include "generalSettings.hpp"
#include "manostatSettings.hpp"
#include "physicalData.hpp"
#include "settings.hpp"
#include "thermostatSettings.hpp"
#include "units.hpp"
#include "version.hpp"

namespace out
{
    /**
     * @brief collect the active energy columns in output order
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
     * @param physicalData the physical data of the system
     * @param settings the simulation settings
     */
    EnergyColumns::EnergyColumns(
        const physicalData::PhysicalData &physicalData,
        const Settings                   &settings
    )
    {
        using enum EnergyColumnFormat;

        constexpr std::string_view kcal = "kcal/mol";

        _add({
            .name  = "TEMPERATURE",
            .unit  = "K",
            .value = physicalData.get_temperature()
                         .quantity_from_zero()
                         .numerical_value_in(K),
        });
        _add({
            .name  = "PRESSURE",
            .unit  = "bar",
            .value = physicalData.get_pressure().numerical_value_in(bar),
        });
        _add({
            .name = "E(TOT)",
            .unit = kcal,
            .value =
                physicalData.getTotalEnergy().numerical_value_in(kcal_per_mol),
        });

        if (isQMActivated(settings.generalSettings.getJobtype()))
        {
            _add({
                .name  = "E(QM)",
                .unit  = kcal,
                .value = (physicalData.get_qmEnergy() -
                          physicalData.get_qmReferenceEnergy())
                             .numerical_value_in(kcal_per_mol),
            });
            _add({
                .name  = "N(QM-ATOMS)",
                .unit  = "-",
                .value = physicalData.get_nQMAtoms(),
            });
        }

        _add({
            .name  = "E(KIN)",
            .unit  = kcal,
            .value = physicalData.get_kineticEnergy().numerical_value_in(
                kcal_per_mol
            ),
        });
        _add({
            .name = "E(INTRA)",
            .unit = kcal,
            .value =
                physicalData.getIntraEnergy().numerical_value_in(kcal_per_mol),
        });

        if (isMMActivated(settings.generalSettings.getJobtype()))
        {
            _add({
                .name  = "E(COUL)",
                .unit  = kcal,
                .value = physicalData.get_coulombEnergy().numerical_value_in(
                    kcal_per_mol
                ),
            });
            _add({
                .name  = "E(NON-COUL)",
                .unit  = kcal,
                .value = physicalData.get_nonCoulombEnergy().numerical_value_in(
                    kcal_per_mol
                ),
            });
        }

        if (settings.forceFieldSettings.isActive())
        {
            _add({
                .name  = "E(BOND)",
                .unit  = kcal,
                .value = physicalData.get_bondEnergy().numerical_value_in(
                    kcal_per_mol
                ),
            });
            _add({
                .name  = "E(ANGLE)",
                .unit  = kcal,
                .value = physicalData.get_angleEnergy().numerical_value_in(
                    kcal_per_mol
                ),
            });
            _add({
                .name  = "E(DIHEDRAL)",
                .unit  = kcal,
                .value = physicalData.get_dihedralEnergy().numerical_value_in(
                    kcal_per_mol
                ),
            });
            _add({
                .name  = "E(IMPROPER)",
                .unit  = kcal,
                .value = physicalData.get_improperEnergy().numerical_value_in(
                    kcal_per_mol
                ),
            });
        }

        if (isHybridJobtype(settings.generalSettings.getJobtype()))
        {
            _add({
                .name  = "N(SM-MOL)",
                .unit  = "-",
                .value = physicalData.get_numberOfSmoothingMol(),
            });
            _add({
                .name  = "E(QM-REF)",
                .unit  = kcal,
                .value = physicalData.get_qmReferenceEnergy()
                             .numerical_value_in(kcal_per_mol),
            });
        }

        if (settings.manostatSettings.getManostatType() != ManostatType::NONE)
        {
            _add({
                .name = "VOLUME",
                .unit = "A^3",
                .value =
                    physicalData.get_volume().numerical_value_in(angstrom3),
            });
            _add({
                .name = "DENSITY",
                .unit = "g/cm^3",
                .value =
                    physicalData.get_density().numerical_value_in(kg_per_litre),
            });

            if (settings.manostatSettings.getFixedAxis() != FixedAxis::NONE)
            {
                _add({
                    .name  = "P(COUPLED)",
                    .unit  = "bar",
                    .value = physicalData.get_coupledPressure()
                                 .numerical_value_in(bar),
                    .aloneInInfoRow = true,
                });
            }
        }

        if (settings.thermostatSettings.getThermostatType() ==
            ThermostatType::NOSE_HOOVER)
        {
            _add({
                .name  = "E(NH-MOMENTUM)",
                .unit  = kcal,
                .value = physicalData.get_noseHooverMomentumEnergy()
                             .numerical_value_in(kcal_per_mol),
            });
            _add({
                .name  = "E(NH-FRICTION)",
                .unit  = kcal,
                .value = physicalData.get_noseHooverFrictionEnergy()
                             .numerical_value_in(one),
            });
        }

        if (settings.constraintSettings.isDistanceConstraintsActivated())
        {
            _add({
                .name  = "LOWER-DIST-CONSTR",
                .unit  = kcal,
                .value = physicalData.get_lowerDistanceConstraints()
                             .numerical_value_in(kcal_per_mol),
            });
            _add({
                .name  = "UPPER-DIST-CONSTR",
                .unit  = kcal,
                .value = physicalData.get_upperDistanceConstraints()
                             .numerical_value_in(kcal_per_mol),
            });
        }

        _add({
            .name  = "MOMENTUM",
            .unit  = "amuA/fs",
            .value = norm(physicalData.get_momentum())
                         .numerical_value_in(g_per_mol * angstrom / fs),
            .format = SCIENTIFIC,
        });
        _add({
            .name   = "LOOPTIME",
            .unit   = "s",
            .value  = physicalData.get_loopTime().numerical_value_in(s),
            .format = TIME,
        });
    }

    /**
     * @brief get the active columns
     *
     * @return std::span<const EnergyColumn>
     */
    std::span<const EnergyColumn> EnergyColumns::get() const
    {
        return std::span<const EnergyColumn>(_columns).first(_size);
    }

    /**
     * @brief append a column
     *
     * @param column
     *
     * @throw std::out_of_range if more than maxColumns columns are added
     */
    void EnergyColumns::_add(const EnergyColumn &column)
    {
        _columns.at(_size) = column;
        ++_size;
    }

    /**
     * @brief column label with unit, e.g. "E(TOT)[kcal/mol]"
     *
     * @param column
     * @return std::string
     */
    std::string energyColumnLabel(const EnergyColumn &column)
    {
        return std::format("{}[{}]", column.name, column.unit);
    }

    /**
     * @brief first line of the .en file, marks it as a PQ energy file
     *
     * @return std::string without trailing newline
     */
    std::string energyFileMarker()
    {
        return std::format("# PQ energy file {}", CURRENT_VERSION.to_string());
    }

    /**
     * @brief column names line of the .en file, one token per data column
     *
     * @details "#STEP[-] TEMPERATURE[K] ...": the '#' is glued
     * to the first token, so token i is column i also in a spreadsheet;
     * names and units are those of the info file, so readers can use it
     * instead
     *
     * @param columns
     * @return std::string without trailing newline
     */
    std::string energyFileHeader(const EnergyColumns &columns)
    {
        auto header = std::format("#{}", energyStepLabel);

        for (const auto &column : columns.get())
        {
            header += ' ';
            header += energyColumnLabel(column);
        }

        return header;
    }

}   // namespace out
