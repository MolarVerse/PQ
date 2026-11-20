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

#include "potential.hpp"

#include "coulombPotential.hpp"
#include "nonCoulombPotential.hpp"
#include "potentialKernels.tpp"   // IWYU pragma: keep -- for template specialization

namespace pot
{
    /**
     * @brief Construct a new Potential object with the given hybrid settings.
     *
     * @param hybridSettings Hybrid settings to be used by the potential.
     */
    Potential::Potential(const HybridSettings &hybridSettings)
        : _hybridSettings(hybridSettings)
    {
    }

    /**
     * @brief Destroy the Potential object
     *
     */
    Potential::~Potential() = default;

    template std::pair<EnergyPerMol, EnergyPerMol> pot::Potential::
        calculateSingleInteraction<pot::QMChargeTag, pot::MMChargeTag>(
            molsys::Box const &,
            molsys::Molecule &,
            molsys::Molecule &,
            molsys::Atom &,
            molsys::Atom &,
            const CoulombPotentialConfig &config
        ) const;

    template std::pair<EnergyPerMol, EnergyPerMol> pot::Potential::
        calculateSingleInteraction<pot::MMChargeTag, pot::QMChargeTag>(
            molsys::Box const &,
            molsys::Molecule &,
            molsys::Molecule &,
            molsys::Atom &,
            molsys::Atom &,
            const CoulombPotentialConfig &config
        ) const;

    template std::pair<EnergyPerMol, EnergyPerMol> pot::Potential::
        calculateSingleInteraction<pot::MMChargeTag, pot::MMChargeTag>(
            molsys::Box const &,
            molsys::Molecule &,
            molsys::Molecule &,
            molsys::Atom &,
            molsys::Atom &,
            const CoulombPotentialConfig &config
        ) const;

    template EnergyPerMol pot::Potential::
        calculateSingleCoulombInteraction<pot::QMChargeTag, pot::MMChargeTag>(
            molsys::Box const &,
            molsys::Atom &,
            molsys::Atom &,
            const CoulombPotentialConfig &config
        ) const;

    template EnergyPerMol pot::Potential::
        calculateSingleCoulombInteraction<pot::MMChargeTag, pot::QMChargeTag>(
            molsys::Box const &,
            molsys::Atom &,
            molsys::Atom &,
            const CoulombPotentialConfig &config
        ) const;

    template std::pair<EnergyPerMol, EnergyPerMol> pot::Potential::
        calculateSingleInteractionOneWay<pot::MMChargeTag, pot::QMChargeTag>(
            molsys::Box const &,
            molsys::Molecule &,
            molsys::Molecule &,
            molsys::Atom &,
            molsys::Atom &,
            const CoulombPotentialConfig &config
        ) const;

    template std::pair<EnergyPerMol, EnergyPerMol> pot::Potential::
        calculateSingleInteractionOneWay<pot::QMChargeTag, pot::MMChargeTag>(
            molsys::Box const &,
            molsys::Molecule &,
            molsys::Molecule &,
            molsys::Atom &,
            molsys::Atom &,
            const CoulombPotentialConfig &config
        ) const;

    /**
     * @brief calculates QMMM forces by invoking core-to-outer, layer-to-outer,
     * and outer-to-outer force calculations
     *
     * @param simulationBox
     * @param physicalData
     * @param cellList
     * @param waterModelSettings Reference to the WaterModelSettings object used
     * to configure water model-specific parameters
     * @param potentialSettings Reference to the PotentialSettings object used
     * to configure interaction cutoffs and other potential-related parameters
     */
    void Potential::calculateQMMMForces(
        molsys::SimulationBox      &simulationBox,
        physicalData::PhysicalData &physicalData,
        const molsys::CellList     &cellList,
        const WaterModelSettings   &waterModelSettings,
        const PotentialSettings    &potentialSettings
    )
    {
        calculateCoreToOuterForces(
            simulationBox,
            physicalData,
            cellList,
            waterModelSettings,
            potentialSettings
        );
        calculateLayerToOuterForces(
            simulationBox,
            physicalData,
            cellList,
            waterModelSettings,
            potentialSettings
        );
        calculateOuterToOuterForces(
            simulationBox,
            physicalData,
            cellList,
            waterModelSettings,
            potentialSettings
        );
    }

    /***************************
     *                         *
     * standard setter methods *
     *                         *
     ***************************/

    /**
     * @brief set the coulomb potential as a shared pointer
     *
     * @param pot
     */
    void Potential::setNonCoulombPotential(
        const std::shared_ptr<NonCoulombPotential> &pot
    )
    {
        _nonCoulombPot = pot;
    }

    /***************************
     *                         *
     * standard setter methods *
     *                         *
     ***************************/

    /**
     * @brief get the coulomb potential
     *
     * @return CoulombPotential&
     */
    CoulombPotential &Potential::getCoulombPotential() const
    {
        return *_coulombPotential;
    }

    /**
     * @brief get the non-coulomb potential
     *
     * @return NonCoulombPotential&
     */
    NonCoulombPotential &Potential::getNonCoulombPotential() const
    {
        return *_nonCoulombPot;
    }

    /**
     * @brief get the coulomb potential as a shared pointer
     *
     * @return std::shared_ptr<pot::CoulombPotential>
     */
    std::shared_ptr<CoulombPotential> Potential::getCoulombPotSharedPtr() const
    {
        return _coulombPotential;
    }

    /**
     * @brief get the non-coulomb potential as a shared pointer
     *
     * @return SharedNonCoulombPot
     */
    std::shared_ptr<NonCoulombPotential> Potential::getNonCoulombPotSharedPtr(
    ) const
    {
        return _nonCoulombPot;
    }

}   // namespace pot
