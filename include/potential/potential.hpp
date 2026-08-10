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

#ifndef _POTENTIAL_HPP_

#define _POTENTIAL_HPP_

#include <memory>    // for shared_ptr
#include <utility>   // for pair

#include "coulombPotential.hpp"
#include "nonCoulombPotential.hpp"
#include "timer.hpp"

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace simulationBox
{
    class Box;             // forward declaration
    class Molecule;        // forward declaration
    class Atom;            // forward declaration
    class CellList;        // forward declaration
    class SimulationBox;   // forward declaration
}   // namespace simulationBox

namespace potential
{
    struct QMChargeTag
    {
    };

    struct MMChargeTag
    {
    };

    /**
     * @class Potential
     *
     * @brief base class for all potential routines
     *
     * @details
     * possible options:
     * - brute force
     * - cell list
     *
     * @note _nonCoulPairsVec is just a container to store the
     * nonCoulombicPairs for later processing
     *
     */
    class Potential : public timings::Timer
    {
       protected:
        std::shared_ptr<CoulombPotential>    _coulombPotential;
        std::shared_ptr<NonCoulombPotential> _nonCoulombPot;

       public:
        virtual ~Potential() = default;

        virtual void calculateForces(
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            simulationBox::CellList &
        ) = 0;

        void calculateQMMMForces(
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            simulationBox::CellList &
        );

        virtual void calculateCoreToOuterForces(
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            simulationBox::CellList &
        ) = 0;

        virtual void calculateLayerToOuterForces(
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            simulationBox::CellList &
        ) = 0;

        virtual void calculateOuterToOuterForces(
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            simulationBox::CellList &
        ) = 0;

        virtual void calculateHotspotSmoothingMMForces(
            simulationBox::SimulationBox &,
            physicalData::PhysicalData &,
            simulationBox::CellList &
        ) = 0;

        virtual std::shared_ptr<Potential> clone() const = 0;

        template <typename ChargeTag1, typename ChargeTag2>
        std::pair<double, double> calculateSingleInteraction(
            const simulationBox::Box &box,
            simulationBox::Molecule  &mol1,
            simulationBox::Molecule  &mol2,
            simulationBox::Atom      &atom1,
            simulationBox::Atom      &atom2
        ) const;

        template <typename ChargeTag1, typename ChargeTag2>
        double calculateSingleCoulombInteraction(
            const simulationBox::Box &box,
            simulationBox::Atom      &atom1,
            simulationBox::Atom      &atom2
        ) const;

        template <typename ChargeTag1, typename ChargeTag2>
        std::pair<double, double> calculateSingleInteractionOneWay(
            const simulationBox::Box &box,
            simulationBox::Molecule  &mol1,
            simulationBox::Molecule  &mol2,
            simulationBox::Atom      &atom1,
            simulationBox::Atom      &atom2
        ) const;

        template <typename T>
        void makeCoulombPotential(T p);

        template <typename T>
        void makeNonCoulombPotential(const T &nonCoulombPot);

        template <typename T>
        double getPartialCharge(simulationBox::Atom &atom) const;

        /***************************
         * standard setter methods *
         ***************************/

        void setNonCoulombPotential(
            const std::shared_ptr<NonCoulombPotential> pot
        );

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] CoulombPotential    &getCoulombPotential() const;
        [[nodiscard]] NonCoulombPotential &getNonCoulombPotential() const;
        [[nodiscard]]
        std::shared_ptr<CoulombPotential> getCoulombPotSharedPtr() const;
        [[nodiscard]]
        std::shared_ptr<NonCoulombPotential> getNonCoulombPotSharedPtr() const;
    };

}   // namespace potential

#ifndef _POTENTIAL_TPP_
#include "potential.tpp.hpp"   // IWYU pragma: keep - DO NOT MOVE THIS LINE
#endif

#endif   // _POTENTIAL_HPP_
