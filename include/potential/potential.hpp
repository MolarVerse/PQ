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

#include <cstddef>   // for size_t
#include <memory>    // for shared_ptr, __shared_ptr_access, make_shared
#include <utility>   // for pair

#include "vector3d.hpp"   // for Vec3D

namespace physicalData
{
    class PhysicalData;
}

namespace simulationBox
{
    class CellList;
    class Molecule;
    class SimulationBox;
    class Box;
}   // namespace simulationBox

namespace potential
{
    class CoulombPotential;      // forward declaration
    class NonCoulombPotential;   // forward declaration

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
     * @note _nonCoulombPairsVector is just a container to store the
     * nonCoulombicPairs for later processing
     *
     */
    class Potential
    {
       protected:
        std::shared_ptr<CoulombPotential>    _coulombPotential;
        std::shared_ptr<NonCoulombPotential> _nonCoulombPotential;

       public:
        virtual ~Potential() = default;

        virtual void
        calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) = 0;

        std::pair<double, double> calculateSingleInteraction(
            const simulationBox::Box &,
            simulationBox::Molecule &,
            simulationBox::Molecule &,
            const size_t,
            const size_t
        ) const;

        template <typename T>
        void makeCoulombPotential(T p)
        {
            _coulombPotential = std::make_shared<T>(p);
        }

        template <typename T>
        void makeNonCoulombPotential(T nonCoulombPotential)
        {
            _nonCoulombPotential = std::make_shared<T>(nonCoulombPotential);
        }

        void setNonCoulombPotential(
            std::shared_ptr<NonCoulombPotential> nonCoulombPotential
        )
        {
            _nonCoulombPotential = nonCoulombPotential;
        }

        [[nodiscard]] CoulombPotential &getCoulombPotential() const
        {
            return *_coulombPotential;
        }
        [[nodiscard]] NonCoulombPotential &getNonCoulombPotential() const
        {
            return *_nonCoulombPotential;
        }
        [[nodiscard]] std::shared_ptr<CoulombPotential> getCoulombPotentialSharedPtr(
        ) const
        {
            return _coulombPotential;
        }
        [[nodiscard]] std::shared_ptr<NonCoulombPotential> getNonCoulombPotentialSharedPtr(
        ) const
        {
            return _nonCoulombPotential;
        }
    };

    /**
     * @class PotentialBruteForce
     *
     * @brief brute force implementation of the potential
     *
     */
    class PotentialBruteForce : public Potential
    {
       public:
        void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &)
            override;
    };

    /**
     * @class PotentialCellList
     *
     * @brief cell list implementation of the potential
     *
     */
    class PotentialCellList : public Potential
    {
       public:
        void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &)
            override;
    };

}   // namespace potential

#endif   // _POTENTIAL_HPP_