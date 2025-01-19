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

#ifndef __SIMULATION_BOX_BASE_HPP__
#define __SIMULATION_BOX_BASE_HPP__

#include "molecule.hpp"
#include "typeAliases.hpp"

namespace simulationBox
{
    /**
     * @class SimulationBoxBase
     *
     * @brief
     *
     *  Base class for the SimulationBox class. Simulation
     * Box class inherits from this class.
     *
     */
    class SimulationBoxBase
    {
       protected:
        size_t _nAtoms           = 0;
        size_t _nMolecules       = 0;
        size_t _degreesOfFreedom = 0;

        Real _totalMass   = 0.0;
        Real _totalCharge = 0.0;

        pq::Vec3D _centerOfMass = {0.0, 0.0, 0.0};

        pq::SharedAtomVec     _atoms;
        std::vector<Molecule> _molecules;

       public:
        void calculateDegreesOfFreedom();

        void addAtom(const std::shared_ptr<Atom> atom);
        void addMolecule(const Molecule& molecule);

        [[nodiscard]] pq::SharedAtomVec&     getAtoms();
        [[nodiscard]] Atom&                  getAtom(const size_t index);
        [[nodiscard]] std::vector<Molecule>& getMolecules();
        [[nodiscard]] Molecule&              getMolecule(const size_t index);

        [[nodiscard]] std::vector<pq::Vec3D> getForcesVec3D() const;
        [[nodiscard]] std::vector<pq::Vec3D> getVelocitiesVec3D() const;
        [[nodiscard]] std::vector<pq::Vec3D> getPositionsVec3D() const;

        [[nodiscard]] size_t     getNumberOfAtoms() const;
        [[nodiscard]] size_t     getNumberOfMolecules() const;
        [[nodiscard]] size_t     getDegreesOfFreedom() const;
        [[nodiscard]] Real       getTotalMass() const;
        [[nodiscard]] Real       getTotalCharge() const;
        [[nodiscard]] pq::Vec3D& getCenterOfMass();

        void setNumberOfAtoms(const size_t nAtoms);
        void setNumberOfMolecules(const size_t nMolecules);
        void setDegreesOfFreedom(const size_t degreesOfFreedom);
        void setTotalMass(const Real totalMass);
        void setTotalCharge(const Real totalCharge);
        void setCenterOfMass(const pq::Vec3D& centerOfMass);
    };

}   // namespace simulationBox

#endif   // __SIMULATION_BOX_BASE_HPP__
