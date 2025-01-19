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

#include "simulationBox_base.hpp"

#include "molecule.hpp"   // IWYU pragma: keep
#include "settings.hpp"

using namespace settings;

namespace simulationBox
{

    /**
     * @brief calculate degrees of freedom
     *
     */
    void SimulationBoxBase::calculateDegreesOfFreedom()
    {
        _degreesOfFreedom = 3 * _nAtoms - Settings::getDimensionality();
    }

    /**
     * @brief add atom
     *
     * @param atom
     */
    void SimulationBoxBase::addAtom(const std::shared_ptr<Atom> atom)
    {
        _atoms.push_back(atom);
    }

    /**
     * @brief add molecule
     *
     * @param molecule
     */
    void SimulationBoxBase::addMolecule(const Molecule& molecule)
    {
        _molecules.push_back(molecule);
    }

    /**
     * @brief Get the number of atoms
     *
     * @return size_t
     */
    size_t SimulationBoxBase::getNumberOfAtoms() const { return _nAtoms; }

    /**
     * @brief Get the number of molecules
     *
     * @return size_t
     */
    size_t SimulationBoxBase::getNumberOfMolecules() const
    {
        return _molecules.size();
    }

    /**
     * @brief Get degrees of freedom
     *
     * @return size_t
     */
    size_t SimulationBoxBase::getDegreesOfFreedom() const
    {
        return _degreesOfFreedom;
    }

    /**
     * @brief Get total mass
     *
     * @return Real
     */
    Real SimulationBoxBase::getTotalMass() const { return _totalMass; }

    /**
     * @brief Get total charge
     *
     * @return Real
     */
    Real SimulationBoxBase::getTotalCharge() const { return _totalCharge; }

    /**
     * @brief Get center of mass
     *
     * @return pq::Vec3D
     */
    pq::Vec3D& SimulationBoxBase::getCenterOfMass() { return _centerOfMass; }

    /**
     * @brief get atoms
     *
     * @return atoms
     */
    pq::SharedAtomVec& SimulationBoxBase::getAtoms() { return _atoms; }

    /**
     * @brief get atom by index
     *
     * @param index
     * @return atom
     */
    Atom& SimulationBoxBase::getAtom(const size_t index)
    {
        return *(_atoms[index]);
    }

    /**
     * @brief get molecule by index
     *
     * @param index
     * @return Molecule&
     */
    Molecule& SimulationBoxBase::getMolecule(const size_t index)
    {
        return _molecules[index];
    }

    /**
     * @brief get molecules
     *
     * @return molecules
     */
    std::vector<Molecule>& SimulationBoxBase::getMolecules()
    {
        return _molecules;
    }

    /**
     * @brief get positions as Vec3D
     *
     * @return std::vector<pq::Vec3D>
     */
    std::vector<pq::Vec3D> SimulationBoxBase::getPositionsVec3D() const
    {
        std::vector<pq::Vec3D> positions;

        for (const auto& atom : _atoms)
            positions.push_back(atom->getPosition());

        return positions;
    }

    /**
     * @brief get velocities as Vec3D
     *
     * @return std::vector<pq::Vec3D>
     */
    std::vector<pq::Vec3D> SimulationBoxBase::getVelocitiesVec3D() const
    {
        std::vector<pq::Vec3D> velocities;

        for (const auto& atom : _atoms)
            velocities.push_back(atom->getVelocity());

        return velocities;
    }

    /**
     * @brief get forces as Vec3D
     *
     * @return std::vector<pq::Vec3D>
     */
    std::vector<pq::Vec3D> SimulationBoxBase::getForcesVec3D() const
    {
        std::vector<pq::Vec3D> forces;

        for (const auto& atom : _atoms)
            forces.push_back(atom->getForce());

        return forces;
    }

    /**
     * @brief Set the number of atoms
     *
     * @param nAtoms
     */
    void SimulationBoxBase::setNumberOfAtoms(const size_t nAtoms)
    {
        _nAtoms = nAtoms;
    }

    /**
     * @brief Set the number of molecules
     *
     * @param nMolecules
     */
    void SimulationBoxBase::setNumberOfMolecules(const size_t nMolecules)
    {
        _nMolecules = nMolecules;
    }

    /**
     * @brief Set the degrees of freedom
     *
     * @param degreesOfFreedom
     */
    void SimulationBoxBase::setDegreesOfFreedom(const size_t degreesOfFreedom)
    {
        _degreesOfFreedom = degreesOfFreedom;
    }

    /**
     * @brief Set the total mass
     *
     * @param totalMass
     */
    void SimulationBoxBase::setTotalMass(const Real totalMass)
    {
        _totalMass = totalMass;
    }

    /**
     * @brief Set the total charge
     *
     * @param totalCharge
     */
    void SimulationBoxBase::setTotalCharge(const Real totalCharge)
    {
        _totalCharge = totalCharge;
    }

    /**
     * @brief Set the center of mass
     *
     * @param centerOfMass
     */
    void SimulationBoxBase::setCenterOfMass(const pq::Vec3D& centerOfMass)
    {
        _centerOfMass = centerOfMass;
    }

}   // namespace simulationBox