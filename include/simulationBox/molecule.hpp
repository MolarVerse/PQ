/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#ifndef _MOLECULE_HPP_

#define _MOLECULE_HPP_

#include "atom.hpp"
#include "vector3d.hpp"

#include <algorithm>
#include <cstddef>   // for size_t
#include <map>
#include <memory>
#include <string>
#include <string_view>   // for string_view
#include <vector>

namespace simulationBox
{
    using c_ul = const size_t;

    /**
     * @class Molecule
     *
     * @brief containing all information about a molecule
     */
    class Molecule
    {
      private:
        std::string _name;
        size_t      _moltype;

        size_t _numberOfAtoms;

        double _charge;   // set via molDescriptor not sum of partial charges!!!
        double _molMass;

        std::map<size_t, size_t> _externalToInternalAtomTypes;

        std::vector<std::shared_ptr<Atom>> _atoms;

        linearAlgebra::Vec3D _centerOfMass = linearAlgebra::Vec3D(0.0, 0.0, 0.0);

      public:
        Molecule() = default;
        explicit Molecule(const std::string_view name) : _name(name){};
        explicit Molecule(c_ul moltype) : _moltype(moltype){};

        void calculateCenterOfMass(const linearAlgebra::Vec3D &);
        void scale(const linearAlgebra::Vec3D &);

        [[nodiscard]] size_t getNumberOfAtomTypes();

        void addAtom(const std::shared_ptr<Atom> atom) { _atoms.push_back(atom); }

        // TODO: check if these are really necessary

        [[nodiscard]] std::string getAtomName(c_ul index) const { return _atoms[index]->getName(); }

        void setAtomPosition(c_ul index, const linearAlgebra::Vec3D &position) { _atoms[index]->setPosition(position); }
        void setAtomVelocity(c_ul index, const linearAlgebra::Vec3D &velocity) { _atoms[index]->setVelocity(velocity); }
        void setAtomForce(c_ul index, const linearAlgebra::Vec3D &force) { _atoms[index]->setForce(force); }
        void setAtomShiftForce(c_ul index, const linearAlgebra::Vec3D &shiftForce) { _atoms[index]->setShiftForce(shiftForce); }

        void setAtomForcesToZero()
        {
            std::ranges::for_each(_atoms, [](auto atom) { atom->setForce({0.0, 0.0, 0.0}); });
        }

        void addAtomPosition(c_ul index, const linearAlgebra::Vec3D &position) { _atoms[index]->addPosition(position); }
        void addAtomVelocity(c_ul index, const linearAlgebra::Vec3D &velocity) { _atoms[index]->addVelocity(velocity); }
        void addAtomForce(c_ul index, const linearAlgebra::Vec3D &force) { _atoms[index]->addForce(force); }
        void addAtomShiftForce(c_ul index, const linearAlgebra::Vec3D &shiftForce) { _atoms[index]->addShiftForce(shiftForce); }

        [[nodiscard]] linearAlgebra::Vec3D getAtomPosition(c_ul index) const { return _atoms[index]->getPosition(); }
        [[nodiscard]] linearAlgebra::Vec3D getAtomVelocity(c_ul index) const { return _atoms[index]->getVelocity(); }
        [[nodiscard]] linearAlgebra::Vec3D getAtomForce(c_ul index) const { return _atoms[index]->getForce(); }
        [[nodiscard]] linearAlgebra::Vec3D getAtomShiftForce(c_ul index) const { return _atoms[index]->getShiftForce(); }

        [[nodiscard]] int    getAtomicNumber(c_ul index) const { return _atoms[index]->getAtomicNumber(); }
        [[nodiscard]] double getAtomMass(c_ul index) const { return _atoms[index]->getMass(); }
        [[nodiscard]] double getPartialCharge(c_ul index) const { return _atoms[index]->getPartialCharge(); }
        [[nodiscard]] size_t getAtomType(c_ul index) const { return _atoms[index]->getAtomType(); }
        [[nodiscard]] size_t getInternalGlobalVDWType(c_ul index) const { return _atoms[index]->getInternalGlobalVDWType(); }

        void setPartialCharges(const std::vector<double> &partialCharges)
        {
            for (size_t i = 0; i < getNumberOfAtoms(); ++i)
                _atoms[i]->setPartialCharge(partialCharges[i]);
        }

        [[nodiscard]] std::vector<double> getPartialCharges() const
        {
            std::vector<double> partialCharges(getNumberOfAtoms());

            for (size_t i = 0; i < getNumberOfAtoms(); ++i)
                partialCharges[i] = _atoms[i]->getPartialCharge();

            return partialCharges;
        }

        [[nodiscard]] std::vector<double> getAtomMasses() const
        {
            std::vector<double> atomMasses(getNumberOfAtoms());

            for (size_t i = 0; i < getNumberOfAtoms(); ++i)
                atomMasses[i] = _atoms[i]->getMass();

            return atomMasses;
        }

        [[nodiscard]] std::vector<size_t> getExternalGlobalVDWTypes()
        {
            std::vector<size_t> externalGlobalVDWTypes(getNumberOfAtoms());

            for (size_t i = 0; i < getNumberOfAtoms(); ++i)
                externalGlobalVDWTypes[i] = _atoms[i]->getExternalGlobalVDWType();

            return externalGlobalVDWTypes;
        }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getMoltype() const { return _moltype; }
        [[nodiscard]] size_t getNumberOfAtoms() const { return _numberOfAtoms; }
        [[nodiscard]] size_t getDegreesOfFreedom() const { return 3 * getNumberOfAtoms(); }

        [[nodiscard]] double getCharge() const { return _charge; }
        [[nodiscard]] double getMolMass() const { return _molMass; }

        [[nodiscard]] std::string getName() const { return _name; }

        [[nodiscard]] Atom                               &getAtom(c_ul index) { return *(_atoms[index]); }
        [[nodiscard]] std::vector<std::shared_ptr<Atom>> &getAtoms() { return _atoms; }

        [[nodiscard]] linearAlgebra::Vec3D getCenterOfMass() const { return _centerOfMass; }

        /***************************
         * standard setter methods *
         ***************************/

        void setNumberOfAtoms(c_ul numberOfAtoms) { _numberOfAtoms = numberOfAtoms; }

        void setName(const std::string_view name) { _name = name; }
        void setMoltype(c_ul moltype) { _moltype = moltype; }
        void setCharge(const double charge) { _charge = charge; }
        void setMolMass(const double molMass) { _molMass = molMass; }
        void setCenterOfMass(const linearAlgebra::Vec3D &centerOfMass) { _centerOfMass = centerOfMass; }
    };

}   // namespace simulationBox

#endif   // _MOLECULE_HPP_
