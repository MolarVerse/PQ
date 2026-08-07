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

#ifndef _MOLECULE_HPP_

#define _MOLECULE_HPP_

#include <cstddef>       // for size_t
#include <map>           // for map
#include <memory>        // for shared_ptr
#include <string>        // for string
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "atom.hpp"   // for Atom

namespace simulationBox
{
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
        size_t      _numberOfAtoms;

        bool _isQMOnly = false;

        double _charge;   // set via molDescriptor not sum of partial charges!!!
        double _molMass;

        linearAlgebra::Vec3D _centerOfMass{0.0, 0.0, 0.0};

        std::map<size_t, size_t>           _externalToInternalAtomTypes;
        std::vector<std::shared_ptr<Atom>> _atoms;

       public:
        Molecule() = default;
        explicit Molecule(const std::string_view name);
        explicit Molecule(const size_t moltype);

        void calculateCenterOfMass(const Box &);
        void reconstructAtomsAroundCenterOfMass(const Box &);
        void scale(const linearAlgebra::tensor3D &, const Box &);
        void scaleVelocity(const linearAlgebra::tensor3D &, const Box &);

        [[nodiscard]] size_t              getNumberOfAtomTypes();
        [[nodiscard]] std::vector<size_t> getExternalGlobalVDWTypes() const;

        [[nodiscard]] std::vector<double> getAtomMasses() const;
        [[nodiscard]] std::vector<double> getPartialCharges() const;

        void setPartialCharges(const std::vector<double> &partialCharges);
        void setAtomForcesToZero();

        /****************************************
         * standard adder methods for atom data *
         *****************************************/

        void addAtom(const std::shared_ptr<Atom> atom);
        void addAtomPosition(
            const size_t                index,
            const linearAlgebra::Vec3D &position
        );
        void addAtomVelocity(
            const size_t                index,
            const linearAlgebra::Vec3D &velocity
        );
        void addAtomForce(
            const size_t                index,
            const linearAlgebra::Vec3D &force
        );
        void addAtomShiftForce(
            const size_t                index,
            const linearAlgebra::Vec3D &shiftForce
        );

        /*****************************************
         * standard setter methods for atom data *
         ****************************************/

        void setAtomPosition(
            const size_t                index,
            const linearAlgebra::Vec3D &position
        );
        void setAtomVelocity(
            const size_t                index,
            const linearAlgebra::Vec3D &velocity
        );
        void setAtomForce(
            const size_t                index,
            const linearAlgebra::Vec3D &force
        );
        void setAtomShiftForce(
            const size_t                index,
            const linearAlgebra::Vec3D &shiftForce
        );

        /****************************************
         * standard getters for atom properties *
         *****************************************/

        [[nodiscard]] linearAlgebra::Vec3D getAtomPosition(
            const size_t index
        ) const;
        [[nodiscard]] std::vector<linearAlgebra::Vec3D> getAtomPositions(
        ) const;
        [[nodiscard]] linearAlgebra::Vec3D getAtomVelocity(
            const size_t index
        ) const;
        [[nodiscard]] linearAlgebra::Vec3D getAtomForce(
            const size_t index
        ) const;
        [[nodiscard]] linearAlgebra::Vec3D getAtomShiftForce(
            const size_t index
        ) const;

        [[nodiscard]] int    getAtomicNumber(const size_t index) const;
        [[nodiscard]] double getAtomMass(const size_t index) const;
        [[nodiscard]] double getPartialCharge(const size_t index) const;
        [[nodiscard]] size_t getAtomType(const size_t index) const;
        [[nodiscard]] size_t getInternalGlobalVDWType(const size_t index) const;
        [[nodiscard]] std::string getAtomName(const size_t index) const;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getMoltype() const;
        [[nodiscard]] size_t getNumberOfAtoms() const;
        [[nodiscard]] size_t getDegreesOfFreedom() const;

        [[nodiscard]] bool isQMOnly() const;

        [[nodiscard]] double getCharge() const;
        [[nodiscard]] double getMolMass() const;

        [[nodiscard]] std::string getName() const;

        [[nodiscard]] linearAlgebra::Vec3D getCenterOfMass() const;

        [[nodiscard]] Atom &getAtom(const size_t index);
        [[nodiscard]] std::vector<std::shared_ptr<Atom>> &getAtoms();

        /***************************
         * standard setter methods *
         ***************************/

        void setName(const std::string_view name);
        void setQMOnly(const bool isQMOnly);

        void setNumberOfAtoms(const size_t numberOfAtoms);
        void setMoltype(const size_t moltype);

        void setCharge(const double charge);
        void setMolMass(const double molMass);
        void setCenterOfMass(const linearAlgebra::Vec3D &centerOfMass);
    };

}   // namespace simulationBox

#endif   // _MOLECULE_HPP_
