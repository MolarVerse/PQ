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

#ifndef _SIMULATION_BOX_HPP_

#define _SIMULATION_BOX_HPP_

#include "atom.hpp"              // for Atom
#include "box.hpp"               // for Box
#include "defaults.hpp"          // for _COULOMB_CUT_OFF_DEFAULT_
#include "exceptions.hpp"        // for ExceptionType
#include "molecule.hpp"          // for Molecule
#include "moleculeType.hpp"      // for MoleculeType
#include "orthorhombicBox.hpp"   // for OrthorhombicBox
#include "triclinicBox.hpp"      // for TriclinicBox

#include <map>        // for map
#include <optional>   // for optional
#include <string>     // for string
#include <vector>     // for vector

/**
 * @namespace simulationBox
 *
 * @brief contains class:
 *  SimulationBox
 *  Box
 *  CellList
 *  Cell
 *  Molecule
 *
 */
namespace simulationBox
{
    using c_ul      = const size_t;
    using vector4d  = std::vector<std::vector<std::vector<std::vector<double>>>>;
    using vector5d  = std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>;
    using map_ul_ul = std::map<size_t, size_t>;

    /**
     * @class SimulationBox
     *
     * @brief
     *
     *  contains all particles and the simulation box
     *
     * @details
     *
     *  The SimulationBox class contains all particles and the simulation box.
     *  The atoms positions, velocities and forces are stored in the SimulationBox class.
     *  Additional molecular information is also stored in the SimulationBox class.
     *
     */
    class SimulationBox
    {
      private:
        int _waterType;
        int _ammoniaType;

        size_t _degreesOfFreedom = 0;

        double _totalMass   = 0.0;
        double _totalCharge = 0.0;
        double _density     = 0.0;

        linearAlgebra::Vec3D _centerOfMass = {0.0, 0.0, 0.0};
        std::shared_ptr<Box> _box          = std::make_shared<OrthorhombicBox>();

        std::vector<std::shared_ptr<Atom>> _atoms;
        std::vector<std::shared_ptr<Atom>> _qmAtoms;
        std::vector<Molecule>              _molecules;
        std::vector<MoleculeType>          _moleculeTypes;

        std::vector<size_t>      _externalGlobalVdwTypes;
        std::map<size_t, size_t> _externalToInternalGlobalVDWTypes;

      public:
        void copy(const SimulationBox &);

        void checkCoulombRadiusCutOff(const customException::ExceptionType) const;
        void setupExternalToInternalGlobalVdwTypesMap();

        void calculateDegreesOfFreedom();
        void calculateTotalMass();
        void calculateCenterOfMass();
        void calculateCenterOfMassMolecules();
        void calculateDensity();

        void setPartialChargesOfMoleculesFromMoleculeTypes();

        void initPositions(const double displacement);

#ifdef WITH_MPI
        std::vector<double> flattenPositions();
        std::vector<double> flattenVelocities();
        std::vector<double> flattenForces();

        void deFlattenPositions(const std::vector<double> &positions);
        void deFlattenVelocities(const std::vector<double> &velocities);
        void deFlattenForces(const std::vector<double> &forces);
#endif

        [[nodiscard]] double               calculateTemperature();
        [[nodiscard]] double               calculateTotalForce();
        [[nodiscard]] linearAlgebra::Vec3D calculateMomentum();
        [[nodiscard]] linearAlgebra::Vec3D calculateAngularMomentum(const linearAlgebra::Vec3D &momentum);

        [[nodiscard]] linearAlgebra::Vec3D calculateBoxDimensionsFromDensity() const;
        [[nodiscard]] linearAlgebra::Vec3D calculateShiftVector(const linearAlgebra::Vec3D &position) const;

        [[nodiscard]] bool                     moleculeTypeExists(const size_t) const;
        [[nodiscard]] std::vector<std::string> getUniqueQMAtomNames();

        [[nodiscard]] std::optional<Molecule>       findMolecule(const size_t moleculeType);
        [[nodiscard]] MoleculeType                 &findMoleculeType(const size_t moleculeType);
        [[nodiscard]] std::optional<size_t>         findMoleculeTypeByString(const std::string &moleculeType) const;
        [[nodiscard]] std::pair<Molecule *, size_t> findMoleculeByAtomIndex(const size_t atomIndex);
        [[nodiscard]] std::vector<MoleculeType>     findNecessaryMoleculeTypes();

        /************************
         * standard add methods *
         ************************/

        void addAtom(const std::shared_ptr<Atom> atom) { _atoms.push_back(atom); }
        void addQMAtom(const std::shared_ptr<Atom> atom) { _qmAtoms.push_back(atom); }
        void addMolecule(const Molecule &molecule) { _molecules.push_back(molecule); }
        void addMoleculeType(const MoleculeType &molecule) { _moleculeTypes.push_back(molecule); }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] int                   getWaterType() const { return _waterType; }
        [[nodiscard]] int                   getAmmoniaType() const { return _ammoniaType; }
        [[nodiscard]] size_t                getNumberOfMolecules() const { return _molecules.size(); }
        [[nodiscard]] size_t                getDegreesOfFreedom() const { return _degreesOfFreedom; }
        [[nodiscard]] size_t                getNumberOfAtoms() const { return _atoms.size(); }
        [[nodiscard]] size_t                getNumberOfQMAtoms() const { return _qmAtoms.size(); }
        [[nodiscard]] double                getTotalMass() const { return _totalMass; }
        [[nodiscard]] double                getTotalCharge() const { return _totalCharge; }
        [[nodiscard]] double                getDensity() const { return _density; }
        [[nodiscard]] linearAlgebra::Vec3D &getCenterOfMass() { return _centerOfMass; }

        [[nodiscard]] Atom         &getAtom(const size_t index) { return *(_atoms[index]); }
        [[nodiscard]] Atom         &getQMAtom(const size_t index) { return *(_qmAtoms[index]); }
        [[nodiscard]] Molecule     &getMolecule(const size_t index) { return _molecules[index]; }
        [[nodiscard]] MoleculeType &getMoleculeType(const size_t index) { return _moleculeTypes[index]; }

        [[nodiscard]] std::vector<std::shared_ptr<Atom>> &getAtoms() { return _atoms; }
        [[nodiscard]] std::vector<std::shared_ptr<Atom>> &getQMAtoms() { return _qmAtoms; }
        [[nodiscard]] std::vector<Molecule>              &getMolecules() { return _molecules; }
        [[nodiscard]] std::vector<MoleculeType>          &getMoleculeTypes() { return _moleculeTypes; }

        [[nodiscard]] std::vector<size_t> &getExternalGlobalVdwTypes() { return _externalGlobalVdwTypes; }
        [[nodiscard]] map_ul_ul           &getExternalToInternalGlobalVDWTypes() { return _externalToInternalGlobalVDWTypes; }

        [[nodiscard]] Box                 &getBox() { return *_box; }
        [[nodiscard]] Box                 &getBox() const { return *_box; }
        [[nodiscard]] std::shared_ptr<Box> getBoxPtr() { return _box; }
        [[nodiscard]] std::shared_ptr<Box> getBoxPtr() const { return _box; }

        /***************************
         * standard setter methods *
         ***************************/

        void setWaterType(const int waterType) { _waterType = waterType; }
        void setAmmoniaType(const int ammoniaType) { _ammoniaType = ammoniaType; }
        void setTotalMass(const double totalMass) { _totalMass = totalMass; }
        void setTotalCharge(const double totalCharge) { _totalCharge = totalCharge; }
        void setDensity(const double density) { _density = density; }
        void setDegreesOfFreedom(const size_t degreesOfFreedom) { _degreesOfFreedom = degreesOfFreedom; }

        template <typename T>
        void setBox(const T &box)
        {
            _box = std::make_shared<T>(box);
        }

        /**********************************************
         * Forwards the box methods to the box object *
         **********************************************/

        void applyPBC(linearAlgebra::Vec3D &position) const { _box->applyPBC(position); }
        void scaleBox(const linearAlgebra::tensor3D &scalingTensor)
        {
            _box->scaleBox(scalingTensor);
            calculateDensity();
        }

        [[nodiscard]] double calculateVolume() const { return _box->calculateVolume(); }
        [[nodiscard]] double getMinimalBoxDimension() const { return _box->getMinimalBoxDimension(); }
        [[nodiscard]] double getVolume() const { return _box->getVolume(); }

        [[nodiscard]] bool getBoxSizeHasChanged() const { return _box->getBoxSizeHasChanged(); }

        [[nodiscard]] linearAlgebra::Vec3D getBoxDimensions() const { return _box->getBoxDimensions(); }
        [[nodiscard]] linearAlgebra::Vec3D getBoxAngles() const { return _box->getBoxAngles(); }

        void setVolume(const double volume) const { _box->setVolume(volume); }
        void setBoxDimensions(const linearAlgebra::Vec3D &boxDimensions) const { _box->setBoxDimensions(boxDimensions); }
        void setBoxSizeHasChanged(const bool boxSizeHasChanged) const { _box->setBoxSizeHasChanged(boxSizeHasChanged); }
    };

}   // namespace simulationBox

#endif   // _SIMULATION_BOX_HPP_