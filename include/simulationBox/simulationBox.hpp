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

#include <map>        // for map
#include <optional>   // for optional
#include <string>     // for string
#include <vector>     // for vector

#include "atom.hpp"              // for Atom
#include "box.hpp"               // for Box
#include "defaults.hpp"          // for _COULOMB_CUT_OFF_DEFAULT_
#include "exceptions.hpp"        // for ExceptionType
#include "molecule.hpp"          // for Molecule
#include "moleculeType.hpp"      // for MoleculeType
#include "orthorhombicBox.hpp"   // for OrthorhombicBox
#include "triclinicBox.hpp"      // for TriclinicBox

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
    using Vec3D     = linearAlgebra::Vec3D;
    using vector3d  = std::vector<std::vector<std::vector<double>>>;
    using vector4d  = std::vector<vector3d>;
    using vector5d  = std::vector<vector4d>;
    using map_ul_ul = std::map<size_t, size_t>;

    using CustomException = customException::ExceptionType;

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
     *  The atoms positions, velocities and forces are stored in the
     * SimulationBox class. Additional molecular information is also stored in
     * the SimulationBox class.
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

        std::shared_ptr<Box> _box = std::make_shared<OrthorhombicBox>();

        Vec3D                              _centerOfMass = {0.0, 0.0, 0.0};
        std::vector<std::shared_ptr<Atom>> _atoms;
        std::vector<std::shared_ptr<Atom>> _qmAtoms;
        std::vector<std::shared_ptr<Atom>> _qmCenterAtoms;
        std::vector<Molecule>              _molecules;
        std::vector<MoleculeType>          _moleculeTypes;

        std::vector<size_t>      _externalGlobalVdwTypes;
        std::map<size_t, size_t> _externalToInternalGlobalVDWTypes;

       public:
        void copy(const SimulationBox &);
        [[nodiscard]] std::shared_ptr<SimulationBox> clone() const;

        void checkCoulombRadiusCutOff(const CustomException) const;
        void setupExternalToInternalGlobalVdwTypesMap();

        void calculateDegreesOfFreedom();
        void calculateTotalMass();
        void calculateCenterOfMass();
        void calculateCenterOfMassMolecules();
        void calculateDensity();

        void updateOldPositions();
        void updateOldVelocities();
        void updateOldForces();

        void setPartialChargesOfMoleculesFromMoleculeTypes();
        void initPositions(const double displacement);

        [[nodiscard]] double calculateTemperature();
        [[nodiscard]] double calculateTotalForce();
        [[nodiscard]] double calculateRMSForce() const;
        [[nodiscard]] double calculateMaxForce() const;
        [[nodiscard]] double calculateRMSForceOld() const;
        [[nodiscard]] double calculateMaxForceOld() const;
        [[nodiscard]] Vec3D  calculateMomentum();
        [[nodiscard]] Vec3D  calculateAngularMomentum(const Vec3D &);
        [[nodiscard]] Vec3D  calculateBoxDimensionsFromDensity() const;
        [[nodiscard]] Vec3D  calculateShiftVector(const Vec3D &position) const;

        [[nodiscard]] bool moleculeTypeExists(const size_t) const;
        [[nodiscard]] std::vector<std::string> getUniqueQMAtomNames();

        [[nodiscard]] std::optional<Molecule> findMolecule(const size_t);
        [[nodiscard]] MoleculeType &findMoleculeType(const size_t moleculeType);
        [[nodiscard]] std::vector<MoleculeType> findNecessaryMoleculeTypes();

        [[nodiscard]] std::optional<size_t> findMoleculeTypeByString(
            const std::string &moleculeType
        ) const;
        [[nodiscard]] std::pair<Molecule *, size_t> findMoleculeByAtomIndex(
            const size_t atomIndex
        );

#ifdef WITH_MPI
        [[nodiscard]] std::vector<size_t> flattenAtomTypes();
        [[nodiscard]] std::vector<size_t> flattenMolTypes();
        [[nodiscard]] std::vector<size_t> flattenInternalGlobalVDWTypes();

        [[nodiscard]] std::vector<double> flattenPositions();
        [[nodiscard]] std::vector<double> flattenVelocities();
        [[nodiscard]] std::vector<double> flattenForces();
        [[nodiscard]] std::vector<double> flattenPartialCharges();

        void deFlattenPositions(const std::vector<double> &positions);
        void deFlattenVelocities(const std::vector<double> &velocities);
        void deFlattenForces(const std::vector<double> &forces);
#endif

        /************************
         * QMMM related methods *
         ************************/

        void addQMCenterAtoms(const std::vector<int> &atomIndices);
        void setupQMOnlyAtoms(const std::vector<int> &atomIndices);
        void setupMMOnlyAtoms(const std::vector<int> &atomIndices);

        /************************
         * standard add methods *
         ************************/

        void addAtom(const std::shared_ptr<Atom> atom);
        void addQMAtom(const std::shared_ptr<Atom> atom);
        void addMolecule(const Molecule &molecule);
        void addMoleculeType(const MoleculeType &molecule);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] int    getWaterType() const;
        [[nodiscard]] int    getAmmoniaType() const;
        [[nodiscard]] size_t getNumberOfMolecules() const;
        [[nodiscard]] size_t getDegreesOfFreedom() const;
        [[nodiscard]] size_t getNumberOfAtoms() const;
        [[nodiscard]] size_t getNumberOfQMAtoms() const;
        [[nodiscard]] double getTotalMass() const;
        [[nodiscard]] double getTotalCharge() const;
        [[nodiscard]] double getDensity() const;
        [[nodiscard]] Vec3D &getCenterOfMass();

        [[nodiscard]] Atom         &getAtom(const size_t index);
        [[nodiscard]] Atom         &getQMAtom(const size_t index);
        [[nodiscard]] Molecule     &getMolecule(const size_t index);
        [[nodiscard]] MoleculeType &getMoleculeType(const size_t index);

        [[nodiscard]] std::vector<double> getAtomicScalarForces() const;
        [[nodiscard]] std::vector<double> getAtomicScalarForcesOld() const;

        [[nodiscard]] std::vector<std::shared_ptr<Atom>> &getAtoms();
        [[nodiscard]] std::vector<std::shared_ptr<Atom>> &getQMAtoms();
        [[nodiscard]] std::vector<Molecule>              &getMolecules();
        [[nodiscard]] std::vector<MoleculeType>          &getMoleculeTypes();

        [[nodiscard]] std::vector<size_t> &getExternalGlobalVdwTypes();
        [[nodiscard]] map_ul_ul &getExternalToInternalGlobalVDWTypes();

        [[nodiscard]] Box                 &getBox();
        [[nodiscard]] Box                 &getBox() const;
        [[nodiscard]] std::shared_ptr<Box> getBoxPtr();
        [[nodiscard]] std::shared_ptr<Box> getBoxPtr() const;

        /***************************
         * standard setter methods *
         ***************************/

        void setWaterType(const int waterType);
        void setAmmoniaType(const int ammoniaType);
        void setTotalMass(const double totalMass);
        void setTotalCharge(const double totalCharge);
        void setDensity(const double density);
        void setDegreesOfFreedom(const size_t degreesOfFreedom);

        template <typename T>
        void setBox(const T &box)
        {
            _box = std::make_shared<T>(box);
        }

        /**********************************************
         * Forwards the box methods to the box object *
         **********************************************/

        void applyPBC(Vec3D &position) const;
        void scaleBox(const tensor3D &scalingTensor);

        [[nodiscard]] double calculateVolume() const;
        [[nodiscard]] double getMinimalBoxDimension() const;
        [[nodiscard]] double getVolume() const;

        [[nodiscard]] bool getBoxSizeHasChanged() const;

        [[nodiscard]] Vec3D getBoxDimensions() const;
        [[nodiscard]] Vec3D getBoxAngles() const;

        void setVolume(const double volume) const;
        void setBoxDimensions(const Vec3D &boxDimensions) const;
        void setBoxSizeHasChanged(const bool boxSizeHasChanged) const;
    };

}   // namespace simulationBox

#endif   // _SIMULATION_BOX_HPP_