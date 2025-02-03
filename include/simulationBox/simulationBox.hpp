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
#include "typeAliases.hpp"       // for pq::Vec3D

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

        pq::Vec3D                 _centerOfMass = {0.0, 0.0, 0.0};
        pq::SharedAtomVec         _atoms;
        pq::SharedAtomVec         _qmAtoms;
        pq::SharedAtomVec         _qmCenterAtoms;
        std::vector<Molecule>     _molecules;
        std::vector<MoleculeType> _moleculeTypes;

        std::vector<size_t>      _externalGlobalVdwTypes;
        std::map<size_t, size_t> _externalToInternalGlobalVDWTypes;

       public:
        void                                         copy(const SimulationBox&);
        [[nodiscard]] std::shared_ptr<SimulationBox> clone() const;

        void checkCoulRadiusCutOff(const customException::ExceptionType) const;
        void setupExternalToInternalGlobalVdwTypesMap();

        void calculateDegreesOfFreedom();
        void calculateTotalMass();
        void calculateCenterOfMass();
        void calculateCenterOfMassMolecules();
        void calculateDensity();

        void updateOldPositions();
        void updateOldVelocities();
        void updateOldForces();

        void resetForces();

        void setPartialChargesOfMoleculesFromMoleculeTypes();
        void initPositions(const double displacement);

        [[nodiscard]] double    calculateTemperature();
        [[nodiscard]] double    calculateTotalForce();
        [[nodiscard]] pq::Vec3D calculateTotalForceVector();
        [[nodiscard]] double    calculateRMSForce() const;
        [[nodiscard]] double    calculateMaxForce() const;
        [[nodiscard]] double    calculateRMSForceOld() const;
        [[nodiscard]] double    calculateMaxForceOld() const;
        [[nodiscard]] pq::Vec3D calculateMomentum();
        [[nodiscard]] pq::Vec3D calculateAngularMomentum(const pq::Vec3D&);
        [[nodiscard]] pq::Vec3D calcBoxDimFromDensity() const;
        [[nodiscard]] pq::Vec3D calcShiftVector(const pq::Vec3D&) const;

        [[nodiscard]] bool moleculeTypeExists(const size_t) const;
        [[nodiscard]] std::vector<std::string> getUniqueQMAtomNames();

        [[nodiscard]] std::optional<Molecule> findMolecule(const size_t);
        [[nodiscard]] MoleculeType& findMoleculeType(const size_t moleculeType);
        [[nodiscard]] std::vector<MoleculeType> findNecessaryMoleculeTypes();

        [[nodiscard]] std::optional<size_t> findMoleculeTypeByString(
            const std::string& moleculeType
        ) const;
        [[nodiscard]] std::pair<Molecule*, size_t> findMoleculeByAtomIndex(
            const size_t atomIndex
        );

#ifdef WITH_MPI
        [[nodiscard]] std::vector<size_t> flattenAtomTypes();
        [[nodiscard]] std::vector<size_t> flattenMolTypes();
        [[nodiscard]] std::vector<size_t> flattenInternalGlobalVDWTypes();

        [[nodiscard]] std::vector<double> flattenVelocities();
        [[nodiscard]] std::vector<double> flattenForces();
        [[nodiscard]] std::vector<double> flattenPartialCharges();

        void deFlattenPositions(const std::vector<double>& positions);
        void deFlattenVelocities(const std::vector<double>& velocities);
        void deFlattenForces(const std::vector<double>& forces);
#endif

        /************************
         * QMMM related methods *
         ************************/

        void addQMCenterAtoms(const std::vector<int>& atomIndices);
        void setupQMOnlyAtoms(const std::vector<int>& atomIndices);
        void setupMMOnlyAtoms(const std::vector<int>& atomIndices);

        /************************
         * standard add methods *
         ************************/

        void addAtom(const pq::SharedAtom atom);
        void addQMAtom(const pq::SharedAtom atom);
        void addMolecule(const Molecule& molecule);
        void addMoleculeType(const MoleculeType& molecule);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] int        getWaterType() const;
        [[nodiscard]] int        getAmmoniaType() const;
        [[nodiscard]] size_t     getNumberOfMolecules() const;
        [[nodiscard]] size_t     getDegreesOfFreedom() const;
        [[nodiscard]] size_t     getNumberOfAtoms() const;
        [[nodiscard]] size_t     getNumberOfQMAtoms() const;
        [[nodiscard]] double     getTotalMass() const;
        [[nodiscard]] double     getTotalCharge() const;
        [[nodiscard]] double     getDensity() const;
        [[nodiscard]] pq::Vec3D& getCenterOfMass();

        [[nodiscard]] Atom&         getAtom(const size_t index);
        [[nodiscard]] Atom&         getQMAtom(const size_t index);
        [[nodiscard]] Molecule&     getMolecule(const size_t index);
        [[nodiscard]] MoleculeType& getMoleculeType(const size_t index);

        [[nodiscard]] std::vector<double> getAtomicScalarForces() const;
        [[nodiscard]] std::vector<double> getAtomicScalarForcesOld() const;

        [[nodiscard]] pq::SharedAtomVec&         getAtoms();
        [[nodiscard]] pq::SharedAtomVec&         getQMAtoms();
        [[nodiscard]] std::vector<Molecule>&     getMolecules();
        [[nodiscard]] std::vector<MoleculeType>& getMoleculeTypes();

        [[nodiscard]] std::vector<size_t>& getExternalGlobalVdwTypes();
        [[nodiscard]] std::map<size_t, size_t>& getExternalToInternalGlobalVDWTypes(
        );

        [[nodiscard]] Box&          getBox();
        [[nodiscard]] Box&          getBox() const;
        [[nodiscard]] pq::SharedBox getBoxPtr();
        [[nodiscard]] pq::SharedBox getBoxPtr() const;

        [[nodiscard]] std::vector<pq::Vec3D> getPositions() const;
        [[nodiscard]] std::vector<pq::Vec3D> getVelocities() const;
        [[nodiscard]] std::vector<pq::Vec3D> getForces() const;
        [[nodiscard]] std::vector<int>       getAtomicNumbers() const;
        [[nodiscard]] std::vector<double>    flattenPositions() const;

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
        void setBox(const T& box);

        /**********************************************
         * Forwards the box methods to the box object *
         **********************************************/

        void applyPBC(pq::Vec3D& position) const;
        void scaleBox(const pq::tensor3D& scalingTensor);

        [[nodiscard]] double calculateVolume() const;
        [[nodiscard]] double getMinimalBoxDimension() const;
        [[nodiscard]] double getVolume() const;

        [[nodiscard]] bool getBoxSizeHasChanged() const;

        [[nodiscard]] pq::Vec3D getBoxDimensions() const;
        [[nodiscard]] pq::Vec3D getBoxAngles() const;

        void setVolume(const double volume) const;
        void setBoxDimensions(const pq::Vec3D& boxDimensions) const;
        void setBoxSizeHasChanged(const bool boxSizeHasChanged) const;
    };

}   // namespace simulationBox

#include "simulationBox.tpp.hpp"   // DO NOT MOVE THIS LINE

#endif   // _SIMULATION_BOX_HPP_