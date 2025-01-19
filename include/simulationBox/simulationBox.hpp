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
#include <memory>     // for shared_ptr
#include <optional>   // for optional
#include <string>     // for string
#include <vector>     // for vector

#include "atom.hpp"                 // for Atom
#include "box.hpp"                  // for Box
#include "exceptions.hpp"           // for ExceptionType
#include "molecule.hpp"             // for Molecule
#include "moleculeType.hpp"         // for MoleculeType
#include "orthorhombicBox.hpp"      // for OrthorhombicBox
#include "simulationBox_base.hpp"   // for SimulationBoxBase
#include "typeAliases.hpp"          // for pq::Vec3D

#ifdef __PQ_GPU__
    #include "coordinates_GPU.hpp"         // for CoordinatesGPU
    #include "device.hpp"                  // for Device
    #include "simulationBox_SoA_GPU.hpp"   // for SimulationBoxSoAGPU
#else
    #include "coordinates.hpp"         // for Coordinates
    #include "simulationBox_SoA.hpp"   // for SimulationBoxSoA
#endif

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
#ifdef __PQ_GPU__
    using _Coordinates      = CoordinatesGPU;
    using _SimulationBoxSoA = SimulationBoxSoAGPU;
#else
    using _Coordinates      = Coordinates;
    using _SimulationBoxSoA = SimulationBoxSoA;
#endif

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
    class SimulationBox : public SimulationBoxBase,
                          public _SimulationBoxSoA,
                          public _Coordinates
    {
       private:
        int _waterType;
        int _ammoniaType;

        double _density = 0.0;

        std::shared_ptr<Box> _box = std::make_shared<OrthorhombicBox>();

        pq::SharedAtomVec         _qmAtoms;
        pq::SharedAtomVec         _qmCenterAtoms;
        std::vector<MoleculeType> _moleculeTypes;

        std::vector<size_t>      _externalGlobalVdwTypes;
        std::map<size_t, size_t> _externalToInternalGlobalVDWTypes;

       public:
        void                                         copy(const SimulationBox&);
        [[nodiscard]] std::shared_ptr<SimulationBox> clone() const;

#ifdef __PQ_GPU__
        ~SimulationBox() override;
        void initDeviceMemory(device::Device& device);
#endif
        void resizeHostVectors(cul nAtoms, cul nMolecules) override;

        void checkCoulRadiusCutOff(const customException::ExceptionType) const;
        void setupExternalToInternalGlobalVdwTypesMap();

        void calculateDensity();

        void scaleVelocities(const double factor);
        void addToVelocities(const pq::Vec3D& velocity);

        void updateOldPositions();
        void updateOldVelocities();
        void updateOldForces();

        void resetForces();

        void setPartialChargesOfMoleculesFromMoleculeTypes();
        void initPositions(const double displacement);

        [[nodiscard]] double    calculateTotalForce();
        [[nodiscard]] double    calculateRMSForce();
        [[nodiscard]] double    calculateMaxForce();
        [[nodiscard]] double    calculateRMSForceOld();
        [[nodiscard]] double    calculateMaxForceOld();
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

        void flattenVelocities();
        void deFlattenPositions();
        void flattenPositions();
        void deFlattenVelocities();
        void flattenForces();
        void deFlattenForces();
        void flattenShiftForces();
        void deFlattenShiftForces();

        void flattenMasses();
        void flattenMolMasses();
        void flattenCharges();
        void flattenComMolecules();
        void deFlattenComMolecules();

        void initAtomsPerMolecule();
        void initMoleculeIndices();
        void initMoleculeOffsets();

#ifndef __PQ_LEGACY__
        void flattenOldPositions();
        void flattenOldVelocities();
        void flattenOldForces();
        void flattenAtomTypes();
        void flattenMolTypes();
        void flattenInternalGlobalVDWTypes();

        void deFlattenOldPositions();
        void deFlattenOldVelocities();
        void deFlattenOldForces();
#endif

#ifdef WITH_MPI
        [[nodiscard]] std::vector<size_t> flattenAtomTypes();
        [[nodiscard]] std::vector<size_t> flattenMolTypes();
        [[nodiscard]] std::vector<size_t> flattenInternalGlobalVDWTypes();

        [[nodiscard]] std::vector<double> flattenPartialCharges();
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

        void addQMAtom(const pq::SharedAtom atom);
        void addMoleculeType(const MoleculeType& molecule);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] int    getWaterType() const;
        [[nodiscard]] int    getAmmoniaType() const;
        [[nodiscard]] size_t getNumberOfQMAtoms() const;
        [[nodiscard]] double getDensity() const;

        [[nodiscard]] Atom&         getQMAtom(const size_t index);
        [[nodiscard]] MoleculeType& getMoleculeType(const size_t index);

        [[nodiscard]] std::vector<Real> getAtomicScalarForces();
        [[nodiscard]] std::vector<Real> getAtomicScalarForcesOld();

        [[nodiscard]] pq::SharedAtomVec&         getQMAtoms();
        [[nodiscard]] std::vector<MoleculeType>& getMoleculeTypes();

        [[nodiscard]] std::vector<size_t>& getExternalGlobalVdwTypes();
        [[nodiscard]] std::map<size_t, size_t>& getExternalToInternalGlobalVDWTypes(
        );

        [[nodiscard]] Box&          getBox();
        [[nodiscard]] Box&          getBox() const;
        [[nodiscard]] pq::SharedBox getBoxPtr();
        [[nodiscard]] pq::SharedBox getBoxPtr() const;

        [[nodiscard]] std::vector<int> getAtomicNumbers() const;

        /***************************
         * standard setter methods *
         ***************************/

        void setWaterType(const int waterType);
        void setAmmoniaType(const int ammoniaType);
        void setDensity(const double density);

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

#include "simulationBox.tpp.hpp"   // IWYU pragma: keep DO NOT MOVE THIS LINE!!!

#endif   // _SIMULATION_BOX_HPP_