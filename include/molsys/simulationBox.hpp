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

#include <memory>
#include <optional>   // for optional
#include <set>
#include <string>   // for string
#include <unordered_map>
#include <vector>   // for vector

#include "atom.hpp"                // for Atom
#include "box.hpp"                 // for Box
#include "molecule.hpp"            // for Molecule
#include "moleculeType.hpp"        // for MoleculeType
#include "orthorhombicBox.hpp"     // for OrthorhombicBox
#include "simulationBoxView.hpp"   // for SimulationBoxView
#include "strongTypes.hpp"

/**
 * @namespace molsys
 *
 * @brief contains class:
 *  SimulationBox
 *  Box
 *  CellList
 *  Cell
 *  Molecule
 *
 */
namespace molsys
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
    class SimulationBox : public SimulationBoxView<SimulationBox>
    {
       private:
        std::optional<size_t> _waterType;
        std::optional<size_t> _ammoniaType;

        size_t _degreesOfFreedom = 0;

        double _totalMass   = 0.0;
        double _totalCharge = 0.0;
        double _density     = 0.0;

        std::shared_ptr<Box> _box = std::make_shared<OrthorhombicBox>();

        linearAlgebra::Vec3D               _centerOfMass = {0.0, 0.0, 0.0};
        std::vector<std::shared_ptr<Atom>> _atoms;
        std::vector<int>                   _innerRegionCenterAtomIndices;
        std::vector<Molecule>              _molecules;
        std::vector<MoleculeType>          _moleculeTypes;

        std::vector<ExtVdwType> _externalGlobalVdwTypes;
        std::unordered_map<ExtVdwType, VdwType>
            _externalToInternalGlobalVDWTypes;

       public:
        void                                         copy(const SimulationBox&);
        [[nodiscard]] std::shared_ptr<SimulationBox> clone() const;

        void checkCoulRadiusCutOff(const ExceptionType) const;
        void setupExternalToInternalGlobalVdwTypesMap();

        void calculateDegreesOfFreedom();
        void calculateTotalMass();
        void calculateCenterOfMass();
        void calculateCenterOfMassMolecules();
        void calculateDensity();

        void updateOldPositions();
        void updateOldVelocities();
        void updateOldForces();

        void resetAllForces();
        void resetForces();
        void resetForcesInner();
        void resetForcesOuter();
        void resetQMCharges();
        void removeNetForce();

        void setPartialChargesOfMoleculesFromMoleculeTypes();
        void initPositions(const double displacement);

        [[nodiscard]] double               calculateTemperature();
        [[nodiscard]] double               calculateTotalForce();
        [[nodiscard]] linearAlgebra::Vec3D calculateTotalForceVector();
        [[nodiscard]] double               calculateRMSForce() const;
        [[nodiscard]] double               calculateMaxForce() const;
        [[nodiscard]] double               calculateRMSForceOld() const;
        [[nodiscard]] double               calculateMaxForceOld() const;
        [[nodiscard]] linearAlgebra::Vec3D calculateMomentum();
        [[nodiscard]] linearAlgebra::Vec3D calculateAngularMomentum(
            const linearAlgebra::Vec3D&
        );
        [[nodiscard]] linearAlgebra::Vec3D calcBoxDimFromDensity() const;
        [[nodiscard]] linearAlgebra::Vec3D calcShiftVector(
            const linearAlgebra::Vec3D& position
        ) const
        {
            return _box->calcShiftVector(position);
        }
        [[nodiscard]] int calcActiveMolCharge() const;

        [[nodiscard]] bool moleculeTypeExists(const size_t) const;

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

        void addInnerRegionCenterAtoms(const std::vector<int>& atomIndices);
        void setupForcedCoreMolecules(const std::vector<int>& moleculeIndices);
        void setupForcedLayerMolecules(const std::vector<int>& moleculeIndices);
        void setupForcedOuterMolecules(const std::vector<int>& moleculeIndices);

        /************************
         * standard add methods *
         ************************/

        void addAtom(const std::shared_ptr<Atom> atom);
        void addMolecule(const Molecule& molecule);
        void addMoleculeType(const MoleculeType& molecule);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] std::optional<size_t> getWaterType() const;
        [[nodiscard]] std::optional<size_t> getAmmoniaType() const;
        [[nodiscard]] size_t                getNumberOfMolecules() const;
        [[nodiscard]] size_t                getDegreesOfFreedom() const;
        [[nodiscard]] size_t                getNumberOfAtoms() const;
        [[nodiscard]] size_t                getNumberOfQMAtoms() const;
        [[nodiscard]] double                getTotalMass() const;
        [[nodiscard]] double                getTotalCharge() const;
        [[nodiscard]] double                getDensity() const;
        [[nodiscard]] linearAlgebra::Vec3D& getCenterOfMass();
        [[nodiscard]] std::vector<int>      getInnerRegionCenterAtomIndices();

        [[nodiscard]] Atom&         getAtom(const size_t index);
        [[nodiscard]] Molecule&     getMolecule(const size_t index);
        [[nodiscard]] MoleculeType& getMoleculeType(const size_t index);

        [[nodiscard]] std::vector<double> getAtomicScalarForces() const;
        [[nodiscard]] std::vector<double> getAtomicScalarForcesOld() const;

        [[nodiscard]] std::vector<std::shared_ptr<Atom>>&       getAtoms();
        [[nodiscard]] const std::vector<std::shared_ptr<Atom>>& getAtoms(
        ) const;
        [[nodiscard]] std::vector<Molecule>&       getMolecules();
        [[nodiscard]] const std::vector<Molecule>& getMolecules() const;
        [[nodiscard]] std::vector<MoleculeType>&   getMoleculeTypes();

        [[nodiscard]]
        std::vector<ExtVdwType>& getExternalGlobalVdwTypes();
        [[nodiscard]]
        std::unordered_map<
            ExtVdwType,
            VdwType>& getExternalToInternalGlobalVDWTypes();

        [[nodiscard]] Box&                 getBox();
        [[nodiscard]] Box&                 getBox() const;
        [[nodiscard]] std::shared_ptr<Box> getBoxPtr();
        [[nodiscard]] std::shared_ptr<Box> getBoxPtr() const;

        [[nodiscard]] std::vector<linearAlgebra::Vec3D> getPositions() const;
        [[nodiscard]] std::vector<linearAlgebra::Vec3D> getVelocities() const;
        [[nodiscard]] std::vector<linearAlgebra::Vec3D> getForces() const;
        [[nodiscard]] std::vector<AtomNumber> getAtomicNumbers() const;
        [[nodiscard]] std::vector<double>     flattenPositions() const;
        [[nodiscard]] std::set<std::string>   getUniqueQMAtomNames() const;
        [[nodiscard]] std::vector<double>     getFlattenedQMPositions() const;

        /***************************
         * standard setter methods *
         ***************************/

        void setWaterType(const size_t waterType);
        void setAmmoniaType(const size_t ammoniaType);
        void setTotalMass(const double totalMass);
        void setTotalCharge(const double totalCharge);
        void setDensity(const double density);
        void setDegreesOfFreedom(const size_t degreesOfFreedom);

        template <typename T>
        void setBox(const T& box);

        /**********************************************
         * Forwards the box methods to the box object *
         **********************************************/

        void applyPBC(linearAlgebra::Vec3D& position) const;
        void scaleBox(const linearAlgebra::tensor3D& scalingTensor);

        [[nodiscard]] double calculateVolume() const;
        [[nodiscard]] double getMinimalBoxDimension() const;
        [[nodiscard]] double getVolume() const;

        [[nodiscard]] bool getBoxSizeHasChanged() const;

        [[nodiscard]] linearAlgebra::Vec3D getBoxDimensions() const;
        [[nodiscard]] linearAlgebra::Vec3D getBoxAngles() const;

        void setVolume(const double volume) const;
        void setBoxDimensions(const linearAlgebra::Vec3D& boxDimensions) const;
        void setBoxSizeHasChanged(const bool boxSizeHasChanged) const;
    };

}   // namespace molsys

#ifndef _SIMULATION_BOX_TPP_
#include "simulationBox.tpp.hpp"   // IWYU pragma: export
#endif

#endif   // _SIMULATION_BOX_HPP_
