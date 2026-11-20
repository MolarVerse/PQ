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

#ifndef MOLSYS__SIMULATION_BOX_HPP
#define MOLSYS__SIMULATION_BOX_HPP

#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "atom.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "simulationBoxView.hpp"
#include "strongTypes.hpp"
#include "unit_types.hpp"

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
        std::optional<MolType> _waterType;
        std::optional<MolType> _ammoniaType;

        size_t _degreesOfFreedom = 0;

        Mass    _totalMass   = Mass::zero();
        Charge  _totalCharge = Charge::zero();
        Density _density     = Density::zero();

        std::shared_ptr<Box> _box;
        bool                 _boxSetViaStartFile = false;

        Pos3D                              _centerOfMass = {Pos::zero()};
        std::vector<std::shared_ptr<Atom>> _atoms;
        std::vector<Molecule>              _molecules;
        std::vector<MoleculeType>          _moleculeTypes;

        // cumulative number of atoms per molecule (0, n0, n0 + n1, ...), only
        // filled between buildAtomIndexLookup() and clearAtomIndexLookup()
        std::vector<size_t> _atomIndexOffsets;

        std::vector<ExtVdwType> _externalGlobalVdwTypes;
        std::unordered_map<ExtVdwType, VdwType>
            _externalToInternalGlobalVDWTypes;

       public:
        SimulationBox();

        void                                         copy(const SimulationBox&);
        [[nodiscard]] std::shared_ptr<SimulationBox> clone() const;

        void checkCoulRadiusCutOff(
            const ExceptionType&,
            const Distance& coulombRadiusCutOff
        ) const;
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
        void initPositions(double displacement);

        [[nodiscard]] Temperature calculateTemperature();
        [[nodiscard]] Force       calculateTotalForce() const;
        [[nodiscard]] Force3D     calculateTotalForceVector() const;
        [[nodiscard]] Force       calculateRMSForce() const;
        [[nodiscard]] Force       calculateMaxForce() const;
        [[nodiscard]] Force       calculateRMSForceOld() const;
        [[nodiscard]] Force       calculateMaxForceOld() const;
        [[nodiscard]] Momentum3D  calculateMomentum();
        [[nodiscard]]
        AngularMomentum3D   calculateAngularMomentum(const Momentum3D&);
        [[nodiscard]] Pos3D calcBoxDimFromDensity() const;
        [[nodiscard]] Pos3D calcShiftVector(const Pos3D& position) const;
        [[nodiscard]] int   calcActiveMolCharge() const;

        [[nodiscard]]
        bool moleculeTypeExists(MolType moleculeType) const;
        [[nodiscard]]
        std::optional<Molecule> findMolecule(MolType moleculeType);
        [[nodiscard]]
        MoleculeType& findMoleculeType(MolType molType);
        [[nodiscard]]
        std::vector<MoleculeType> findNecessaryMoleculeTypes();
        [[nodiscard]]
        std::optional<MolType> findMoleculeTypeByString(
            const std::string& moleculeType
        ) const;
        [[nodiscard]]
        std::pair<Molecule*, AtomIndex> findMoleculeByGlobalAtomIndex(
            size_t atomIndex
        );
        void buildAtomIndexLookup();
        void clearAtomIndexLookup();

        [[nodiscard]] std::vector<Pos3D> flattenPositions() const;
        [[nodiscard]] std::vector<Force> flattenForces() const;

#ifdef WITH_MPI
        [[nodiscard]] std::vector<Vel>    flattenVelocities() const;
        [[nodiscard]] std::vector<Charge> flattenPartialCharges() const;

        void deFlattenPositions(const std::vector<Pos>& positions);
        void deFlattenVelocities(const std::vector<Vel>& velocities);
        void deFlattenForces(const std::vector<Force>& forces);
#endif

        /************************
         * QMMM related methods *
         ************************/

        void setupForcedCoreMolecules(const std::vector<int>& moleculeIndices);
        void setupForcedLayerMolecules(const std::vector<int>& moleculeIndices);
        void setupForcedOuterMolecules(const std::vector<int>& moleculeIndices);

        /************************
         * standard add methods *
         ************************/

        void addAtom(const std::shared_ptr<Atom>& atom);
        void addMolecule(const Molecule& molecule);
        void addMoleculeType(const MoleculeType& molecule);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] std::optional<MolType> getWaterType() const;
        [[nodiscard]] std::optional<MolType> getAmmoniaType() const;
        [[nodiscard]] size_t                 getNumberOfMolecules() const;
        [[nodiscard]] size_t                 getDegreesOfFreedom() const;
        [[nodiscard]] size_t                 getNumberOfAtoms() const;
        [[nodiscard]] size_t                 getNumberOfQMAtoms() const;
        [[nodiscard]] const Mass&            getTotalMass() const;
        [[nodiscard]] const Charge&          getTotalCharge() const;
        [[nodiscard]] const Density&         getDensity() const;
        [[nodiscard]] const Pos3D&           getCenterOfMass() const;
        [[nodiscard]] std::vector<size_t>    getInnerRegionCenterAtomIndices();

        [[nodiscard]] Atom&         getAtom(size_t index);
        [[nodiscard]] Molecule&     getMolecule(size_t index);
        [[nodiscard]] MoleculeType& getMoleculeType(size_t index);

        [[nodiscard]] std::vector<Force> getAtomicScalarForces() const;
        [[nodiscard]] std::vector<Force> getAtomicScalarForcesOld() const;

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

        [[nodiscard]] std::vector<Pos3D>      getPositions() const;
        [[nodiscard]] std::vector<Vel3D>      getVelocities() const;
        [[nodiscard]] std::vector<Force3D>    getForces() const;
        [[nodiscard]] std::vector<AtomNumber> getAtomicNumbers() const;
        [[nodiscard]] std::set<std::string>   getUniqueQMAtomNames() const;
        [[nodiscard]]
        std::vector<double> getFlattenedQMPositionsInAngstrom() const;

        [[nodiscard]] bool wasBoxSetViaStartFile() const;

        /***************************
         * standard setter methods *
         ***************************/

        void setWaterType(MolType waterType);
        void setAmmoniaType(MolType ammoniaType);
        void setTotalMass(const Mass& totalMass);
        void setTotalCharge(const Charge& totalCharge);
        void setDensity(const Density& density);
        void setDegreesOfFreedom(size_t degreesOfFreedom);

        template <typename T>
        void setBox(const T& box, bool boxSetViaStartFile);

        /**********************************************
         * Forwards the box methods to the box object *
         **********************************************/

        void applyPBC(Pos3D& position) const;
        void scaleBox(const QuantityTensor3D& scalingTensor);

        [[nodiscard]] Volume getVolume() const;
        [[nodiscard]] Pos    getMinimalBoxDimension() const;

        [[nodiscard]] const Pos3D&   getBoxDimensions() const;
        [[nodiscard]] const Angle3D& getBoxAngles() const;
    };

}   // namespace molsys

#ifndef MOLSYS__SIMULATION_BOX_TPP
#include "simulationBox.tpp"   // IWYU pragma: export
#endif

#endif   // MOLSYS__SIMULATION_BOX_HPP
