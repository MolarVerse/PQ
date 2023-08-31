#ifndef _SIMULATION_BOX_HPP_

#define _SIMULATION_BOX_HPP_

#include "box.hpp"
#include "defaults.hpp"
#include "exceptions.hpp"
#include "molecule.hpp"

#include <map>
#include <optional>
#include <string>
#include <vector>

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
    using c_ul     = const size_t;
    using vector4d = std::vector<std::vector<std::vector<std::vector<double>>>>;
    using vector5d = std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>;

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
        int    _waterType;
        int    _ammoniaType;
        size_t _degreesOfFreedom    = 0;
        double _coulombRadiusCutOff = defaults::_COULOMB_CUT_OFF_DEFAULT_;

        Box _box;

        std::vector<Molecule> _molecules;
        std::vector<Molecule> _moleculeTypes;

        std::vector<size_t>      _externalGlobalVdwTypes;
        std::map<size_t, size_t> _externalToInternalGlobalVDWTypes;

      public:
        void checkCoulombRadiusCutOff(customException::ExceptionType) const;
        void setupExternalToInternalGlobalVdwTypesMap();

        void calculateDegreesOfFreedom();
        void calculateCenterOfMassMolecules();

        [[nodiscard]] bool   moleculeTypeExists(const size_t) const;
        [[nodiscard]] size_t getNumberOfAtoms() const;

        [[nodiscard]] std::optional<Molecule>       findMolecule(const size_t moleculeType);
        [[nodiscard]] Molecule                     &findMoleculeType(const size_t moleculeType);
        [[nodiscard]] std::optional<size_t>         findMoleculeTypeByString(const std::string &moleculeType) const;
        [[nodiscard]] std::pair<Molecule *, size_t> findMoleculeByAtomIndex(const size_t atomIndex);
        [[nodiscard]] std::vector<Molecule>         findNecessaryMoleculeTypes();

        void setPartialChargesOfMoleculesFromMoleculeTypes();
        void resizeInternalGlobalVDWTypes();

        /************************
         * standard add methods *
         ************************/

        void addMolecule(const Molecule &molecule) { _molecules.push_back(molecule); }
        void addMoleculeType(const Molecule &molecule) { _moleculeTypes.push_back(molecule); }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] int    getWaterType() const { return _waterType; }
        [[nodiscard]] int    getAmmoniaType() const { return _ammoniaType; }
        [[nodiscard]] double getCoulombRadiusCutOff() const { return _coulombRadiusCutOff; }
        [[nodiscard]] size_t getNumberOfMolecules() const { return _molecules.size(); }
        [[nodiscard]] size_t getDegreesOfFreedom() const { return _degreesOfFreedom; }

        [[nodiscard]] std::vector<Molecule> &getMolecules() { return _molecules; }
        [[nodiscard]] std::vector<Molecule> &getMoleculeTypes() { return _moleculeTypes; }
        [[nodiscard]] Molecule              &getMolecule(const size_t moleculeIndex) { return _molecules[moleculeIndex]; }
        [[nodiscard]] Molecule &getMoleculeType(const size_t moleculeTypeIndex) { return _moleculeTypes[moleculeTypeIndex]; }

        [[nodiscard]] std::vector<size_t>      &getExternalGlobalVdwTypes() { return _externalGlobalVdwTypes; }
        [[nodiscard]] std::map<size_t, size_t> &getExternalToInternalGlobalVDWTypes()
        {
            return _externalToInternalGlobalVDWTypes;
        }

        /***************************
         * standard setter methods *
         ***************************/

        void setWaterType(const int waterType) { _waterType = waterType; }
        void setAmmoniaType(const int ammoniaType) { _ammoniaType = ammoniaType; }
        void setCoulombRadiusCutOff(const double rcCutOff) { _coulombRadiusCutOff = rcCutOff; }

        /**********************************************
         * Forwards the box methods to the box object *
         **********************************************/

        void applyPBC(linearAlgebra::Vec3D &position) const { _box.applyPBC(position); }
        void scaleBox(const linearAlgebra::Vec3D &scaleFactors) { _box.scaleBox(scaleFactors); }

        [[nodiscard]] double               calculateVolume() { return _box.calculateVolume(); }
        [[nodiscard]] linearAlgebra::Vec3D calculateBoxDimensionsFromDensity()
        {
            return _box.calculateBoxDimensionsFromDensity();
        }

        [[nodiscard]] double getMinimalBoxDimension() const { return _box.getMinimalBoxDimension(); }
        [[nodiscard]] bool   getBoxSizeHasChanged() const { return _box.getBoxSizeHasChanged(); }

        [[nodiscard]] double               getDensity() const { return _box.getDensity(); }
        [[nodiscard]] double               getTotalMass() const { return _box.getTotalMass(); }
        [[nodiscard]] double               getTotalCharge() const { return _box.getTotalCharge(); }
        [[nodiscard]] double               getVolume() const { return _box.getVolume(); }
        [[nodiscard]] linearAlgebra::Vec3D getBoxDimensions() const { return _box.getBoxDimensions(); }
        [[nodiscard]] linearAlgebra::Vec3D getBoxAngles() const { return _box.getBoxAngles(); }

        void setDensity(const double density) { _box.setDensity(density); }
        void setTotalMass(const double mass) { _box.setTotalMass(mass); }
        void setTotalCharge(const double charge) { _box.setTotalCharge(charge); }
        void setVolume(const double volume) { _box.setVolume(volume); }
        void setBoxDimensions(const linearAlgebra::Vec3D &boxDimensions) { _box.setBoxDimensions(boxDimensions); }
        void setBoxAngles(const linearAlgebra::Vec3D &boxAngles) { _box.setBoxAngles(boxAngles); }
        void setBoxSizeHasChanged(const bool boxSizeHasChanged) { _box.setBoxSizeHasChanged(boxSizeHasChanged); }
    };

}   // namespace simulationBox

#endif   // _SIMULATION_BOX_HPP_