#ifndef _MOLECULE_HPP_

#define _MOLECULE_HPP_

#include "vector3d.hpp"

#include <algorithm>
#include <cstddef>   // for size_t
#include <map>
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
        size_t      _numberOfAtoms;

        double _charge;   // set via molDescriptor not sum of partial charges!!!
        double _molMass;

        std::vector<std::string> _atomNames;
        std::vector<std::string> _atomTypeNames;

        std::vector<size_t> _externalGlobalVDWTypes;
        std::vector<size_t> _internalGlobalVDWTypes;

        std::vector<int>         _atomicNumbers;
        std::vector<size_t>      _externalAtomTypes;
        std::vector<size_t>      _atomTypes;
        std::map<size_t, size_t> _externalToInternalAtomTypes;

        std::vector<double> _partialCharges;
        std::vector<double> _masses;

        std::vector<linearAlgebra::Vec3D> _positions;
        std::vector<linearAlgebra::Vec3D> _velocities;
        std::vector<linearAlgebra::Vec3D> _forces;
        std::vector<linearAlgebra::Vec3D> _shiftForces;

        linearAlgebra::Vec3D _centerOfMass = linearAlgebra::Vec3D(0.0, 0.0, 0.0);

      public:
        Molecule() = default;
        explicit Molecule(const std::string_view name) : _name(name){};
        explicit Molecule(c_ul moltype) : _moltype(moltype){};

        void calculateCenterOfMass(const linearAlgebra::Vec3D &);
        void scale(const linearAlgebra::Vec3D &);
        void scaleVelocities(const double scaleFactor);
        void correctVelocities(const linearAlgebra::Vec3D &correction);

        [[nodiscard]] size_t getNumberOfAtomTypes();

        /***************************
         * standard resize methods *
         ***************************/

        void resizeAtomShiftForces() { _shiftForces.resize(_numberOfAtoms); }
        void resizeInternalGlobalVDWTypes() { _internalGlobalVDWTypes.resize(_numberOfAtoms); }

        /************************
         * standard add methods *
         ************************/

        void addAtomName(const std::string &atomName) { _atomNames.push_back(atomName); }
        void addAtomTypeName(const std::string &atomTypeName) { _atomTypeNames.push_back(atomTypeName); }
        void addAtomType(c_ul atomType) { _atomTypes.push_back(atomType); }
        void addExternalAtomType(c_ul atomType) { _externalAtomTypes.push_back(atomType); }
        void addExternalToInternalAtomTypeElement(c_ul value, c_ul key) { _externalToInternalAtomTypes.try_emplace(value, key); }

        void addPartialCharge(const double partialCharge) { _partialCharges.push_back(partialCharge); }
        void addExternalGlobalVDWType(c_ul globalVDWType) { _externalGlobalVDWTypes.push_back(globalVDWType); }
        void addInternalGlobalVDWType(c_ul globalVDWType) { _internalGlobalVDWTypes.push_back(globalVDWType); }
        void addAtomMass(const double mass) { _masses.push_back(mass); }
        void addAtomicNumber(const int atomicNumber) { _atomicNumbers.push_back(atomicNumber); }

        void addAtomPosition(const linearAlgebra::Vec3D &position) { _positions.push_back(position); }
        void addAtomPosition(c_ul index, const linearAlgebra::Vec3D &position) { _positions[index] += position; }

        void addAtomVelocity(const linearAlgebra::Vec3D &velocity) { _velocities.push_back(velocity); }
        void addAtomVelocity(c_ul index, const linearAlgebra::Vec3D &velocity) { _velocities[index] += velocity; }

        void addAtomForce(const linearAlgebra::Vec3D &force) { _forces.push_back(force); }
        void addAtomForce(c_ul index, const linearAlgebra::Vec3D &force) { _forces[index] += force; }

        void addAtomShiftForce(c_ul index, const linearAlgebra::Vec3D &shiftForce) { _shiftForces[index] += shiftForce; }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getMoltype() const { return _moltype; }
        [[nodiscard]] size_t getNumberOfAtoms() const { return _numberOfAtoms; }
        [[nodiscard]] size_t getDegreesOfFreedom() const { return 3 * getNumberOfAtoms(); }

        [[nodiscard]] size_t getAtomType(c_ul index) const { return _atomTypes[index]; }
        [[nodiscard]] size_t getInternalAtomType(c_ul type) const { return _externalToInternalAtomTypes.at(type); }
        [[nodiscard]] size_t getExternalAtomType(c_ul index) const { return _externalAtomTypes[index]; }
        [[nodiscard]] size_t getExternalGlobalVDWType(c_ul index) const { return _externalGlobalVDWTypes[index]; }
        [[nodiscard]] size_t getInternalGlobalVDWType(c_ul index) const { return _internalGlobalVDWTypes[index]; }

        [[nodiscard]] int getAtomicNumber(c_ul index) const { return _atomicNumbers[index]; }

        [[nodiscard]] double getCharge() const { return _charge; }
        [[nodiscard]] double getPartialCharge(c_ul index) const { return _partialCharges[index]; }
        [[nodiscard]] double getMolMass() const { return _molMass; }
        [[nodiscard]] double getAtomMass(c_ul index) const { return _masses[index]; }

        [[nodiscard]] std::vector<double> getPartialCharges() const { return _partialCharges; }
        [[nodiscard]] std::vector<double> getAtomMasses() const { return _masses; }

        [[nodiscard]] std::string getName() const { return _name; }
        [[nodiscard]] std::string getAtomName(c_ul index) const { return _atomNames[index]; }
        [[nodiscard]] std::string getAtomTypeName(c_ul index) const { return _atomTypeNames[index]; }

        [[nodiscard]] linearAlgebra::Vec3D getAtomPosition(c_ul index) const { return _positions[index]; }
        [[nodiscard]] linearAlgebra::Vec3D getAtomVelocity(c_ul index) const { return _velocities[index]; }
        [[nodiscard]] linearAlgebra::Vec3D getAtomForce(c_ul index) const { return _forces[index]; }
        [[nodiscard]] linearAlgebra::Vec3D getAtomShiftForce(c_ul index) const { return _shiftForces[index]; }
        [[nodiscard]] linearAlgebra::Vec3D getCenterOfMass() const { return _centerOfMass; }

        [[nodiscard]] std::vector<size_t>  getExternalAtomTypes() const { return _externalAtomTypes; }
        [[nodiscard]] std::vector<size_t>  getExternalGlobalVDWTypes() const { return _externalGlobalVDWTypes; }
        [[nodiscard]] std::vector<size_t> &getExternalGlobalVDWTypes() { return _externalGlobalVDWTypes; }
        [[nodiscard]] std::vector<size_t>  getInternalGlobalVDWTypes() const { return _internalGlobalVDWTypes; }

        [[nodiscard]] std::vector<linearAlgebra::Vec3D> getAtomShiftForces() const { return _shiftForces; }

        /***************************
         * standard setter methods *
         ***************************/

        void setPartialCharge(c_ul index, const double partialCharge) { _partialCharges[index] = partialCharge; }
        void setPartialCharges(const std::vector<double> &partialCharges) { _partialCharges = partialCharges; }

        void setName(const std::string_view name) { _name = name; }
        void setMoltype(c_ul moltype) { _moltype = moltype; }
        void setCharge(const double charge) { _charge = charge; }
        void setMolMass(const double molMass) { _molMass = molMass; }
        void setNumberOfAtoms(c_ul numberOfAtoms) { _numberOfAtoms = numberOfAtoms; }
        void setAtomPosition(c_ul index, const linearAlgebra::Vec3D &position) { _positions[index] = position; }
        void setAtomVelocity(c_ul index, const linearAlgebra::Vec3D &velocity) { _velocities[index] = velocity; }
        void setAtomForce(c_ul index, const linearAlgebra::Vec3D &force) { _forces[index] = force; }
        void setAtomShiftForces(c_ul index, const linearAlgebra::Vec3D &shiftForce) { _shiftForces[index] = shiftForce; }
        void setCenterOfMass(const linearAlgebra::Vec3D &centerOfMass) { _centerOfMass = centerOfMass; }
        void setExternalAtomTypes(const std::vector<size_t> &externalAtomTypes) { _externalAtomTypes = externalAtomTypes; }

        void setAtomForcesToZero() { std::ranges::fill(_forces, linearAlgebra::Vec3D(0.0, 0.0, 0.0)); }
    };

}   // namespace simulationBox

#endif   // _MOLECULE_HPP_
