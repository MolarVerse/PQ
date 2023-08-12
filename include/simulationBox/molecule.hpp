#ifndef _MOLECULE_HPP_

#define _MOLECULE_HPP_

#include "vector3d.hpp"

#include <map>
#include <string>
#include <vector>

namespace simulationBox
{
    class Molecule;
}

/**
 * @class Molecule
 *
 * @brief containing all information about a molecule
 */
class simulationBox::Molecule
{
  private:
    std::string _name;
    size_t      _moltype;
    size_t      _numberOfAtoms;

    double _charge;
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
    explicit Molecule(const size_t moltype) : _moltype(moltype){};

    void calculateCenterOfMass(const linearAlgebra::Vec3D &);
    void scale(const linearAlgebra::Vec3D &);
    void scaleVelocities(const double scaleFactor);
    void correctVelocities(const linearAlgebra::Vec3D &correction);

    void                 resizeAtomShiftForces() { _shiftForces.resize(_forces.size()); }
    [[nodiscard]] size_t getNumberOfAtomTypes();

    /************************
     * standard add methods *
     ************************/

    void addAtomName(const std::string &atomName) { _atomNames.push_back(atomName); }
    void addAtomTypeName(const std::string &atomTypeName) { _atomTypeNames.push_back(atomTypeName); }
    void addAtomType(const size_t atomType) { _atomTypes.push_back(atomType); }
    void addExternalAtomType(const size_t atomType) { _externalAtomTypes.push_back(atomType); }
    void addExternalToInternalAtomTypeElement(const size_t externalAtomType, size_t internalAtomType)
    {
        _externalToInternalAtomTypes.try_emplace(externalAtomType, internalAtomType);
    }

    void addPartialCharge(const double partialCharge) { _partialCharges.push_back(partialCharge); }
    void addExternalGlobalVDWType(const size_t globalVDWType) { _externalGlobalVDWTypes.push_back(globalVDWType); }
    void addAtomMass(const double mass) { _masses.push_back(mass); }
    void addAtomicNumber(const int atomicNumber) { _atomicNumbers.push_back(atomicNumber); }

    void addAtomPosition(const linearAlgebra::Vec3D &position) { _positions.push_back(position); }
    void addAtomPosition(const size_t index, const linearAlgebra::Vec3D &position) { _positions[index] += position; }
    void addAtomVelocity(const linearAlgebra::Vec3D &velocity) { _velocities.push_back(velocity); }
    void addAtomVelocity(const size_t index, const linearAlgebra::Vec3D &velocity) { _velocities[index] += velocity; }
    void addAtomForce(const linearAlgebra::Vec3D &force) { _forces.push_back(force); }
    void addAtomForce(const size_t index, const linearAlgebra::Vec3D &force) { _forces[index] += force; }
    void addAtomShiftForce(const size_t index, const linearAlgebra::Vec3D &shiftForce) { _shiftForces[index] += shiftForce; }

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] size_t getMoltype() const { return _moltype; }
    [[nodiscard]] size_t getNumberOfAtoms() const { return _numberOfAtoms; }
    [[nodiscard]] size_t getDegreesOfFreedom() const { return 3 * getNumberOfAtoms(); }
    [[nodiscard]] size_t getAtomType(const size_t index) const { return _atomTypes[index]; }
    [[nodiscard]] size_t getInternalAtomType(const size_t externalAtomType)
    {
        return _externalToInternalAtomTypes.at(externalAtomType);
    }
    [[nodiscard]] size_t getExternalAtomType(const size_t index) const { return _externalAtomTypes[index]; }

    [[nodiscard]] size_t getExternalGlobalVDWType(const size_t index) const { return _externalGlobalVDWTypes[index]; }
    [[nodiscard]] int    getAtomicNumber(const size_t index) const { return _atomicNumbers[index]; }

    [[nodiscard]] double getCharge() const { return _charge; }
    [[nodiscard]] double getMolMass() const { return _molMass; }
    [[nodiscard]] double getPartialCharge(const size_t index) const { return _partialCharges[index]; }
    [[nodiscard]] double getAtomMass(const size_t index) const { return _masses[index]; }

    [[nodiscard]] std::string getName() const { return _name; }
    [[nodiscard]] std::string getAtomName(const size_t index) const { return _atomNames[index]; }
    [[nodiscard]] std::string getAtomTypeName(const size_t index) const { return _atomTypeNames[index]; }

    [[nodiscard]] linearAlgebra::Vec3D getAtomPosition(const size_t index) const { return _positions[index]; }
    [[nodiscard]] linearAlgebra::Vec3D getAtomVelocity(const size_t index) const { return _velocities[index]; }
    [[nodiscard]] linearAlgebra::Vec3D getAtomForce(const size_t index) const { return _forces[index]; }
    [[nodiscard]] linearAlgebra::Vec3D getAtomShiftForce(const size_t index) const { return _shiftForces[index]; }
    [[nodiscard]] linearAlgebra::Vec3D getCenterOfMass() const { return _centerOfMass; }

    [[nodiscard]] std::vector<size_t> &getExternalAtomTypes() { return _externalAtomTypes; }
    [[nodiscard]] std::vector<size_t> &getExternalGlobalVDWTypes() { return _externalGlobalVDWTypes; }

    [[nodiscard]] std::vector<linearAlgebra::Vec3D> getAtomShiftForces() const { return _shiftForces; }

    /***************************
     * standard setter methods *
     ***************************/

    void setName(const std::string_view name) { _name = name; }
    void setMoltype(const size_t moltype) { _moltype = moltype; }
    void setCharge(const double charge) { _charge = charge; }
    void setMolMass(const double molMass) { _molMass = molMass; }
    void setNumberOfAtoms(const size_t numberOfAtoms) { _numberOfAtoms = numberOfAtoms; }

    void setAtomPosition(const size_t index, const linearAlgebra::Vec3D &position) { _positions[index] = position; }
    void setAtomVelocity(const size_t index, const linearAlgebra::Vec3D &velocity) { _velocities[index] = velocity; }
    void setAtomForce(const size_t index, const linearAlgebra::Vec3D &force) { _forces[index] = force; }
    void setAtomShiftForces(const size_t index, const linearAlgebra::Vec3D &shiftForce) { _shiftForces[index] = shiftForce; }
    void setAtomForcesToZero() { std::ranges::fill(_forces, linearAlgebra::Vec3D(0.0, 0.0, 0.0)); }
    void setCenterOfMass(const linearAlgebra::Vec3D &centerOfMass) { _centerOfMass = centerOfMass; }

    void setExternalAtomTypes(const std::vector<size_t> &externalAtomTypes) { _externalAtomTypes = externalAtomTypes; }
};

#endif   // _MOLECULE_HPP_
