#ifndef _MOLECULE_H_

#define _MOLECULE_H_

#include <string>
#include <vector>
#include <map>

/**
 * @class Molecule
 *
 * @brief containing all information about a molecule
 */
class Molecule
{
private:
    std::string _name;

    size_t _moltype;
    size_t _numberOfAtoms;

    double _charge;
    double _molMass;

    std::vector<std::string> _atomNames;
    std::vector<std::string> _atomTypeNames;

    std::vector<size_t> _externalAtomTypes;
    std::vector<size_t> _atomTypes;
    std::map<size_t, size_t> _externalToInternalAtomTypes;

    std::vector<double> _partialCharges;
    std::vector<int> _globalVDWTypes;
    std::vector<double> _masses;
    std::vector<int> _atomicNumbers;

    std::vector<double> _positions;
    std::vector<double> _velocities;
    std::vector<double> _forces;
    std::vector<double> _shiftForces;

    std::vector<double> _centerOfMass = {0.0, 0.0, 0.0};

    void addVector(std::vector<double> &vector, const std::vector<double> &vectorToAdd) const;

public:
    Molecule() = default;
    explicit Molecule(const std::string_view name) : _name(name){};
    explicit Molecule(const size_t moltype) : _moltype(moltype){};

    void calculateCenterOfMass(const std::vector<double> &);

    void setNumberOfAtoms(const size_t numberOfAtoms);
    size_t getNumberOfAtomTypes(); // cannot be const due to iterator in it

    // standard getter and setters
    void setName(const std::string_view name) { _name = name; };
    std::string getName() const { return _name; };

    void setMoltype(const size_t moltype) { _moltype = moltype; };
    [[nodiscard]] size_t getMoltype() const { return _moltype; };

    [[nodiscard]] size_t getNumberOfAtoms() const { return _numberOfAtoms; };

    void setCharge(const double charge) { _charge = charge; };
    [[nodiscard]] double getCharge() const { return _charge; };

    void setMolMass(const double molMass) { _molMass = molMass; };
    [[nodiscard]] double getMolMass() const { return _molMass; };

    void addAtomName(const std::string &atomName) { _atomNames.push_back(atomName); };
    std::string getAtomName(const size_t index) const { return _atomNames[index]; };

    void addAtomTypeName(const std::string &atomTypeName) { _atomTypeNames.push_back(atomTypeName); };
    std::string getAtomTypeName(const size_t index) const { return _atomTypeNames[index]; };

    void addExternalAtomType(const size_t atomType) { _externalAtomTypes.push_back(atomType); };
    [[nodiscard]] size_t getExternalAtomType(const size_t index) const { return _externalAtomTypes[index]; };
    std::vector<size_t> getExternalAtomTypes() const { return _externalAtomTypes; };
    void addExternalToInternalAtomTypeElement(const size_t externalAtomType, size_t internalAtomType) { _externalToInternalAtomTypes.try_emplace(externalAtomType, internalAtomType); };

    void addAtomType(const size_t atomType) { _atomTypes.push_back(atomType); };
    size_t getInternalAtomType(const size_t externalAtomType) { return _externalToInternalAtomTypes[externalAtomType]; };
    [[nodiscard]] size_t getAtomType(const size_t index) const { return _atomTypes[index]; };

    void addPartialCharge(const double partialCharge) { _partialCharges.push_back(partialCharge); };
    [[nodiscard]] double getPartialCharge(const size_t index) const { return _partialCharges[index]; };

    void addGlobalVDWType(const int globalVDWType) { _globalVDWTypes.push_back(globalVDWType); };
    [[nodiscard]] int getGlobalVDWType(const size_t index) const { return _globalVDWTypes[index]; };

    void addMass(const double mass) { _masses.push_back(mass); };
    [[nodiscard]] double getMass(const size_t index) const { return _masses[index]; };

    void addAtomicNumber(const int atomicNumber) { _atomicNumbers.push_back(atomicNumber); };
    [[nodiscard]] int getAtomicNumber(const size_t index) const { return _atomicNumbers[index]; };

    void addAtomPositions(const std::vector<double> &positions) { addVector(_positions, positions); }
    std::vector<double> getAtomPositions(const size_t index) { return {_positions[3 * index], _positions[3 * index + 1], _positions[3 * index + 2]}; }
    void getAtomPositions(const size_t index, std::vector<double> &position) const
    {
        position[0] = _positions[3 * index];
        position[1] = _positions[3 * index + 1];
        position[2] = _positions[3 * index + 2];
    }
    void setAtomPositions(const size_t index, const std::vector<double> &position)
    {
        _positions[3 * index] = position[0];
        _positions[3 * index + 1] = position[1];
        _positions[3 * index + 2] = position[2];
    }

    void addAtomVelocities(const std::vector<double> &velocities) { addVector(_velocities, velocities); }
    std::vector<double> getAtomVelocities(const size_t index) { return {_velocities[3 * index], _velocities[3 * index + 1], _velocities[3 * index + 2]}; }
    void getAtomVelocities(const size_t index, std::vector<double> &velocity) const
    {
        velocity[0] = _velocities[3 * index];
        velocity[1] = _velocities[3 * index + 1];
        velocity[2] = _velocities[3 * index + 2];
    }
    void setAtomVelocities(const size_t index, const std::vector<double> &velocity)
    {
        _velocities[3 * index] = velocity[0];
        _velocities[3 * index + 1] = velocity[1];
        _velocities[3 * index + 2] = velocity[2];
    }

    std::vector<double> getAtomForces(const size_t index) { return {_forces[3 * index], _forces[3 * index + 1], _forces[3 * index + 2]}; }
    void getAtomForces(const size_t index, std::vector<double> &force) const
    {
        force[0] = _forces[3 * index];
        force[1] = _forces[3 * index + 1];
        force[2] = _forces[3 * index + 2];
    }
    void addAtomForces(const std::vector<double> &forces) { addVector(_forces, forces); }
    void addAtomForces(const size_t index, const std::vector<double> &force)
    {
        _forces[3 * index] += force[0];
        _forces[3 * index + 1] += force[1];
        _forces[3 * index + 2] += force[2];
    }
    void subtractAtomForces(const size_t index, const std::vector<double> &force)
    {
        _forces[3 * index] -= force[0];
        _forces[3 * index + 1] -= force[1];
        _forces[3 * index + 2] -= force[2];
    }
    void resetAtomForces() { std::fill(_forces.begin(), _forces.end(), 0.0); }

    void resizeAtomShiftForces() { _shiftForces.resize(_forces.size()); }
    void setAtomShiftForces(const size_t index, const std::vector<double> &shiftForces)
    {
        _shiftForces[3 * index] = shiftForces[0];
        _shiftForces[3 * index + 1] = shiftForces[1];
        _shiftForces[3 * index + 2] = shiftForces[2];
    }
    void getAtomShiftForces(const size_t index, std::vector<double> &shiftForce) const
    {
        shiftForce[0] = _shiftForces[3 * index];
        shiftForce[1] = _shiftForces[3 * index + 1];
        shiftForce[2] = _shiftForces[3 * index + 2];
    }
    void addAtomShifForces(const size_t index, const std::vector<double> &shiftForces)
    {
        _shiftForces[3 * index] += shiftForces[0];
        _shiftForces[3 * index + 1] += shiftForces[1];
        _shiftForces[3 * index + 2] += shiftForces[2];
    }

    void getCenterOfMass(std::vector<double> &centerOfMass) const
    {
        centerOfMass[0] = _centerOfMass[0];
        centerOfMass[1] = _centerOfMass[1];
        centerOfMass[2] = _centerOfMass[2];
    }
};

#endif
