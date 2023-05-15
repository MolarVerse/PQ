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

    int _moltype;
    int _numberOfAtoms;
    double _charge;
    double _molMass;

    std::vector<std::string> _atomNames;
    std::vector<std::string> _atomTypeNames;

    std::vector<int> _externalAtomTypes;
    std::vector<int> _atomTypes;
    std::map<int, int> _externalToInternalAtomTypes;

    std::vector<double> _partialCharges;
    std::vector<int> _globalVDWTypes;
    std::vector<double> _masses;
    std::vector<int> _atomicNumbers;

    std::vector<double> _positions;
    std::vector<double> _velocities;
    std::vector<double> _forces;

    std::vector<double> _positionsOld;
    std::vector<double> _velocitiesOld;
    std::vector<double> _forcesOld;

    // std::vector<double> centerOfMass;

    void addVector(std::vector<double> &vector, const std::vector<double> &vectorToAdd) const;

public:
    Molecule() = default;
    explicit Molecule(std::string_view name) : _name(name){};
    explicit Molecule(int moltype) : _moltype(moltype){};

    void setName(std::string_view name) { _name = name; };
    std::string getName() const { return _name; };

    void setMoltype(int moltype) { _moltype = moltype; };
    int getMoltype() const { return _moltype; };

    void setNumberOfAtoms(int numberOfAtoms);
    int getNumberOfAtoms() const { return _numberOfAtoms; };

    void setCharge(double charge) { _charge = charge; };
    double getCharge() const { return _charge; };

    void setMolMass(double molMass) { _molMass = molMass; };
    double getMolMass() const { return _molMass; };

    void addAtomName(const std::string &atomName) { _atomNames.push_back(atomName); };
    std::string getAtomName(int index) const { return _atomNames[index]; };

    void addAtomTypeName(const std::string &atomTypeName) { _atomTypeNames.push_back(atomTypeName); };
    std::string getAtomTypeName(int index) const { return _atomTypeNames[index]; };

    void addExternalAtomType(int atomType) { _externalAtomTypes.push_back(atomType); };
    int getExternalAtomType(int index) const { return _externalAtomTypes[index]; };
    std::vector<int> getExternalAtomTypes() const { return _externalAtomTypes; };
    void addExternalToInternalAtomTypeElement(int externalAtomType, int internalAtomType) { _externalToInternalAtomTypes.try_emplace(externalAtomType, internalAtomType); };

    void addAtomType(int atomType) { _atomTypes.push_back(atomType); };
    int getInternalAtomType(int externalAtomType) { return _externalToInternalAtomTypes[externalAtomType]; };
    int getAtomType(int index) const { return _atomTypes[index]; };

    void addPartialCharge(double partialCharge) { _partialCharges.push_back(partialCharge); };
    double getPartialCharge(int index) const { return _partialCharges[index]; };

    void addGlobalVDWType(int globalVDWType) { _globalVDWTypes.push_back(globalVDWType); };
    int getGlobalVDWType(int index) const { return _globalVDWTypes[index]; };

    void addMass(double mass) { _masses.push_back(mass); };
    double getMass(int index) const { return _masses[index]; };

    void addAtomicNumber(int atomicNumber) { _atomicNumbers.push_back(atomicNumber); };
    int getAtomicNumber(int index) const { return _atomicNumbers[index]; };

    void addAtomPosition(const std::vector<double> &positions) { addVector(_positions, positions); }
    std::vector<double> getAtomPosition(int index) { return {_positions[3 * index], _positions[3 * index + 1], _positions[3 * index + 2]}; }

    void addAtomVelocity(const std::vector<double> &velocities) { addVector(_velocities, velocities); }
    std::vector<double> getAtomVelocity(int index) { return {_velocities[3 * index], _velocities[3 * index + 1], _velocities[3 * index + 2]}; }

    void addAtomForce(const std::vector<double> &forces) { addVector(_forces, forces); }
    std::vector<double> getAtomForce(int index) { return {_forces[3 * index], _forces[3 * index + 1], _forces[3 * index + 2]}; }

    void addAtomPositionOld(const std::vector<double> &positionsOld) { addVector(_positionsOld, positionsOld); }
    std::vector<double> getAtomPositionOld(int index) { return {_positionsOld[3 * index], _positionsOld[3 * index + 1], _positionsOld[3 * index + 2]}; }

    void addAtomVelocityOld(const std::vector<double> &velocitiesOld) { addVector(_velocitiesOld, velocitiesOld); }
    std::vector<double> getAtomVelocityOld(int index) { return {_velocitiesOld[3 * index], _velocitiesOld[3 * index + 1], _velocitiesOld[3 * index + 2]}; }

    void addAtomForceOld(const std::vector<double> &forcesOld) { addVector(_forcesOld, forcesOld); }
    std::vector<double> getAtomForceOld(int index) { return {_forcesOld[3 * index], _forcesOld[3 * index + 1], _forcesOld[3 * index + 2]}; }

    int getNumberOfAtomTypes();
};

#endif
