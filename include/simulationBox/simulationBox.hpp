#ifndef _SIMULATION_BOX_H_

#define _SIMULATION_BOX_H_

#include <vector>
#include <string>

#include "box.hpp"
#include "molecule.hpp"
#include "defaults.hpp"

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
    int _waterType;
    int _ammoniaType;

    double _rcCutOff = _RC_CUT_OFF_DEF_;

public:
    std::vector<Molecule> _moleculeTypes;
    std::vector<Molecule> _molecules;

    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> _guffCoefficients;
    std::vector<std::vector<std::vector<std::vector<double>>>> _rncCutOffs;
    std::vector<std::vector<std::vector<std::vector<double>>>> _coulombCoefficients;

    Box _box;

    SimulationBox() = default;
    ~SimulationBox() = default;

    void setWaterType(int waterType) { _waterType = waterType; };
    int getWaterType() const { return _waterType; };

    void setAmmoniaType(int ammoniaType) { _ammoniaType = ammoniaType; };
    int getAmmoniaType() const { return _ammoniaType; };

    void setRcCutOff(double rcCutOff) { _rcCutOff = rcCutOff; };
    double getRcCutOff() const { return _rcCutOff; };

    std::vector<double> &getGuffCoefficients(int moltype1, int moltype2, int atomType1, int atomType2) { return _guffCoefficients[moltype1 - 1][moltype2 - 1][atomType1][atomType2]; };
    double getRncCutOff(int moltype1, int moltype2, int atomType1, int atomType2) { return _rncCutOffs[moltype1 - 1][moltype2 - 1][atomType1][atomType2]; };
    double getCoulombCoefficient(int moltype1, int moltype2, int atomType1, int atomType2) { return _coulombCoefficients[moltype1 - 1][moltype2 - 1][atomType1][atomType2]; };

    Molecule findMoleculeType(int moltype) const;

    int getNumberOfMolecules() const { return _molecules.size(); };
};

#endif