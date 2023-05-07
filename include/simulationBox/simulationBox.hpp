#ifndef _SIMULATION_BOX_H_

#define _SIMULATION_BOX_H_

#include <vector>
#include <string>

#include "box.hpp"
#include "molecule.hpp"

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

public:
    std::vector<Molecule> _moleculeTypes;
    std::vector<Molecule> _molecules;

    Box _box;

    SimulationBox() = default;
    ~SimulationBox() = default;

    void setWaterType(int waterType) { _waterType = waterType; };
    int getWaterType() const { return _waterType; };

    void setAmmoniaType(int ammoniaType) { _ammoniaType = ammoniaType; };
    int getAmmoniaType() const { return _ammoniaType; };

    Molecule findMoleculeType(int moltype) const;
};

#endif