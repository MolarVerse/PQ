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

    std::vector<std::string> _atomtypeNames;
    std::vector<int> _moltype;
    std::vector<double> _positions;
    std::vector<double> _velocities;
    std::vector<double> _forces;

    std::vector<double> _positionsOld;
    std::vector<double> _velocitiesOld;
    std::vector<double> _forcesOld;

    Box _box;

    SimulationBox() = default;
    ~SimulationBox() = default;

    void setWaterType(int waterType) { _waterType = waterType; };
    int getWaterType() const { return _waterType; };

    void setAmmoniaType(int ammoniaType) { _ammoniaType = ammoniaType; };
    int getAmmoniaType() const { return _ammoniaType; };

    void addAtomicProperties(std::vector<double> &, std::vector<double>) const;

    template <typename T>
    void addAtomicProperties(std::vector<T> &, T) const;
};

/**
 * @brief sets the atomic 1d properties
 *
 * @tparam T
 * @param target
 * @param toAdd
 */
// FIXME: move this function into a separate header file
template <typename T>
void SimulationBox::addAtomicProperties(std::vector<T> &target, T toAdd) const
{
    target.push_back(toAdd);
}

#endif