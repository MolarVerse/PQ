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

    int _degreesOfFreedom = 0;

    double _rcCutOff = _COULOMB_CUT_OFF_DEFAULT_;

public:
    std::vector<Molecule> _moleculeTypes;
    std::vector<Molecule> _molecules;

    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> _guffCoefficients;
    std::vector<std::vector<std::vector<std::vector<double>>>> _rncCutOffs;
    std::vector<std::vector<std::vector<std::vector<double>>>> _coulombCoefficients;
    std::vector<std::vector<std::vector<std::vector<double>>>> _cEnergyCutOffs;
    std::vector<std::vector<std::vector<std::vector<double>>>> _cForceCutOffs;
    std::vector<std::vector<std::vector<std::vector<double>>>> _ncEnergyCutOffs;
    std::vector<std::vector<std::vector<std::vector<double>>>> _ncForceCutOffs;

    Box _box;

    SimulationBox() = default;
    ~SimulationBox() = default;

    void setWaterType(const int waterType) { _waterType = waterType; };
    [[nodiscard]] int getWaterType() const { return _waterType; };

    void setAmmoniaType(const int ammoniaType) { _ammoniaType = ammoniaType; };
    [[nodiscard]] int getAmmoniaType() const { return _ammoniaType; };

    void setRcCutOff(const double rcCutOff) { _rcCutOff = rcCutOff; };
    [[nodiscard]] double getRcCutOff() const { return _rcCutOff; };

    [[nodiscard]] std::vector<double> &getGuffCoefficients(const size_t moltype1, const size_t moltype2, const size_t atomType1, const size_t atomType2) { return _guffCoefficients[moltype1 - 1][moltype2 - 1][atomType1][atomType2]; };
    void getGuffCoefficients(const size_t moltype1, const size_t moltype2, const size_t atomType1, const size_t atomType2, std::vector<double> &guffCoefficients)
    {
        for (size_t i = 0; i < 22; ++i)
        {
            guffCoefficients[i] = _guffCoefficients[moltype1 - 1][moltype2 - 1][atomType1][atomType2][i];
        }
    }
    [[nodiscard]] double getRncCutOff(const size_t moltype1, const size_t moltype2, const size_t atomType1, const size_t atomType2) { return _rncCutOffs[moltype1 - 1][moltype2 - 1][atomType1][atomType2]; };
    [[nodiscard]] double getCoulombCoefficient(const size_t moltype1, const size_t moltype2, const size_t atomType1, const size_t atomType2) { return _coulombCoefficients[moltype1 - 1][moltype2 - 1][atomType1][atomType2]; };
    [[nodiscard]] double getcEnergyCutOff(const size_t moltype1, const size_t moltype2, const size_t atomType1, const size_t atomType2) { return _cEnergyCutOffs[moltype1 - 1][moltype2 - 1][atomType1][atomType2]; };
    [[nodiscard]] double getcForceCutOff(const size_t moltype1, const size_t moltype2, const size_t atomType1, const size_t atomType2) { return _cForceCutOffs[moltype1 - 1][moltype2 - 1][atomType1][atomType2]; };
    [[nodiscard]] double getncEnergyCutOff(const size_t moltype1, const size_t moltype2, const size_t atomType1, const size_t atomType2) { return _ncEnergyCutOffs[moltype1 - 1][moltype2 - 1][atomType1][atomType2]; };
    [[nodiscard]] double getncForceCutOff(const size_t moltype1, const size_t moltype2, const size_t atomType1, const size_t atomType2) { return _ncForceCutOffs[moltype1 - 1][moltype2 - 1][atomType1][atomType2]; };

    [[nodiscard]] Molecule findMoleculeType(const size_t moltype) const;

    [[nodiscard]] size_t getNumberOfMolecules() const { return _molecules.size(); };

    void calculateDegreesOfFreedom();
    [[nodiscard]] int getDegreesOfFreedom() const { return _degreesOfFreedom; };

    void calculateCenterOfMassMolecules();
};

#endif