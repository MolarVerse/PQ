#ifndef _SIMULATION_BOX_H_

#define _SIMULATION_BOX_H_

#include <vector>
#include <string>

#include "box.hpp"
#include "molecule.hpp"
#include "defaults.hpp"

namespace simulationBox
{
    class SimulationBox;
}

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
class simulationBox::SimulationBox
{
private:
    int _waterType;
    int _ammoniaType;

    int _degreesOfFreedom = 0;

    double _rcCutOff = _COULOMB_CUT_OFF_DEFAULT_;

    Box _box;

    std::vector<Molecule> _molecules;

public:
    std::vector<Molecule> _moleculeTypes;

    std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>> _guffCoefficients;
    std::vector<std::vector<std::vector<std::vector<double>>>> _rncCutOffs;
    std::vector<std::vector<std::vector<std::vector<double>>>> _coulombCoefficients;
    std::vector<std::vector<std::vector<std::vector<double>>>> _cEnergyCutOffs;
    std::vector<std::vector<std::vector<std::vector<double>>>> _cForceCutOffs;
    std::vector<std::vector<std::vector<std::vector<double>>>> _ncEnergyCutOffs;
    std::vector<std::vector<std::vector<std::vector<double>>>> _ncForceCutOffs;

    SimulationBox() = default;
    ~SimulationBox() = default;

    [[nodiscard]] size_t getNumberOfParticles() const;

    [[nodiscard]] Molecule findMoleculeType(const size_t moltype) const;

    [[nodiscard]] size_t getNumberOfMolecules() const { return _molecules.size(); };

    void calculateDegreesOfFreedom();
    [[nodiscard]] int getDegreesOfFreedom() const { return _degreesOfFreedom; };

    void calculateCenterOfMassMolecules();

    void scaleBox(const Vec3D &scaleFactors) { _box.scaleBox(scaleFactors); };

    // standard getters and setters
    [[nodiscard]] double getDensity() const { return _box.getDensity(); }
    void setDensity(const double density) { _box.setDensity(density); }

    [[nodiscard]] double getTotalMass() const { return _box.getTotalMass(); }
    void setTotalMass(const double mass) { _box.setTotalMass(mass); }

    [[nodiscard]] double getTotalCharge() const { return _box.getTotalCharge(); }
    void setTotalCharge(const double charge) { _box.setTotalCharge(charge); }

    [[nodiscard]] double getVolume() const { return _box.getVolume(); }
    void setVolume(const double volume) { _box.setVolume(volume); }

    [[nodiscard]] double getMinimalBoxDimension() const { return _box.getMinimalBoxDimension(); }

    [[nodiscard]] Vec3D getBoxDimensions() const { return _box.getBoxDimensions(); }
    void setBoxDimensions(const Vec3D &boxDimensions) { _box.setBoxDimensions(boxDimensions); }

    [[nodiscard]] Vec3D getBoxAngles() const { return _box.getBoxAngles(); }
    void setBoxAngles(const Vec3D &boxAngles) { _box.setBoxAngles(boxAngles); }

    [[nodiscard]] bool getBoxSizeHasChanged() const { return _box.getBoxSizeHasChanged(); }
    void applyPBC(Vec3D &position) const { _box.applyPBC(position); };

    [[nodiscard]] double calculateVolume() { return _box.calculateVolume(); }

    [[nodiscard]] Vec3D calculateBoxDimensionsFromDensity() { return _box.calculateBoxDimensionsFromDensity(); }

    /******************/

    [[nodiscard]] std::vector<Molecule> &getMolecules() { return _molecules; };
    [[nodiscard]] Molecule &getMolecule(const size_t moleculeIndex) { return _molecules[moleculeIndex]; };
    void addMolecule(const Molecule &molecule) { _molecules.push_back(molecule); };

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
};

#endif