#ifndef _SIMULATION_BOX_H_

#define _SIMULATION_BOX_H_

#include "box.hpp"
#include "defaults.hpp"
#include "molecule.hpp"

#include <string>
#include <vector>

using c_ul     = const size_t;
using vector4d = std::vector<std::vector<std::vector<std::vector<double>>>>;
using vector5d = std::vector<std::vector<std::vector<std::vector<std::vector<double>>>>>;

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
    int    _waterType;
    int    _ammoniaType;
    int    _degreesOfFreedom = 0;
    double _rcCutOff         = config::_COULOMB_CUT_OFF_DEFAULT_;

    Box _box;

    std::vector<Molecule> _molecules;
    std::vector<Molecule> _moleculeTypes;

    vector5d _guffCoefficients;
    vector4d _rncCutOffs;
    vector4d _coulombCoefficients;
    vector4d _cEnergyCutOffs;
    vector4d _cForceCutOffs;
    vector4d _ncEnergyCutOffs;
    vector4d _ncForceCutOffs;

  public:
    void addMolecule(const Molecule &molecule) { _molecules.push_back(molecule); }
    void addMoleculeType(const Molecule &molecule) { _moleculeTypes.push_back(molecule); }

    size_t getNumberOfAtoms() const;
    int    getDegreesOfFreedom() const { return _degreesOfFreedom; }

    Molecule                      findMoleculeType(const size_t moltype) const;
    std::pair<Molecule *, size_t> findMoleculeByAtomIndex(const size_t atomIndex);

    void calculateDegreesOfFreedom();
    void calculateCenterOfMassMolecules();

    void resizeGuff(c_ul numberOfMoleculeTypes);
    void resizeGuff(c_ul m1, c_ul numberOfMoleulceTypes);
    void resizeGuff(c_ul m1, c_ul m2, c_ul numberOfAtoms);
    void resizeGuff(c_ul m1, c_ul m2, c_ul a1, c_ul numberOfAtoms);

    /***************************
     * standatd getter methods *
     ***************************/

    int    getWaterType() const { return _waterType; }
    int    getAmmoniaType() const { return _ammoniaType; }
    double getRcCutOff() const { return _rcCutOff; }
    size_t getNumberOfMolecules() const { return _molecules.size(); }

    std::vector<Molecule> &getMolecules() { return _molecules; }
    std::vector<Molecule> &getMoleculeTypes() { return _moleculeTypes; }
    Molecule              &getMolecule(const size_t moleculeIndex) { return _molecules[moleculeIndex]; }
    Molecule              &getMoleculeType(const size_t moleculeTypeIndex) { return _moleculeTypes[moleculeTypeIndex]; }

    double getRncCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2) { return _rncCutOffs[m1 - 1][m2 - 1][a1][a2]; }
    double getCoulombCoefficient(c_ul m1, c_ul m2, c_ul a1, c_ul a2) { return _coulombCoefficients[m1 - 1][m2 - 1][a1][a2]; }
    double getcEnergyCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2) { return _cEnergyCutOffs[m1 - 1][m2 - 1][a1][a2]; }
    double getcForceCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2) { return _cForceCutOffs[m1 - 1][m2 - 1][a1][a2]; }
    double getncEnergyCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2) { return _ncEnergyCutOffs[m1 - 1][m2 - 1][a1][a2]; }
    double getncForceCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2) { return _ncForceCutOffs[m1 - 1][m2 - 1][a1][a2]; }

    std::vector<double> &getGuffCoefficients(c_ul m1, c_ul m2, c_ul a1, c_ul a2)
    {
        return _guffCoefficients[m1 - 1][m2 - 1][a1][a2];
    }

    vector5d &getGuffCoefficients() { return _guffCoefficients; }
    vector4d &getRncCutOffs() { return _rncCutOffs; }
    vector4d &getCoulombCoefficients() { return _coulombCoefficients; }
    vector4d &getcEnergyCutOffs() { return _cEnergyCutOffs; }
    vector4d &getcForceCutOffs() { return _cForceCutOffs; }

    /***************************
     * standatd setter methods *
     ***************************/

    void setWaterType(const int waterType) { _waterType = waterType; }
    void setAmmoniaType(const int ammoniaType) { _ammoniaType = ammoniaType; }
    void setRcCutOff(const double rcCutOff) { _rcCutOff = rcCutOff; }

    void setGuffCoefficients(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const std::vector<double> &guffCoefficients)
    {
        _guffCoefficients[m1 - 1][m2 - 1][a1][a2] = guffCoefficients;
    }
    void setRncCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double rncCutOff)
    {
        _rncCutOffs[m1 - 1][m2 - 1][a1][a2] = rncCutOff;
    }
    void setCoulombCoefficient(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double coulombCoefficient)
    {
        _coulombCoefficients[m1 - 1][m2 - 1][a1][a2] = coulombCoefficient;
    }
    void setcEnergyCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double cEnergyCutOff)
    {
        _cEnergyCutOffs[m1 - 1][m2 - 1][a1][a2] = cEnergyCutOff;
    }
    void setcForceCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double cForceCutOff)
    {
        _cForceCutOffs[m1 - 1][m2 - 1][a1][a2] = cForceCutOff;
    }
    void setncEnergyCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double ncEnergyCutOff)
    {
        _ncEnergyCutOffs[m1 - 1][m2 - 1][a1][a2] = ncEnergyCutOff;
    }
    void setncForceCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double ncForceCutOff)
    {
        _ncForceCutOffs[m1 - 1][m2 - 1][a1][a2] = ncForceCutOff;
    }

    /**********************************************
     * Forwards the box methods to the box object *
     **********************************************/

    void applyPBC(vector3d::Vec3D &position) const { _box.applyPBC(position); }
    void scaleBox(const vector3d::Vec3D &scaleFactors) { _box.scaleBox(scaleFactors); }

    double          calculateVolume() { return _box.calculateVolume(); }
    vector3d::Vec3D calculateBoxDimensionsFromDensity() { return _box.calculateBoxDimensionsFromDensity(); }

    double getMinimalBoxDimension() const { return _box.getMinimalBoxDimension(); }
    bool   getBoxSizeHasChanged() const { return _box.getBoxSizeHasChanged(); }

    double          getDensity() const { return _box.getDensity(); }
    double          getTotalMass() const { return _box.getTotalMass(); }
    double          getTotalCharge() const { return _box.getTotalCharge(); }
    double          getVolume() const { return _box.getVolume(); }
    vector3d::Vec3D getBoxDimensions() const { return _box.getBoxDimensions(); }
    vector3d::Vec3D getBoxAngles() const { return _box.getBoxAngles(); }

    void setDensity(const double density) { _box.setDensity(density); }
    void setTotalMass(const double mass) { _box.setTotalMass(mass); }
    void setTotalCharge(const double charge) { _box.setTotalCharge(charge); }
    void setVolume(const double volume) { _box.setVolume(volume); }
    void setBoxDimensions(const vector3d::Vec3D &boxDimensions) { _box.setBoxDimensions(boxDimensions); }
    void setBoxAngles(const vector3d::Vec3D &boxAngles) { _box.setBoxAngles(boxAngles); }
    void setBoxSizeHasChanged(const bool boxSizeHasChanged) { _box.setBoxSizeHasChanged(boxSizeHasChanged); }
};

#endif