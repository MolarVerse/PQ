#ifndef _SIMULATION_BOX_HPP_

#define _SIMULATION_BOX_HPP_

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
    size_t _degreesOfFreedom    = 0;
    double _coulombRadiusCutOff = defaults::_COULOMB_CUT_OFF_DEFAULT_;

    Box _box;

    std::vector<Molecule> _molecules;
    std::vector<Molecule> _moleculeTypes;

    std::vector<size_t>      _externalGlobalVdwTypes;
    std::map<size_t, size_t> _externalToInternalGlobalVDWTypes;

    vector5d _guffCoefficients;
    vector4d _nonCoulombRadiusCutOffs;
    vector4d _coulombCoefficients;
    vector4d _coulombEnergyCutOffs;
    vector4d _coulombForceCutOffs;
    vector4d _nonCoulombEnergyCutOffs;
    vector4d _nonCoulombForceCutOffs;

  public:
    void addMolecule(const Molecule &molecule) { _molecules.push_back(molecule); }
    void addMoleculeType(const Molecule &molecule) { _moleculeTypes.push_back(molecule); }

    size_t               getNumberOfAtoms() const;
    [[nodiscard]] size_t getDegreesOfFreedom() const { return _degreesOfFreedom; }

    Molecule                      findMoleculeType(const size_t moltype) const;
    std::pair<Molecule *, size_t> findMoleculeByAtomIndex(const size_t atomIndex);
    void                          setupExternalToInternalGlobalVdwTypesMap();

    void calculateDegreesOfFreedom();
    void calculateCenterOfMassMolecules();

    void resizeGuff(c_ul numberOfMoleculeTypes);
    void resizeGuff(c_ul m1, c_ul numberOfMoleculeTypes);
    void resizeGuff(c_ul m1, c_ul m2, c_ul numberOfAtoms);
    void resizeGuff(c_ul m1, c_ul m2, c_ul a1, c_ul numberOfAtoms);

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] int    getWaterType() const { return _waterType; }
    [[nodiscard]] int    getAmmoniaType() const { return _ammoniaType; }
    [[nodiscard]] double getCoulombRadiusCutOff() const { return _coulombRadiusCutOff; }
    [[nodiscard]] size_t getNumberOfMolecules() const { return _molecules.size(); }

    [[nodiscard]] std::vector<Molecule> &getMolecules() { return _molecules; }
    [[nodiscard]] std::vector<Molecule> &getMoleculeTypes() { return _moleculeTypes; }
    [[nodiscard]] Molecule              &getMolecule(const size_t moleculeIndex) { return _molecules[moleculeIndex]; }
    [[nodiscard]] Molecule &getMoleculeType(const size_t moleculeTypeIndex) { return _moleculeTypes[moleculeTypeIndex]; }

    [[nodiscard]] std::vector<size_t>      &getExternalGlobalVdwTypes() { return _externalGlobalVdwTypes; }
    [[nodiscard]] std::map<size_t, size_t> &getExternalToInternalGlobalVDWTypes() { return _externalToInternalGlobalVDWTypes; }

    [[nodiscard]] double getNonCoulombRadiusCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2)
    {
        return _nonCoulombRadiusCutOffs[m1 - 1][m2 - 1][a1][a2];
    }
    [[nodiscard]] double getCoulombCoefficient(c_ul m1, c_ul m2, c_ul a1, c_ul a2)
    {
        return _coulombCoefficients[m1 - 1][m2 - 1][a1][a2];
    }
    [[nodiscard]] double getCoulombEnergyCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2)
    {
        return _coulombEnergyCutOffs[m1 - 1][m2 - 1][a1][a2];
    }
    [[nodiscard]] double getCoulombForceCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2)
    {
        return _coulombForceCutOffs[m1 - 1][m2 - 1][a1][a2];
    }
    [[nodiscard]] double getNonCoulombEnergyCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2)
    {
        return _nonCoulombEnergyCutOffs[m1 - 1][m2 - 1][a1][a2];
    }
    [[nodiscard]] double getNonCoulombForceCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2)
    {
        return _nonCoulombForceCutOffs[m1 - 1][m2 - 1][a1][a2];
    }

    [[nodiscard]] std::vector<double> &getGuffCoefficients(c_ul m1, c_ul m2, c_ul a1, c_ul a2)
    {
        return _guffCoefficients[m1 - 1][m2 - 1][a1][a2];
    }

    [[nodiscard]] vector5d &getGuffCoefficients() { return _guffCoefficients; }
    [[nodiscard]] vector4d &getNonCoulombRadiusCutOffs() { return _nonCoulombRadiusCutOffs; }
    [[nodiscard]] vector4d &getCoulombCoefficients() { return _coulombCoefficients; }
    [[nodiscard]] vector4d &getCoulombEnergyCutOffs() { return _coulombEnergyCutOffs; }
    [[nodiscard]] vector4d &getCoulombForceCutOffs() { return _coulombForceCutOffs; }

    /***************************
     * standard setter methods *
     ***************************/

    void setWaterType(const int waterType) { _waterType = waterType; }
    void setAmmoniaType(const int ammoniaType) { _ammoniaType = ammoniaType; }
    void setCoulombRadiusCutOff(const double rcCutOff) { _coulombRadiusCutOff = rcCutOff; }

    void setGuffCoefficients(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const std::vector<double> &guffCoefficients)
    {
        _guffCoefficients[m1 - 1][m2 - 1][a1][a2] = guffCoefficients;
    }
    void setNonCoulombRadiusCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double rncCutOff)
    {
        _nonCoulombRadiusCutOffs[m1 - 1][m2 - 1][a1][a2] = rncCutOff;
    }
    void setCoulombCoefficient(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double coulombCoefficient)
    {
        _coulombCoefficients[m1 - 1][m2 - 1][a1][a2] = coulombCoefficient;
    }
    void setCoulombEnergyCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double cEnergyCutOff)
    {
        _coulombEnergyCutOffs[m1 - 1][m2 - 1][a1][a2] = cEnergyCutOff;
    }
    void setCoulombForceCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double cForceCutOff)
    {
        _coulombForceCutOffs[m1 - 1][m2 - 1][a1][a2] = cForceCutOff;
    }
    void setNonCoulombEnergyCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double ncEnergyCutOff)
    {
        _nonCoulombEnergyCutOffs[m1 - 1][m2 - 1][a1][a2] = ncEnergyCutOff;
    }
    void setNonCoulombForceCutOff(c_ul m1, c_ul m2, c_ul a1, c_ul a2, const double ncForceCutOff)
    {
        _nonCoulombForceCutOffs[m1 - 1][m2 - 1][a1][a2] = ncForceCutOff;
    }

    /**********************************************
     * Forwards the box methods to the box object *
     **********************************************/

    void applyPBC(linearAlgebra::Vec3D &position) const { _box.applyPBC(position); }
    void scaleBox(const linearAlgebra::Vec3D &scaleFactors) { _box.scaleBox(scaleFactors); }

    double               calculateVolume() { return _box.calculateVolume(); }
    linearAlgebra::Vec3D calculateBoxDimensionsFromDensity() { return _box.calculateBoxDimensionsFromDensity(); }

    [[nodiscard]] double getMinimalBoxDimension() const { return _box.getMinimalBoxDimension(); }
    [[nodiscard]] bool   getBoxSizeHasChanged() const { return _box.getBoxSizeHasChanged(); }

    [[nodiscard]] double               getDensity() const { return _box.getDensity(); }
    [[nodiscard]] double               getTotalMass() const { return _box.getTotalMass(); }
    [[nodiscard]] double               getTotalCharge() const { return _box.getTotalCharge(); }
    [[nodiscard]] double               getVolume() const { return _box.getVolume(); }
    [[nodiscard]] linearAlgebra::Vec3D getBoxDimensions() const { return _box.getBoxDimensions(); }
    [[nodiscard]] linearAlgebra::Vec3D getBoxAngles() const { return _box.getBoxAngles(); }

    void setDensity(const double density) { _box.setDensity(density); }
    void setTotalMass(const double mass) { _box.setTotalMass(mass); }
    void setTotalCharge(const double charge) { _box.setTotalCharge(charge); }
    void setVolume(const double volume) { _box.setVolume(volume); }
    void setBoxDimensions(const linearAlgebra::Vec3D &boxDimensions) { _box.setBoxDimensions(boxDimensions); }
    void setBoxAngles(const linearAlgebra::Vec3D &boxAngles) { _box.setBoxAngles(boxAngles); }
    void setBoxSizeHasChanged(const bool boxSizeHasChanged) { _box.setBoxSizeHasChanged(boxSizeHasChanged); }
};

#endif   // _SIMULATION_BOX_HPP_