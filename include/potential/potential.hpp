#ifndef _POTENTIAL_H_

#define _POTENTIAL_H_

#include "celllist.hpp"
#include "coulombPotential.hpp"
#include "nonCoulombPotential.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"

#include <memory>
#include <vector>

namespace potential
{
    class Potential;
    class PotentialBruteForce;
    class PotentialCellList;
}   // namespace potential

/**
 * @class Potential
 *
 * @brief
 * This class is the base class for all potentials.
 * It contains a pointer to a CoulombPotential and a NonCoulombPotential.
 * Default values for the potential types are "guff".
 *
 */
class potential::Potential
{
  protected:
    std::string _coulombType    = "guff";
    std::string _nonCoulombType = "guff";

    std::unique_ptr<CoulombPotential>    _coulombPotential;
    std::unique_ptr<NonCoulombPotential> _nonCoulombPotential;

  public:
    virtual ~Potential() = default;

    virtual void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) = 0;

    void calcCoulomb(const double coulombCoefficient,
                     const double rcCutoff,
                     const double distance,
                     double      &energy,
                     double      &force,
                     const double energy_cutoff,
                     const double force_cutoff) const
    {
        _coulombPotential->calcCoulomb(coulombCoefficient, rcCutoff, distance, energy, force, energy_cutoff, force_cutoff);
    }

    void calcNonCoulomb(const std::vector<double> &guffCoefficients,
                        const double               rncCutoff,
                        const double               distance,
                        double                    &energy,
                        double                    &force,
                        const double               energy_cutoff,
                        const double               force_cutoff) const
    {
        _nonCoulombPotential->calcNonCoulomb(guffCoefficients, rncCutoff, distance, energy, force, energy_cutoff, force_cutoff);
    }

    /********************************
     * standard getters and setters *
     ********************************/
    std::string getCoulombType() const { return _coulombType; }
    std::string getNonCoulombType() const { return _nonCoulombType; }

    void                    setCoulombType(const std::string_view coulombType) { _coulombType = coulombType; }
    void                    setNonCoulombType(const std::string_view nonCoulombType) { _nonCoulombType = nonCoulombType; }
    template <class T> void setCoulombPotential(const T &potential) { _coulombPotential = std::make_unique<T>(potential); }
    template <class T> void setNonCoulombPotential(const T &potential) { _nonCoulombPotential = std::make_unique<T>(potential); }
};

/**
 * @class PotentialBruteForce inherits Potential
 *
 * @brief
 * This class is the brute force implementation of the potential.
 * It contains a pointer to a CoulombPotential and a NonCoulombPotential.
 * Default values for the potential types are "guff".
 *
 */
class potential::PotentialBruteForce : public potential::Potential
{
  public:
    void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) override;
};

/**
 * @class PotentialCellList inherits Potential
 *
 * @brief
 * This class is the cell list implementation of the potential.
 * It contains a pointer to a CoulombPotential and a NonCoulombPotential.
 * Default values for the potential types are "guff".
 *
 */
class potential::PotentialCellList : public potential::Potential
{
  public:
    void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) override;
};

#endif   // _POTENTIAL_H_