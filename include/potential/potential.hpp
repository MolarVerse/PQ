#ifndef _POTENTIAL_H_

#define _POTENTIAL_H_

#include <vector>
#include <memory>

#include "coulombPotential.hpp"
#include "nonCoulombPotential.hpp"
#include "simulationBox.hpp"
#include "physicalData.hpp"
#include "celllist.hpp"

/**
 * @class Potential
 *
 * @brief
 * This class is the base class for all potentials.
 * It contains a pointer to a CoulombPotential and a NonCoulombPotential.
 * Default values for the potential types are "guff".
 *
 */
class Potential
{
private:
    std::string _coulombType = "guff";
    std::string _nonCoulombType = "guff";

public:
    virtual ~Potential() = default;

    std::unique_ptr<CoulombPotential> _coulombPotential;
    std::unique_ptr<NonCoulombPotential> _nonCoulombPotential;

    virtual void calculateForces(SimulationBox &, PhysicalData &, CellList &) = 0;

    // standard getter and setters
    std::string getCoulombType() const { return _coulombType; };
    void setCoulombType(const std::string_view coulombType) { _coulombType = coulombType; };

    std::string getNonCoulombType() const { return _nonCoulombType; };
    void setNonCoulombType(const std::string_view nonCoulombType) { _nonCoulombType = nonCoulombType; };
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
class PotentialBruteForce : public Potential
{
public:
    void calculateForces(SimulationBox &, PhysicalData &, CellList &) override;
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
class PotentialCellList : public Potential
{
public:
    void calculateForces(SimulationBox &, PhysicalData &, CellList &) override;
};

#endif // _POTENTIAL_H_