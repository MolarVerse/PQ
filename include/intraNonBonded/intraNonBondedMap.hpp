#ifndef _INTRA_NON_BONDED_MAP_HPP_

#define _INTRA_NON_BONDED_MAP_HPP_

#include "coulombPotential.hpp"
#include "intraNonBondedContainer.hpp"
#include "molecule.hpp"
#include "nonCoulombPotential.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"

#include <cstddef>

namespace intraNonBonded
{
    class IntraNonBondedMap;
}   // namespace intraNonBonded

/**
 * @class IntraNonBondedInteraction
 *
 * @brief
 */
class intraNonBonded::IntraNonBondedMap
{
  private:
    static inline double _scale14Coulomb     = defaults::_SCALE_14_COULOMB_DEFAULT_;
    static inline double _scale14VanDerWaals = defaults::_SCALE_14_VAN_DER_WAALS_DEFAULT_;

    simulationBox::Molecule *_molecule;
    IntraNonBondedContainer *_intraNonBondedType;

  public:
    explicit IntraNonBondedMap(simulationBox::Molecule *molecule, IntraNonBondedContainer *intraNonBondedType)
        : _molecule(molecule), _intraNonBondedType(intraNonBondedType)
    {
    }

    void calculate(const potential::CoulombPotential *,
                   potential::NonCoulombPotential *,
                   const simulationBox::SimulationBox &,
                   physicalData::PhysicalData &);

    [[nodiscard]] IntraNonBondedContainer      *getIntraNonBondedType() const { return _intraNonBondedType; }
    [[nodiscard]] simulationBox::Molecule      *getMolecule() const { return _molecule; }
    [[nodiscard]] std::vector<std::vector<int>> getAtomIndices() const { return _intraNonBondedType->getAtomIndices(); }
    [[nodiscard]] static double                 getScale14Coulomb() { return _scale14Coulomb; }
    [[nodiscard]] static double                 getScale14VanDerWaals() { return _scale14VanDerWaals; }

    static void setScale14Coulomb(const double scale14Coulomb) { IntraNonBondedMap::_scale14Coulomb = scale14Coulomb; }
    static void setScale14VanDerWaals(const double scale14VanDerWaals)
    {
        IntraNonBondedMap::_scale14VanDerWaals = scale14VanDerWaals;
    }
};

#endif   // _INTRA_NON_BONDED_MAP_HPP_