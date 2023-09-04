#ifndef _FORCE_FIELD_HPP_

#define _FORCE_FIELD_HPP_

#include <cstddef>   // for size_t

namespace potential
{
    class CoulombPotential;      // forward declaration
    class NonCoulombPotential;   // forward declaration
}   // namespace potential

namespace simulationBox
{
    class Molecule;   // forward declaration
}

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace forceField
{
    double correctLinker(const potential::CoulombPotential &,
                         potential::NonCoulombPotential &,
                         physicalData::PhysicalData &,
                         const simulationBox::Molecule *,
                         const simulationBox::Molecule *,
                         const size_t atomIndex1,
                         const size_t atomIndex2,
                         const double distance,
                         const bool   isDihedral);
}

#endif   // _FORCE_FIELD_HPP_