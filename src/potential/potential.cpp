#include "potential.hpp"

#include <cmath>
#include <iostream>

using namespace std;
using namespace simulationBox;
using namespace potential;
using namespace physicalData;
using namespace linearAlgebra;

/**
 * @brief inner part of the double loop to calculate non-bonded inter molecular interactions
 *
 * @param box
 * @param molecule1
 * @param molecule2
 * @param atom1
 * @param atom2
 * @return std::pair<double, double>
 */
std::pair<double, double> Potential::calculateSingleInteraction(const linearAlgebra::Vec3D &box,
                                                                simulationBox::Molecule    &molecule1,
                                                                simulationBox::Molecule    &molecule2,
                                                                const size_t                atom1,
                                                                const size_t                atom2)
{
    auto coulombEnergy    = 0.0;
    auto nonCoulombEnergy = 0.0;

    const auto xyz_i = molecule1.getAtomPosition(atom1);
    const auto xyz_j = molecule2.getAtomPosition(atom2);

    auto dxyz = xyz_i - xyz_j;

    const auto txyz = -box * round(dxyz / box);

    // dxyz += txyz;
    dxyz[0] += txyz[0];
    dxyz[1] += txyz[1];
    dxyz[2] += txyz[2];

    const double distanceSquared = normSquared(dxyz);

    if (const auto RcCutOff = CoulombPotential::getCoulombRadiusCutOff(); distanceSquared < RcCutOff * RcCutOff)
    {
        const double distance   = ::sqrt(distanceSquared);
        const size_t atomType_i = molecule1.getAtomType(atom1);
        const size_t atomType_j = molecule2.getAtomType(atom2);

        // TODO: think of a clever solution for guff routine
        //  const size_t externalGlobalVdwType_i = molecule1.getExternalGlobalVDWType(atom1);
        //  const size_t externalGlobalVdwType_j = molecule2.getExternalGlobalVDWType(atom2);

        // const size_t globalVdwType_i =
        // simBox.getExternalToInternalGlobalVDWTypes().at(externalGlobalVdwType_i); const size_t globalVdwType_j
        // = simBox.getExternalToInternalGlobalVDWTypes().at(externalGlobalVdwType_j);

        const size_t globalVdwType_i = 0;
        const size_t globalVdwType_j = 0;

        const auto moltype_i = molecule1.getMoltype();
        const auto moltype_j = molecule2.getMoltype();

        const auto combinedIndices = {moltype_i, moltype_j, atomType_i, atomType_j, globalVdwType_i, globalVdwType_j};

        const auto coulombPreFactor = 1.0;   // TODO: implement for force field

        auto [energy, force] = _coulombPotential->calculate(combinedIndices, distance, coulombPreFactor);
        coulombEnergy        = energy;

        const auto nonCoulombicPair = _nonCoulombPotential->getNonCoulombPair(combinedIndices);

        if (const auto rncCutOff = nonCoulombicPair->getRadialCutOff(); distance < rncCutOff)
        {
            const auto &[energy, nonCoulombForce] = nonCoulombicPair->calculateEnergyAndForce(distance);
            nonCoulombEnergy                      = energy;

            force += nonCoulombForce;
        }

        force /= distance;

        const auto forcexyz = force * dxyz;

        const auto shiftForcexyz = forcexyz * txyz;

        molecule1.addAtomForce(atom1, forcexyz);
        molecule2.addAtomForce(atom2, -forcexyz);

        molecule1.addAtomShiftForce(atom1, shiftForcexyz);
    }

    return {coulombEnergy, nonCoulombEnergy};
}