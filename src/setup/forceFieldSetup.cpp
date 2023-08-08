#include "forceFieldSetup.hpp"

#include "exceptions.hpp"

#include <ranges>

using namespace std;
using namespace setup;
using namespace forceField;

/**
 * @brief setup all bonds for force field
 *
 * @details find bond type by id and set equilibrium bond length and force constant
 *
 * @note bond types are deleted afterwards from force field
 *
 */
void ForceFieldSetup::setupBonds()
{
    auto *forceField = _engine.getForceFieldPtr();

    auto addForceFieldParameters = [forceField](auto &bond)
    {
        const auto bondType = forceField->findBondTypeById(bond.getType());
        bond.setEquilibriumBondLength(bondType.getEquilibriumBondLength());
        bond.setForceConstant(bondType.getForceConstant());
    };

    ranges::for_each(forceField->getBonds(), addForceFieldParameters);

    forceField->clearBondTypes();
}

/**
 * @brief setup all angles for force field
 *
 * @details find angle type by id and set equilibrium angle and force constant
 *
 * @note angle types are deleted afterwards from force field
 *
 */
void ForceFieldSetup::setupAngles()
{
    auto *forceField = _engine.getForceFieldPtr();

    auto addForceFieldParameters = [forceField](auto &angle)
    {
        const auto angleType = forceField->findAngleTypeById(angle.getType());
        angle.setEquilibriumAngle(angleType.getEquilibriumAngle());
        angle.setForceConstant(angleType.getForceConstant());
    };

    ranges::for_each(forceField->getAngles(), addForceFieldParameters);

    forceField->clearAngleTypes();
}

/**
 * @brief setup all dihedrals for force field
 *
 * @details find dihedral type by id and set force constants
 *
 * @note dihedral types are deleted afterwards from force field
 *
 */
void ForceFieldSetup::setupDihedrals()
{
    auto *forceField = _engine.getForceFieldPtr();

    auto addForceFieldParameters = [forceField](auto &dihedral)
    {
        const auto dihedralType = forceField->findDihedralTypeById(dihedral.getType());
        dihedral.setForceConstant(dihedralType.getForceConstant());
    };

    ranges::for_each(forceField->getDihedrals(), addForceFieldParameters);

    forceField->clearDihedralTypes();
}

/**
 * @brief setup all improper dihedrals for force field
 *
 * @details find improper dihedral type by id and set force constants
 *
 * @note improper dihedral types are deleted afterwards from force field
 *
 */
void ForceFieldSetup::setupImproperDihedrals()
{
    auto *forceField = _engine.getForceFieldPtr();

    auto addForceFieldParameters = [forceField](auto &improper)
    {
        const auto improperType = forceField->findImproperDihedralTypeById(improper.getType());
        improper.setForceConstant(improperType.getForceConstant());
    };

    ranges::for_each(forceField->getImproperDihedrals(), addForceFieldParameters);

    forceField->clearImproperDihedralTypes();
}

void ForceFieldSetup::setupNonCoulombics()
{
    _engine.getSimulationBox().setupExternalToInternalGlobalVdwTypesMap();

    _engine.getForceFieldPtr()->determineInternalGlobalVdwTypes(_engine.getSimulationBox().getExternalToInternalGlobalVDWTypes());

    const auto numberOfGlobalVdwTypes           = _engine.getSimulationBox().getExternalGlobalVdwTypes().size();
    auto       selfInteractionNonCoulombicPairs = _engine.getForceFieldPtr()->getSelfInteractionNonCoulombicPairs();

    if (selfInteractionNonCoulombicPairs.size() != numberOfGlobalVdwTypes)
        throw customException::ParameterFileException(
            "Not all self interacting non coulombics were set in the noncoulombics section of the parameter file");

    ranges::sort(selfInteractionNonCoulombicPairs,
                 [](const auto &nonCoulombicPair1, const auto &nonCoulombicPair2)
                 { return nonCoulombicPair1->getInternalType1() < nonCoulombicPair2->getInternalType1(); });

    for (size_t i = 0; i < numberOfGlobalVdwTypes; ++i)
        if (selfInteractionNonCoulombicPairs[i]->getInternalType1() != i)
            throw customException::ParameterFileException(
                "Not all self interacting non coulombics were set in the noncoulombics section of the parameter file");

    _engine.getForceFieldPtr()->fillDiagonalElementsOfNonCoulombicPairsMatrix(selfInteractionNonCoulombicPairs);
    _engine.getForceFieldPtr()->fillNonDiagonalElementsOfNonCoulombicPairsMatrix();

    _engine.getForceFieldPtr()->clearNonCoulombicPairs();
}