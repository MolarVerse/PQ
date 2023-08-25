#include "forceFieldSetup.hpp"

#include "angleForceField.hpp"   // for forceField
#include "engine.hpp"            // for Engine
#include "forceField.hpp"        // for ForceField
#include "potential.hpp"         // for Potential

#include <algorithm>    // for __for_each_fn, for_each
#include <functional>   // for identity

using namespace setup;
using namespace forceField;

void ForceFieldSetup::setup()
{
    _engine.getForceField().setNonCoulombPotential(_engine.getPotential().getNonCoulombPotentialSharedPtr());
    _engine.getForceField().setCoulombPotential(_engine.getPotential().getCoulombPotentialSharedPtr());

    setupBonds();
    setupAngles();
    setupDihedrals();
    setupImproperDihedrals();
}

void setup::setupForceField(engine::Engine &engine)
{
    if (!engine.isForceFieldActivated())
        return;

    ForceFieldSetup forceFieldSetup(engine);
    forceFieldSetup.setup();
}

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

    std::ranges::for_each(forceField->getBonds(), addForceFieldParameters);

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

    std::ranges::for_each(forceField->getAngles(), addForceFieldParameters);

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
        dihedral.setPhaseShift(dihedralType.getPhaseShift());
        dihedral.setPeriodicity(dihedralType.getPeriodicity());
    };

    std::ranges::for_each(forceField->getDihedrals(), addForceFieldParameters);

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
        improper.setPhaseShift(improperType.getPhaseShift());
        improper.setPeriodicity(improperType.getPeriodicity());
    };

    std::ranges::for_each(forceField->getImproperDihedrals(), addForceFieldParameters);

    forceField->clearImproperDihedralTypes();
}