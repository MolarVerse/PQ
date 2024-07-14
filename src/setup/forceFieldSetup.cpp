/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "forceFieldSetup.hpp"

#include <algorithm>    // for __for_each_fn, for_each
#include <functional>   // for identity

#include "engine.hpp"               // for Engine
#include "forceFieldClass.hpp"      // for ForceField
#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "potential.hpp"            // for Potential

using namespace setup;
using namespace engine;
using namespace settings;

/**
 * @brief wrapper to construct ForceFieldSetup object and setup the force field
 *
 * @details the setup is only performed if the force field is activated
 *
 * @param engine
 */
void setup::setupForceField(Engine &engine)
{
    if (!ForceFieldSettings::isActive())
        return;

    engine.getStdoutOutput().writeSetup("Force Field");
    engine.getLogOutput().writeSetup("Force Field");

    ForceFieldSetup forceFieldSetup(engine);
    forceFieldSetup.setup();
}

/**
 * @brief Construct a new Force Field Setup:: Force Field Setup object
 *
 * @param engine
 */
ForceFieldSetup::ForceFieldSetup(Engine &engine) : _engine(engine){};

/**
 * @brief setup force field
 *
 * @details
 * 1) set nonCoulombPotential and coulombPotential in the ForceField class
 * 2) setup bonds
 * 3) setup angles
 * 4) setup dihedrals
 * 5) setup improper dihedrals
 *
 */
void ForceFieldSetup::setup()
{
    auto       &forceField    = _engine.getForceField();
    const auto &potential     = _engine.getPotential();
    const auto &nonCoulombPot = potential.getNonCoulombPotSharedPtr();
    const auto &coulombPot    = potential.getCoulombPotSharedPtr();

    forceField.setNonCoulombPotential(nonCoulombPot);
    forceField.setCoulombPotential(coulombPot);

    setupBonds();
    setupAngles();
    setupDihedrals();
    setupImproperDihedrals();
    setupJCouplings();

    writeSetupInfo();
}

/**
 * @brief setup all bonds for force field
 *
 * @details find bond type by id and set equilibrium bond length and force
 * constant
 *
 * @note bond types are deleted afterwards from force field
 *
 */
void ForceFieldSetup::setupBonds()
{
    auto &forceField = _engine.getForceField();

    auto addForceFieldParameters = [&forceField](auto &bond)
    {
        const auto bondType = forceField.findBondTypeById(bond.getType());
        bond.setEquilibriumBondLength(bondType.getEquilibriumBondLength());
        bond.setForceConstant(bondType.getForceConstant());
    };

    std::ranges::for_each(forceField.getBonds(), addForceFieldParameters);

    _nBondTypes = forceField.getBondTypes().size();

    forceField.clearBondTypes();
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
    auto &forceField = _engine.getForceField();

    auto addForceFieldParameters = [&forceField](auto &angle)
    {
        const auto angleType = forceField.findAngleTypeById(angle.getType());
        angle.setEquilibriumAngle(angleType.getEquilibriumAngle());
        angle.setForceConstant(angleType.getForceConstant());
    };

    std::ranges::for_each(forceField.getAngles(), addForceFieldParameters);

    _nAngleTypes = forceField.getAngleTypes().size();

    forceField.clearAngleTypes();
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
    auto &ff = _engine.getForceField();

    auto addForceFieldParameters = [&ff](auto &dihedral)
    {
        const auto dihedralType = ff.findDihedralTypeById(dihedral.getType());
        dihedral.setForceConstant(dihedralType.getForceConstant());
        dihedral.setPhaseShift(dihedralType.getPhaseShift());
        dihedral.setPeriodicity(dihedralType.getPeriodicity());
    };

    std::ranges::for_each(ff.getDihedrals(), addForceFieldParameters);

    _nDihedralTypes = ff.getDihedralTypes().size();

    ff.clearDihedralTypes();
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
    auto &ff = _engine.getForceField();

    auto addForceFieldParameters = [&ff](auto &improper)
    {
        const auto improperType = ff.findImproperTypeById(improper.getType());
        improper.setForceConstant(improperType.getForceConstant());
        improper.setPhaseShift(improperType.getPhaseShift());
        improper.setPeriodicity(improperType.getPeriodicity());
    };

    std::ranges::for_each(ff.getImproperDihedrals(), addForceFieldParameters);

    _nImproperTypes = ff.getImproperTypes().size();

    ff.clearImproperDihedralTypes();
}

/**
 * @brief setup all J couplings for force field
 *
 * @details find J coupling type by id and set force constants
 *
 * @note J coupling types are deleted afterwards from force field
 *
 */
void ForceFieldSetup::setupJCouplings()
{
    auto &ff = _engine.getForceField();

    auto addForceFieldParameters = [&ff](auto &jCoupling)
    {
        const auto type          = jCoupling.getType();
        const auto jCouplingType = ff.findJCouplingTypeById(type);

        jCoupling.setForceConstant(jCouplingType.getForceConstant());
        jCoupling.setA(jCouplingType.getA());
        jCoupling.setB(jCouplingType.getB());
        jCoupling.setC(jCouplingType.getC());
        jCoupling.setJ0(jCouplingType.getJ0());
        jCoupling.setUpperSymmetry(jCouplingType.getUpperSymmetry());
        jCoupling.setLowerSymmetry(jCouplingType.getLowerSymmetry());
    };

    std::ranges::for_each(ff.getJCouplings(), addForceFieldParameters);

    _nJCouplingTypes = ff.getJCouplTypes().size();

    ff.clearJCouplingTypes();
}

/**
 * @brief write setup information to log output
 *
 */
void ForceFieldSetup::writeSetupInfo()
{
    auto &forceField = _engine.getForceField();

    const auto nBonds      = forceField.getBonds().size();
    const auto nAngles     = forceField.getAngles().size();
    const auto nDihedrals  = forceField.getDihedrals().size();
    const auto nImpropers  = forceField.getImproperDihedrals().size();
    const auto nJCouplings = forceField.getJCouplings().size();

    const auto nBondMsg      = std::format("Bonds:       {}", nBonds);
    const auto nAngleMsg     = std::format("Angles:      {}", nAngles);
    const auto nDihedralMsg  = std::format("Dihedrals:   {}", nDihedrals);
    const auto nImproperMsg  = std::format("Impropers:   {}", nImpropers);
    const auto nJCouplingMsg = std::format("J-Couplings: {}", nJCouplings);

    // clang-format off
    const auto nBondTypeMsg      = std::format("Bond Types:       {}", _nBondTypes);
    const auto nAngleTypeMsg     = std::format("Angle Types:      {}", _nAngleTypes);
    const auto nDihedralTypeMsg  = std::format("Dihedral Types:   {}", _nDihedralTypes);
    const auto nImproperTypeMsg  = std::format("Improper Types:   {}", _nImproperTypes);
    const auto nJCouplingTypeMsg = std::format("J-Coupling Types: {}", _nJCouplingTypes);
    // clang-format on

    auto &logOutput = _engine.getLogOutput();

    logOutput.writeSetupInfo(nBondMsg);
    logOutput.writeSetupInfo(nAngleMsg);
    logOutput.writeSetupInfo(nDihedralMsg);
    logOutput.writeSetupInfo(nImproperMsg);
    logOutput.writeSetupInfo(nJCouplingMsg);
    logOutput.writeEmptyLine();

    logOutput.writeSetupInfo(nBondTypeMsg);
    logOutput.writeSetupInfo(nAngleTypeMsg);
    logOutput.writeSetupInfo(nDihedralTypeMsg);
    logOutput.writeSetupInfo(nImproperTypeMsg);
    logOutput.writeSetupInfo(nJCouplingTypeMsg);
    logOutput.writeEmptyLine();
}