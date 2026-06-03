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

#include <gtest/gtest.h>

#include <memory>

#include "atom.hpp"
#include "celllist.hpp"
#include "constraints.hpp"
#include "forceFieldClass.hpp"
#include "intraNonBonded.hpp"
#include "mmEvaluator.hpp"
#include "molecularVirial.hpp"
#include "molecule.hpp"
#include "physicalData.hpp"
#include "potentialBruteForce.hpp"
#include "simulationBox.hpp"

using namespace opt;
using simulationBox::Atom;
using simulationBox::CellList;
using simulationBox::Molecule;
using simulationBox::SimulationBox;
using physicalData::PhysicalData;

namespace
{
    // Wire an MMEvaluator with the minimum set of dependencies needed for
    // evaluate() to walk through all of its calls without throwing.
    void wireUp(MMEvaluator &eval)
    {
        auto box = std::make_shared<SimulationBox>();
        box->setBoxDimensions(linearAlgebra::Vec3D(10.0, 10.0, 10.0));

        // One molecule with two atoms so the brute-force inter-molecular loop
        // simply has no pairs to iterate.
        auto mol = Molecule();
        mol.setNumberOfAtoms(2);

        auto a1 = std::make_shared<Atom>();
        auto a2 = std::make_shared<Atom>();
        a1->setPosition(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        a2->setPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        a1->setForce(linearAlgebra::Vec3D(0.5, 0.0, 0.0));
        a2->setForce(linearAlgebra::Vec3D(0.0, 0.5, 0.0));
        a1->setMass(1.0);
        a2->setMass(1.0);

        mol.addAtom(a1);
        mol.addAtom(a2);

        box->addMolecule(mol);
        box->addAtom(a1);
        box->addAtom(a2);

        eval.setSimulationBox(box);
        eval.setCellList(std::make_shared<CellList>());
        eval.setPotential(std::make_shared<potential::PotentialBruteForce>());
        eval.setPhysicalData(std::make_shared<PhysicalData>());
        eval.setPhysicalDataOld(std::make_shared<PhysicalData>());
        eval.setForceField(std::make_shared<forceField::ForceField>());
        eval.setIntraNonBonded(std::make_shared<intraNonBonded::IntraNonBonded>());
        eval.setVirial(std::make_shared<virial::MolecularVirial>());
        eval.setConstraints(std::make_shared<constraints::Constraints>());
    }
}   // namespace

TEST(TestMMEvaluator, cloneProducesMMEvaluatorInstance)
{
    const MMEvaluator src;
    const auto        cloned = src.clone();
    ASSERT_NE(cloned, nullptr);
    EXPECT_NE(std::dynamic_pointer_cast<MMEvaluator>(cloned), nullptr);
}

TEST(TestMMEvaluator, evaluateRunsWithMinimalDependencies)
{
    MMEvaluator eval;
    wireUp(eval);
    EXPECT_NO_THROW(eval.evaluate());
}

TEST(TestMMEvaluator, evaluateZeroesForcesAtomically)
{
    // After evaluate(), the brute-force loop on a single molecule produces no
    // inter-molecular force contribution, and the cleared force buffer should
    // be (0, 0, 0) per atom.
    MMEvaluator eval;
    wireUp(eval);

    // Borrow the wired-up simulation box back from a fresh setup by re-wiring
    // a second evaluator on the same shared structures isn't trivial, so test
    // the post-condition via the evaluator's own evaluate() exit state.
    eval.evaluate();

    // We can't reach the box pointer through MMEvaluator's public API, so
    // re-wire a fresh evaluator with a known shared box and check the box.
    auto box = std::make_shared<SimulationBox>();
    box->setBoxDimensions(linearAlgebra::Vec3D(10.0, 10.0, 10.0));

    auto mol = Molecule();
    mol.setNumberOfAtoms(1);

    auto a = std::make_shared<Atom>();
    a->setPosition(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
    a->setForce(linearAlgebra::Vec3D(7.0, 7.0, 7.0));
    a->setMass(1.0);
    mol.addAtom(a);

    box->addMolecule(mol);
    box->addAtom(a);

    MMEvaluator eval2;
    eval2.setSimulationBox(box);
    eval2.setCellList(std::make_shared<CellList>());
    eval2.setPotential(std::make_shared<potential::PotentialBruteForce>());
    eval2.setPhysicalData(std::make_shared<PhysicalData>());
    eval2.setPhysicalDataOld(std::make_shared<PhysicalData>());
    eval2.setForceField(std::make_shared<forceField::ForceField>());
    eval2.setIntraNonBonded(std::make_shared<intraNonBonded::IntraNonBonded>());
    eval2.setVirial(std::make_shared<virial::MolecularVirial>());
    eval2.setConstraints(std::make_shared<constraints::Constraints>());

    eval2.evaluate();

    EXPECT_EQ(
        box->getAtoms()[0]->getForce(),
        linearAlgebra::Vec3D(0.0, 0.0, 0.0)
    );
}
