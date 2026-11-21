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

#include <cstddef>
#include <unordered_set>

#include "exceptions.hpp"
#include "molecule.hpp"
#include "moleculeType.hpp"
#include "qmmmMDEngine.hpp"
#include "simulationBox.hpp"
#include "throwWithMessage.hpp"
#include "units.hpp"

class QMMMEnergiesTest : public ::testing::Test
{
   protected:
    static void _scaleAndAccumulate(engine::QMMMMDEngine &engine, double smF)
    {
        engine._scaleAndAccumulateEnergies(smF);
    }

    static void _moveEnergies(engine::QMMMMDEngine &engine)
    {
        engine._moveEnergiesToPhysicalData();
    }

    static void _calculateReference(engine::QMMMMDEngine &engine)
    {
        engine._calculateQMReferenceEnergy();
    }

    static std::unordered_set<size_t> _inactiveSet(
        size_t bits,
        size_t nMolecules
    )
    {
        return engine::QMMMMDEngine::_generateInactiveSmoothingMoleculeSet(
            bits,
            nMolecules
        );
    }

    static size_t _numberOfConfigurations(size_t nMolecules)
    {
        return engine::QMMMMDEngine::_numberOfSmoothingConfigurations(
            nMolecules
        );
    }
};

TEST_F(QMMMEnergiesTest, exactSmoothingCountsIntraEnergyOnce)
{
    Settings             settings;
    engine::QMMMMDEngine engine{settings};
    auto                &data = engine.get_physicalData();

    data.resetEnergies();
    data.addCoulombEnergy(10.0 * kcal_per_mol);
    data.addIntraCoulombEnergy(1.0 * kcal_per_mol);
    data.addNonCoulombEnergy(20.0 * kcal_per_mol);
    data.addIntraNonCoulombEnergy(2.0 * kcal_per_mol);
    data.setQMReferenceEnergy(-4.0 * kcal_per_mol);

    ASSERT_DOUBLE_EQ(
        data.getCoulombEnergy().numerical_value_in(kcal_per_mol),
        11.0
    );
    ASSERT_DOUBLE_EQ(
        data.getNonCoulombEnergy().numerical_value_in(kcal_per_mol),
        22.0
    );

    _scaleAndAccumulate(engine, 0.5);
    _scaleAndAccumulate(engine, 0.5);
    _moveEnergies(engine);

    EXPECT_DOUBLE_EQ(
        data.getCoulombEnergy().numerical_value_in(kcal_per_mol),
        11.0
    );
    EXPECT_DOUBLE_EQ(
        data.getNonCoulombEnergy().numerical_value_in(kcal_per_mol),
        22.0
    );
    EXPECT_DOUBLE_EQ(
        data.getIntraCoulombEnergy().numerical_value_in(kcal_per_mol),
        1.0
    );
    EXPECT_DOUBLE_EQ(
        data.getIntraNonCoulombEnergy().numerical_value_in(kcal_per_mol),
        2.0
    );
    EXPECT_DOUBLE_EQ(
        data.getQMReferenceEnergy().numerical_value_in(kcal_per_mol),
        -4.0
    );
}

TEST_F(QMMMEnergiesTest, exactSmoothingHandlesMoleculeIndicesAbove31)
{
    EXPECT_EQ(
        _inactiveSet(size_t{1} << 40U, 41),
        std::unordered_set<size_t>{40}
    );
    EXPECT_EQ(_inactiveSet(0b101, 3), (std::unordered_set<size_t>{0, 2}));

    EXPECT_EQ(_numberOfConfigurations(0), size_t{1});
    EXPECT_EQ(_numberOfConfigurations(40), size_t{1} << 40U);
    EXPECT_THROW_MSG(
        static_cast<void>(_numberOfConfigurations(64)),
        exc::HybridMDEngineException,
        "Exact smoothing supports at most 63 smoothing molecules, found 64"
    );
}

TEST_F(QMMMEnergiesTest, sumsReferenceEnergiesOfActiveMolecules)
{
    Settings             settings;
    engine::QMMMMDEngine engine{settings};
    auto                &simBox = engine.getSimulationBox();

    auto water = molsys::MoleculeType{MolType{1}};
    water.setQMReferenceEnergy(-10.0 * kcal_per_mol);
    auto ion = molsys::MoleculeType{MolType{2}};
    ion.setQMReferenceEnergy(3.0 * kcal_per_mol);
    simBox.addMoleculeType(water);
    simBox.addMoleculeType(ion);

    auto inactiveWater = molsys::Molecule{MolType{1}};
    inactiveWater.deactivateMolecule();

    simBox.addMolecule(molsys::Molecule{MolType{1}});
    simBox.addMolecule(molsys::Molecule{MolType{1}});
    simBox.addMolecule(inactiveWater);
    simBox.addMolecule(molsys::Molecule{MolType{2}});
    simBox.addMolecule(molsys::Molecule{MolType{0}});

    auto &data = engine.get_physicalData();
    data.setQMEnergy(100.0 * kcal_per_mol);

    _calculateReference(engine);

    // 2 * -10 + 3; inactive and moltype-0 molecules add nothing
    EXPECT_DOUBLE_EQ(
        data.getQMEnergy().numerical_value_in(kcal_per_mol),
        100.0
    );
    EXPECT_DOUBLE_EQ(
        data.getQMReferenceEnergy().numerical_value_in(kcal_per_mol),
        -17.0
    );
    EXPECT_DOUBLE_EQ(
        data.getTotalEnergy().numerical_value_in(kcal_per_mol),
        117.0
    );
}
