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

#include <cmath>

#include "convergence.hpp"
#include "units.hpp"

namespace
{
    constexpr bool _enableAll = true;
    constexpr auto _relThresh = 1.0e-4;
    constexpr auto _absThresh = 1.0e-4 * kcal_per_mol;
    constexpr auto _maxThresh = 1.0e-3 * kcal_per_mol / angstrom;
    constexpr auto _rmsThresh = 1.0e-3 * kcal_per_mol / angstrom;

    opt::Convergence makeConv(ConvStrategy strat)
    {
        return {
            _enableAll,
            _enableAll,
            _enableAll,
            _relThresh,
            _absThresh,
            _maxThresh,
            _rmsThresh,
            strat
        };
    }

    opt::Convergence makeConv() { return makeConv(ConvStrategy::RIGOROUS); }
}   // namespace

/* ---------- constructor and getters ---------- */

TEST(TestConvergence, defaultConstructedFlagsAreTrue)
{
    // Default-construct via the parameterized ctor so the enum gets a
    // defined value — the no-arg ctor leaves _energyConvStrategy unset and
    // -fpermissive refuses that for a const object.
    const auto conv = makeConv();
    EXPECT_TRUE(conv.isRelEnergyConv());
    EXPECT_TRUE(conv.isAbsEnergyConv());
    EXPECT_TRUE(conv.isAbsMaxForceConv());
    EXPECT_TRUE(conv.isAbsRMSForceConv());
    EXPECT_TRUE(conv.checkConvergence());
}

TEST(TestConvergence, constructorStoresThresholdsAndStrategy)
{
    const auto conv = makeConv(ConvStrategy::LOOSE);
    EXPECT_DOUBLE_EQ(
        conv.getRelEnergyConvThreshold().numerical_value_in(one),
        _relThresh
    );
    EXPECT_DOUBLE_EQ(
        conv.getAbsEnergyConvThreshold().numerical_value_in(kcal_per_mol),
        _absThresh.numerical_value_in(kcal_per_mol)
    );
    EXPECT_DOUBLE_EQ(
        conv.getAbsMaxForceConvThreshold().numerical_value_in(
            kcal_per_mol / angstrom
        ),
        _maxThresh.numerical_value_in(kcal_per_mol / angstrom)
    );
    EXPECT_DOUBLE_EQ(
        conv.getAbsRMSForceConvThreshold().numerical_value_in(
            kcal_per_mol / angstrom
        ),
        _rmsThresh.numerical_value_in(kcal_per_mol / angstrom)
    );
    EXPECT_EQ(conv.getEnConvStrategy(), ConvStrategy::LOOSE);
    EXPECT_TRUE(conv.isEnergyConvEnabled());
    EXPECT_TRUE(conv.isMaxForceConvEnabled());
    EXPECT_TRUE(conv.isRMSForceConvEnabled());
}

/* ---------- calcEnergyConvergence ---------- */

TEST(TestConvergence, calcEnergyConvergenceFlagsBothBelowThreshold)
{
    auto conv = makeConv();
    // |1.0e-5 - 0| = 1.0e-5 < absThresh (1.0e-4)
    // rel = 1.0e-5 / 1.0 = 1.0e-5 < relThresh (1.0e-4)
    conv.calcEnergyConvergence(
        1.0 * kcal_per_mol,
        (1.0 + 1.0e-5) * kcal_per_mol
    );
    EXPECT_NEAR(
        conv.getAbsEnergy().numerical_value_in(kcal_per_mol),
        1.0e-5,
        1.0e-12
    );
    EXPECT_NEAR(conv.getRelEnergy().numerical_value_in(one), 1.0e-5, 1.0e-12);
    EXPECT_TRUE(conv.isAbsEnergyConv());
    EXPECT_TRUE(conv.isRelEnergyConv());
}

TEST(TestConvergence, calcEnergyConvergenceFlagsAboveThreshold)
{
    auto conv = makeConv();
    conv.calcEnergyConvergence(
        1.0 * kcal_per_mol,
        2.0 * kcal_per_mol
    );   // abs=1, rel=1
    EXPECT_FALSE(conv.isAbsEnergyConv());
    EXPECT_FALSE(conv.isRelEnergyConv());
}

TEST(TestConvergence, calcEnergyConvergenceSkippedWhenDisabled)
{
    opt::Convergence conv(
        false,   // energy disabled
        _enableAll,
        _enableAll,
        _relThresh,
        _absThresh,
        _maxThresh,
        _rmsThresh,
        ConvStrategy::RIGOROUS
    );
    conv.calcEnergyConvergence(
        1.0 * kcal_per_mol,
        2.0 * kcal_per_mol
    );   // would fail thresholds
    // Disabled → flags retain default (true) regardless of energy diff.
    EXPECT_TRUE(conv.isAbsEnergyConv());
    EXPECT_TRUE(conv.isRelEnergyConv());
}

/* ---------- calcForceConvergence ---------- */

TEST(TestConvergence, calcForceConvergenceFlagsBelowThreshold)
{
    auto conv = makeConv();
    conv.calcForceConvergence(
        1.0e-5 * kcal_per_mol / angstrom,
        1.0e-5 * kcal_per_mol / angstrom
    );
    EXPECT_DOUBLE_EQ(
        conv.getAbsMaxForce().numerical_value_in(kcal_per_mol / angstrom),
        1.0e-5
    );
    EXPECT_DOUBLE_EQ(
        conv.getAbsRMSForce().numerical_value_in(kcal_per_mol / angstrom),
        1.0e-5
    );
    EXPECT_TRUE(conv.isAbsMaxForceConv());
    EXPECT_TRUE(conv.isAbsRMSForceConv());
}

TEST(TestConvergence, calcForceConvergenceFlagsAboveThreshold)
{
    auto conv = makeConv();
    conv.calcForceConvergence(
        1.0 * kcal_per_mol / angstrom,
        1.0 * kcal_per_mol / angstrom
    );
    EXPECT_FALSE(conv.isAbsMaxForceConv());
    EXPECT_FALSE(conv.isAbsRMSForceConv());
}

TEST(TestConvergence, calcForceConvergenceUsesAbsoluteValue)
{
    auto conv = makeConv();
    conv.calcForceConvergence(
        -2.5 * kcal_per_mol / angstrom,
        -3.5 * kcal_per_mol / angstrom
    );
    EXPECT_DOUBLE_EQ(
        conv.getAbsMaxForce().numerical_value_in(kcal_per_mol / angstrom),
        2.5
    );
    EXPECT_DOUBLE_EQ(
        conv.getAbsRMSForce().numerical_value_in(kcal_per_mol / angstrom),
        3.5
    );
}

TEST(TestConvergence, calcForceConvergenceSkippedWhenDisabled)
{
    opt::Convergence conv(
        _enableAll,
        false,   // max force disabled
        false,   // rms force disabled
        _relThresh,
        _absThresh,
        _maxThresh,
        _rmsThresh,
        ConvStrategy::RIGOROUS
    );
    conv.calcForceConvergence(
        1.0 * kcal_per_mol / angstrom,
        1.0 * kcal_per_mol / angstrom
    );
    EXPECT_TRUE(conv.isAbsMaxForceConv());
    EXPECT_TRUE(conv.isAbsRMSForceConv());
    EXPECT_FALSE(conv.isMaxForceConvEnabled());
    EXPECT_FALSE(conv.isRMSForceConvEnabled());
}

/* ---------- checkConvergence ConvStrategy branches ---------- */

TEST(TestConvergence, checkConvergenceRigorousRequiresBothEnergyFlags)
{
    auto conv = makeConv(ConvStrategy::RIGOROUS);

    // Below thresholds → both energy flags true.
    conv.calcEnergyConvergence(
        1.0 * kcal_per_mol,
        (1.0 + 1.0e-5) * kcal_per_mol
    );
    conv.calcForceConvergence(
        1.0e-5 * kcal_per_mol / angstrom,
        1.0e-5 * kcal_per_mol / angstrom
    );
    EXPECT_TRUE(conv.checkConvergence());

    // Now bust the absolute threshold → abs false, rel still true.
    conv.calcEnergyConvergence(1.0 * kcal_per_mol, (1.0 + 1.0) * kcal_per_mol);
    EXPECT_FALSE(conv.checkConvergence());
}

TEST(TestConvergence, checkConvergenceLooseAcceptsEitherEnergyFlag)
{
    auto conv = makeConv(ConvStrategy::LOOSE);
    conv.calcForceConvergence(
        1.0e-5 * kcal_per_mol / angstrom,
        1.0e-5 * kcal_per_mol / angstrom
    );

    // abs=1.0e-5 (true), rel=1.0e-5/1.0e10=1.0e-15 (true) — both true.
    conv.calcEnergyConvergence(
        1.0e10 * kcal_per_mol,
        (1.0e10 + 1.0e-5) * kcal_per_mol
    );
    EXPECT_TRUE(conv.checkConvergence());

    // Force only the relative flag to fail (abs ok, rel huge).
    // abs=1.0e-5 (true), rel=1.0e-5/1.0e-3=1.0e-2 (false).
    conv.calcEnergyConvergence(
        1.0e-3 * kcal_per_mol,
        (1.0e-3 + 1.0e-5) * kcal_per_mol
    );
    EXPECT_TRUE(conv.isAbsEnergyConv());
    EXPECT_FALSE(conv.isRelEnergyConv());
    EXPECT_TRUE(conv.checkConvergence());   // loose: OR of energy flags
}

TEST(TestConvergence, checkConvergenceAbsoluteIgnoresRelativeFlag)
{
    auto conv = makeConv(ConvStrategy::ABSOLUTE);
    conv.calcForceConvergence(
        1.0e-5 * kcal_per_mol / angstrom,
        1.0e-5 * kcal_per_mol / angstrom
    );
    // abs=1.0e-5 (true), rel=1.0e-5/1.0e-3=1.0e-2 (false)
    conv.calcEnergyConvergence(
        1.0e-3 * kcal_per_mol,
        (1.0e-3 + 1.0e-5) * kcal_per_mol
    );
    EXPECT_FALSE(conv.isRelEnergyConv());
    EXPECT_TRUE(conv.isAbsEnergyConv());
    EXPECT_TRUE(conv.checkConvergence());
}

TEST(TestConvergence, checkConvergenceRelativeIgnoresAbsoluteFlag)
{
    auto conv = makeConv(ConvStrategy::RELATIVE);
    conv.calcForceConvergence(
        1.0e-5 * kcal_per_mol / angstrom,
        1.0e-5 * kcal_per_mol / angstrom
    );
    // abs=1.0 (false), rel=1.0/1.0e10=1.0e-10 (true)
    conv.calcEnergyConvergence(
        1.0e10 * kcal_per_mol,
        (1.0e10 + 1.0) * kcal_per_mol
    );
    EXPECT_FALSE(conv.isAbsEnergyConv());
    EXPECT_TRUE(conv.isRelEnergyConv());
    EXPECT_TRUE(conv.checkConvergence());
}

TEST(TestConvergence, checkConvergenceFailsIfMaxForceFails)
{
    auto conv = makeConv();
    conv.calcEnergyConvergence(
        1.0 * kcal_per_mol,
        (1.0 + 1.0e-5) * kcal_per_mol
    );
    conv.calcForceConvergence(
        1.0 * kcal_per_mol / angstrom,
        1.0e-5 * kcal_per_mol / angstrom
    );   // max=1.0 → above threshold
    EXPECT_FALSE(conv.checkConvergence());
}

TEST(TestConvergence, checkConvergenceFailsIfRmsForceFails)
{
    auto conv = makeConv();
    conv.calcEnergyConvergence(
        1.0 * kcal_per_mol,
        (1.0 + 1.0e-5) * kcal_per_mol
    );
    conv.calcForceConvergence(
        1.0e-5 * kcal_per_mol / angstrom,
        1.0 * kcal_per_mol / angstrom
    );   // rms=1.0 → above threshold
    EXPECT_FALSE(conv.checkConvergence());
}
