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

#include <gtest/gtest.h>   // for Test, TestInfo

#include <cmath>   // for M_PI

#include "constants/conversionFactors.hpp"           // for _ANGSTROM_TO_METER_
#include "constants/internalConversionFactors.hpp"   // for _FORCE_UNIT_TO_SI_, ...
#include "constants/natureConstants.hpp"             // for _AVOGADRO_NUMBER_
#include "gtest/gtest.h"                             // for Message

/*********************
 * natural constants *
 *********************/

TEST(TestConstants, avogadroNumber)
{
    EXPECT_NEAR(constants::AVOGADRO_NUMBER / 6.02214076e23, 1.0, 1e-9);
}

TEST(TestConstants, bohrRadius)
{
    EXPECT_NEAR(constants::BOHR_RADIUS / 5.29177210903e-11, 1.0, 1e-9);
}

TEST(TestConstants, planckConstant)
{
    EXPECT_NEAR(constants::PLANCK_CONSTANT / 6.62607015e-34, 1.0, 1e-9);
}
TEST(TestConstants, reducedPlanckConstant)
{
    EXPECT_NEAR(
        constants::REDUCED_PLANCK_CONSTANT / 1.054571817e-34,
        1.0,
        1e-9
    );
}

TEST(TestConstants, boltzmannConstant)
{
    EXPECT_NEAR(constants::BOLTZMANN_CONSTANT / 1.380649e-23, 1.0, 1e-9);
}
TEST(TestConstants, universalGasConstant)
{
    EXPECT_NEAR(
        constants::UNIVERSAL_GAS_CONSTANT / 8.3144626181532395,
        1.0,
        1e-9
    );
}

TEST(TestConstants, electronCharge)
{
    EXPECT_NEAR(constants::ELECTRON_CHARGE / 1.602176634e-19, 1.0, 1e-9);
}
TEST(TestConstants, electronChargeSquared)
{
    EXPECT_NEAR(
        constants::ELECTRON_CHARGE2 /
            (constants::ELECTRON_CHARGE * constants::ELECTRON_CHARGE),
        1.0,
        1e-9
    );
}

TEST(TestConstants, electronMass)
{
    EXPECT_NEAR(constants::ELECTRON_MASS / 9.109389754e-31, 1.0, 1e-9);
}

TEST(TestConstants, permittivityVacuum)
{
    EXPECT_NEAR(constants::PERMITTIVITY_VACUUM / 8.8541878128e-12, 1.0, 1e-9);
}

TEST(TestConstants, speedOfLight)
{
    EXPECT_NEAR(constants::SPEED_OF_LIGHT / 299792458.0, 1.0, 1e-9);
}

/**********************
 * conversion factors *
 **********************/

// for degree units
TEST(TestConstants, degreesToRadians)
{
    EXPECT_NEAR(constants::DEG_TO_RAD / (M_PI / 180.0), 1.0, 1e-9);
}
TEST(TestConstants, radiansToDegrees)
{
    EXPECT_NEAR(constants::RAD_TO_DEG / (180.0 / M_PI), 1.0, 1e-9);
}

// for mass units
TEST(TestConstants, gramToKilogram)
{
    EXPECT_NEAR(constants::G_TO_KG / 1.0e-3, 1.0, 1e-9);
}
TEST(TestConstants, kilogramToGram)
{
    EXPECT_NEAR(constants::KG_TO_GRAM / 1.0e3, 1.0, 1e-9);
}
TEST(TestConstants, amuToKilogram)
{
    EXPECT_NEAR(constants::AMU_TO_KG / 1.6605402e-27, 1.0, 1e-6);
}
TEST(TestConstants, kilogramToAmu)
{
    EXPECT_NEAR(constants::KG_TO_AMU * constants::AMU_TO_KG, 1.0, 1e-9);
}

// for length units
TEST(TestConstants, angstromToMeter)
{
    EXPECT_NEAR(constants::ANGSTROM_TO_M / 1.0e-10, 1.0, 1e-9);
}
TEST(TestConstants, meterToAngstrom)
{
    EXPECT_NEAR(constants::M_TO_ANGSTROM / 1.0e10, 1.0, 1e-9);
}
TEST(TestConstants, bohrRadiusToMeter)
{
    EXPECT_NEAR(constants::BOHR_TO_M / constants::BOHR_RADIUS, 1.0, 1e-9);
}
TEST(TestConstants, meterToBohrRadius)
{
    EXPECT_NEAR(constants::M_TO_BOHR * constants::BOHR_RADIUS, 1.0, 1e-9);
}
TEST(TestConstants, angstromToBohrRadius)
{
    EXPECT_NEAR(
        constants::ANGSTROM_TO_BOHR /
            (constants::ANGSTROM_TO_M / constants::BOHR_RADIUS),
        1.0,
        1e-9
    );
}
TEST(TestConstants, bohrRadiusToAngstrom)
{
    EXPECT_NEAR(
        constants::BOHR_TO_ANGSTROM *
            (constants::ANGSTROM_TO_M / constants::BOHR_RADIUS),
        1.0,
        1e-9
    );
}

// for area units
TEST(TestConstants, angstromSquaredToMeterSquared)
{
    EXPECT_NEAR(
        constants::ANGSTROM2_TO_M2 /
            (constants::ANGSTROM_TO_M * constants::ANGSTROM_TO_M),
        1.0,
        1e-9
    );
}
TEST(TestConstants, meterSquaredToAngstromSquared)
{
    EXPECT_NEAR(
        constants::M2_TO_ANGSTROM2 *
            (constants::ANGSTROM_TO_M * constants::ANGSTROM_TO_M),
        1.0,
        1e-9
    );
}

// for volume units
TEST(TestConstants, angstromCubicToMeterCubic)
{
    EXPECT_NEAR(
        constants::ANGSTROM3_TO_M3 /
            (constants::ANGSTROM_TO_M * constants::ANGSTROM_TO_M *
             constants::ANGSTROM_TO_M),
        1.0,
        1e-9
    );
}
TEST(TestConstants, meterCubicToAngstromCubic)
{
    EXPECT_NEAR(
        constants::M3_TO_ANGSTROM3 *
            (constants::ANGSTROM_TO_M * constants::ANGSTROM_TO_M *
             constants::ANGSTROM_TO_M),
        1.0,
        1e-9
    );
}
TEST(TestConstants, angstromCubicToLiter)
{
    EXPECT_NEAR(
        constants::ANGSTROM3_TO_L /
            (constants::ANGSTROM_TO_M * constants::ANGSTROM_TO_M *
             constants::ANGSTROM_TO_M * 1.0e3),
        1.0,
        1e-9
    );
}
TEST(TestConstants, literToAngstromCubic)
{
    EXPECT_NEAR(
        constants::L_TO_ANGSTROM3 *
            (constants::ANGSTROM_TO_M * constants::ANGSTROM_TO_M *
             constants::ANGSTROM_TO_M * 1.0e3),
        1.0,
        1e-9
    );
}
TEST(TestConstants, bohrRadiusCubicToAngstromCubic)
{
    EXPECT_NEAR(
        constants::BOHR3_TO_ANGSTROM3 /
            (constants::BOHR_TO_ANGSTROM * constants::BOHR_TO_ANGSTROM *
             constants::BOHR_TO_ANGSTROM),
        1.0,
        1e-9
    );
}

// for density units
TEST(TestConstants, kgPerLiterToAmuPerAngstromCubic)
{
    EXPECT_NEAR(
        constants::KG_PER_L_TO_AMU_PER_ANGSTROM3 /
            (constants::KG_TO_AMU / constants::L_TO_ANGSTROM3),
        1.0,
        1e-9
    );
}
TEST(TestConstants, amuPerAngstromCubicToKgPerLiter)
{
    EXPECT_NEAR(
        constants::AMU_PER_ANGSTROM3_TO_KG_PER_L *
            constants::KG_PER_L_TO_AMU_PER_ANGSTROM3,
        1.0,
        1e-9
    );
}

// for energy units
TEST(TestConstants, kcalToJoule)
{
    EXPECT_NEAR(constants::KCAL_TO_J / 4184.0, 1.0, 1e-9);
}
TEST(TestConstants, jouleToKcal)
{
    EXPECT_NEAR(constants::J_TO_KCAL * 4184.0, 1.0, 1e-9);
}
TEST(TestConstants, jouleToKcalPerMol)
{
    EXPECT_NEAR(
        constants::J_TO_KCAL_PER_MOL / constants::J_TO_KCAL /
            constants::AVOGADRO_NUMBER,
        1.0,
        1e-9
    );
}
TEST(TestConstants, kcalPerMolToJoule)
{
    EXPECT_NEAR(
        constants::KCAL_PER_MOL_TO_J * constants::J_TO_KCAL *
            constants::AVOGADRO_NUMBER,
        1.0,
        1e-9
    );
}
TEST(TestConstants, hartreeToKcalPerMol)
{
    EXPECT_NEAR(
        constants::HARTREE_TO_KCAL_PER_MOL / 627.5096080305927,
        1.0,
        1e-9
    );
}
TEST(TestConstants, boltzmannConstantInKcalPerMol)
{
    EXPECT_NEAR(
        constants::BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL /
            constants::BOLTZMANN_CONSTANT / constants::J_TO_KCAL_PER_MOL,
        1.0,
        1e-9
    );
}

// for squared energy units
TEST(TestConstants, boltzmannConstantSquared)
{
    EXPECT_NEAR(
        constants::BOLTZMANN_CONSTANT2 / constants::BOLTZMANN_CONSTANT /
            constants::BOLTZMANN_CONSTANT,
        1.0,
        1e-9
    );
}
TEST(TestConstants, reducedPlanckConstantSquared)
{
    EXPECT_NEAR(
        constants::REDUCED_PLANCK_CONSTANT2 /
            constants::REDUCED_PLANCK_CONSTANT /
            constants::REDUCED_PLANCK_CONSTANT,
        1.0,
        1e-9
    );
}

// for force units
TEST(TestConstants, hartreePerBohrToKcalPerMolPerAngstrom)
{
    EXPECT_NEAR(
        constants::HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM /
            (constants::HARTREE_TO_KCAL_PER_MOL / constants::BOHR_TO_ANGSTROM),
        1.0,
        1e-9
    );
}

// for stress units
TEST(TestConstants, hartreePerBohrCubicToKcalPerMolPerAngstromCubic)
{
    EXPECT_NEAR(
        constants::HARTREE_PER_BOHR3_TO_KCAL_PER_MOL_PER_ANGSTROM3 /
            (constants::HARTREE_TO_KCAL_PER_MOL / constants::BOHR_TO_ANGSTROM /
             constants::BOHR_TO_ANGSTROM / constants::BOHR_TO_ANGSTROM),
        1.0,
        1e-9
    );
}

// for time units
TEST(TestConstants, femtosecondToSecond)
{
    EXPECT_NEAR(constants::FS_TO_S / 1.0e-15, 1.0, 1e-9);
}
TEST(TestConstants, secondToFemtosecond)
{
    EXPECT_NEAR(constants::S_TO_FS / 1.0e15, 1.0, 1e-9);
}
TEST(TestConstants, picosecondToFemtosecond)
{
    EXPECT_NEAR(constants::PS_TO_FS / 1.0e3, 1.0, 1e-9);
}
TEST(TestConstants, femtosecondToPicosecond)
{
    EXPECT_NEAR(constants::FS_TO_PS / 1.0e-3, 1.0, 1e-9);
}

// for pressure units
TEST(TestConstants, barToPascal)
{
    EXPECT_NEAR(constants::BAR_TO_P / 1.0e5, 1.0, 1e-9);
}
TEST(TestConstants, pascalToBar)
{
    EXPECT_NEAR(constants::P_TO_BAR * 1.0e5, 1.0, 1e-9);
}

// for velocity units
TEST(TestConstants, meterPerSecondToCentimeterPerPicosecond)
{
    EXPECT_NEAR(constants::M_PER_S_TO_CM_PER_S / 1.0e2, 1.0, 1e-9);
}
TEST(TestConstants, speedOfLightInCentimeterPerSecond)
{
    EXPECT_NEAR(
        constants::SPEED_OF_LIGHT_IN_CM_PER_S /
            (constants::SPEED_OF_LIGHT * constants::M_PER_S_TO_CM_PER_S),
        1.0,
        1e-9
    );
}

// for frequency units
TEST(TestConstants, perCentiMeterToHertz)
{
    EXPECT_NEAR(
        constants::PER_CM_TO_HZ / constants::SPEED_OF_LIGHT_IN_CM_PER_S,
        1.0,
        1e-9
    );
}

/*******************************
 * internal conversion factors *
 *******************************/

// for internal to SI units
TEST(TestConstants, forceUnitToSI)
{
    EXPECT_NEAR(
        constants::FORCE_UNIT_TO_SI /
            (constants::KCAL_PER_MOL_TO_J / constants::ANGSTROM_TO_M),
        1.0,
        1e-9
    );
}
TEST(TestConstants, massUnitToSI)
{
    EXPECT_NEAR(constants::MASS_UNIT_TO_SI / constants::AMU_TO_KG, 1.0, 1e-9);
}
TEST(TestConstants, timeUnitToSI)
{
    EXPECT_NEAR(constants::TIME_UNIT_TO_SI / constants::FS_TO_S, 1.0, 1e-9);
}
TEST(TestConstants, velocityUnitToSI)
{
    EXPECT_NEAR(
        constants::VELOCITY_UNIT_TO_SI / constants::ANGSTROM_TO_M,
        1.0,
        1e-9
    );
}
TEST(TestConstants, energyUnitToSI)
{
    EXPECT_NEAR(
        constants::ENERGY_UNIT_TO_SI /
            (constants::KCAL_TO_J / constants::AVOGADRO_NUMBER),
        1.0,
        1e-9
    );
}
TEST(TestConstants, volumeUnitToSI)
{
    EXPECT_NEAR(
        constants::VOLUME_UNIT_TO_SI / (constants::ANGSTROM3_TO_M3),
        1.0,
        1e-9
    );
}
TEST(TestConstants, pressureUnitToSI)
{
    EXPECT_NEAR(
        constants::PRESSURE_UNIT_TO_SI / constants::BAR_TO_P,
        1.0,
        1e-9
    );
}
TEST(TestConstants, lengthUnitToSI)
{
    EXPECT_NEAR(
        constants::LENGTH_UNIT_TO_SI / constants::ANGSTROM_TO_M,
        1.0,
        1e-9
    );
}
TEST(TestConstants, momentumUnitToSI)
{
    EXPECT_NEAR(
        constants::MOMENTUM_UNIT_TO_SI /
            (constants::G_TO_KG * constants::ANGSTROM_TO_M /
             constants::AVOGADRO_NUMBER),
        1.0,
        1e-9
    );
}

// for SI to internal units
TEST(TestConstants, siToVelocityUnit)
{
    EXPECT_NEAR(
        constants::SI_TO_VELOCITY_UNIT * constants::VELOCITY_UNIT_TO_SI,
        1.0,
        1e-9
    );
}
TEST(TestConstants, siToEnergyUnit)
{
    EXPECT_NEAR(
        constants::SI_TO_ENERGY_UNIT * constants::ENERGY_UNIT_TO_SI,
        1.0,
        1e-9
    );
}
TEST(TestConstants, siToPressureUnit)
{
    EXPECT_NEAR(
        constants::SI_TO_PRESSURE_UNIT * constants::PRESSURE_UNIT_TO_SI,
        1.0,
        1e-9
    );
}
TEST(TestConstants, siToLengthUnit)
{
    EXPECT_NEAR(
        constants::SI_TO_LENGTH_UNIT * constants::LENGTH_UNIT_TO_SI,
        1.0,
        1e-9
    );
}
TEST(TestConstants, siToForceUnit)
{
    EXPECT_NEAR(
        constants::SI_TO_FORCE_UNIT * constants::FORCE_UNIT_TO_SI,
        1.0,
        1e-9
    );
}

// for velocity verlet integrator
TEST(TestConstants, vVerletVelocityFactor)
{
    EXPECT_NEAR(
        constants::V_VERLET_VELOCITY_FACTOR /
            (0.5 * (constants::FORCE_UNIT_TO_SI / constants::MASS_UNIT_TO_SI) *
             constants::TIME_UNIT_TO_SI * constants::SI_TO_VELOCITY_UNIT),
        1.0,
        1e-9
    );
}

// for temperature calculation
TEST(TestConstants, temperatureFactor)
{
    EXPECT_NEAR(
        constants::TEMPERATURE_FACTOR /
            (constants::VELOCITY_UNIT_TO_SI * constants::VELOCITY_UNIT_TO_SI *
             constants::MASS_UNIT_TO_SI / constants::BOLTZMANN_CONSTANT),
        1.0,
        1e-9
    );
}

// for kinetic energy
TEST(TestConstants, kineticEnergyFactor)
{
    EXPECT_NEAR(
        constants::KINETIC_ENERGY_FACTOR /
            (0.5 * constants::MASS_UNIT_TO_SI * constants::VELOCITY_UNIT_TO_SI *
             constants::VELOCITY_UNIT_TO_SI * constants::SI_TO_ENERGY_UNIT),
        1.0,
        1e-9
    );
}

// for pressure calculation
TEST(TestConstants, pressureFactor)
{
    EXPECT_NEAR(
        constants::PRESSURE_FACTOR /
            (constants::ENERGY_UNIT_TO_SI / constants::VOLUME_UNIT_TO_SI *
             constants::SI_TO_PRESSURE_UNIT),
        1.0,
        1e-9
    );
}

// for coulomb prefactor
TEST(TestConstants, coulombPrefactor)
{
    EXPECT_NEAR(
        constants::COULOMB_PREFACTOR /
            (constants::ELECTRON_CHARGE * constants::ELECTRON_CHARGE *
             constants::SI_TO_LENGTH_UNIT * constants::SI_TO_ENERGY_UNIT) *
            constants::PERMITTIVITY_VACUUM * 4 * M_PI,
        1.0,
        1e-9
    );
}

// for ring polymer molecular dynamics
TEST(TestConstants, ringPolymerMolecularDynamics)
{
    EXPECT_NEAR(
        constants::RPMD_PREFACTOR /
            (constants::BOLTZMANN_CONSTANT2 /
             constants::REDUCED_PLANCK_CONSTANT2 / constants::M2_TO_ANGSTROM2 *
             constants::G_TO_KG * constants::J_TO_KCAL),
        1.0,
        1e-9
    );
}

// for momentum to force
TEST(TestConstants, momentumToForce)
{
    EXPECT_NEAR(
        constants::MOMENTUM_TO_FORCE /
            (constants::MASS_UNIT_TO_SI * constants::VELOCITY_UNIT_TO_SI *
             constants::SI_TO_FORCE_UNIT),
        1.0,
        1e-9
    );
}
