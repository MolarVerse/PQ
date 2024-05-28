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

#include <cmath>    // for M_PI
#include <memory>   // for allocator

#include "constants/conversionFactors.hpp"           // for _ANGSTROM_TO_METER_
#include "constants/internalConversionFactors.hpp"   // for _FORCE_UNIT_TO_SI_, ...
#include "constants/natureConstants.hpp"             // for _AVOGADRO_NUMBER_
#include "gtest/gtest.h"                             // for Message

/*********************
 * natural constants *
 *********************/

TEST(TestConstants, avogadroNumber)
{
    EXPECT_NEAR(constants::_AVOGADRO_NUMBER_ / 6.02214076e23, 1.0, 1e-9);
}

TEST(TestConstants, bohrRadius)
{
    EXPECT_NEAR(constants::_BOHR_RADIUS_ / 5.29177210903e-11, 1.0, 1e-9);
}

TEST(TestConstants, planckConstant)
{
    EXPECT_NEAR(constants::_PLANCK_CONSTANT_ / 6.62607015e-34, 1.0, 1e-9);
}
TEST(TestConstants, reducedPlanckConstant)
{
    EXPECT_NEAR(
        constants::_REDUCED_PLANCK_CONSTANT_ / 1.054571817e-34,
        1.0,
        1e-9
    );
}

TEST(TestConstants, boltzmannConstant)
{
    EXPECT_NEAR(constants::_BOLTZMANN_CONSTANT_ / 1.380649e-23, 1.0, 1e-9);
}
TEST(TestConstants, universalGasConstant)
{
    EXPECT_NEAR(
        constants::_UNIVERSAL_GAS_CONSTANT_ / 8.3144626181532395,
        1.0,
        1e-9
    );
}

TEST(TestConstants, electronCharge)
{
    EXPECT_NEAR(constants::_ELECTRON_CHARGE_ / 1.602176634e-19, 1.0, 1e-9);
}
TEST(TestConstants, electronChargeSquared)
{
    EXPECT_NEAR(
        constants::_ELECTRON_CHARGE_SQUARED_ /
            (constants::_ELECTRON_CHARGE_ * constants::_ELECTRON_CHARGE_),
        1.0,
        1e-9
    );
}

TEST(TestConstants, electronMass)
{
    EXPECT_NEAR(constants::_ELECTRON_MASS_ / 9.109389754e-31, 1.0, 1e-9);
}

TEST(TestConstants, permittivityVacuum)
{
    EXPECT_NEAR(constants::_PERMITTIVITY_VACUUM_ / 8.8541878128e-12, 1.0, 1e-9);
}

TEST(TestConstants, speedOfLight)
{
    EXPECT_NEAR(constants::_SPEED_OF_LIGHT_ / 299792458.0, 1.0, 1e-9);
}

/**********************
 * conversion factors *
 **********************/

// for degree units
TEST(TestConstants, degreesToRadians)
{
    EXPECT_NEAR(constants::_DEG_TO_RAD_ / (M_PI / 180.0), 1.0, 1e-9);
}
TEST(TestConstants, radiansToDegrees)
{
    EXPECT_NEAR(constants::_RAD_TO_DEG_ / (180.0 / M_PI), 1.0, 1e-9);
}

// for mass units
TEST(TestConstants, gramToKilogram)
{
    EXPECT_NEAR(constants::_GRAM_TO_KG_ / 1.0e-3, 1.0, 1e-9);
}
TEST(TestConstants, kilogramToGram)
{
    EXPECT_NEAR(constants::_KG_TO_GRAM_ / 1.0e3, 1.0, 1e-9);
}
TEST(TestConstants, amuToKilogram)
{
    EXPECT_NEAR(constants::_AMU_TO_KG_ / 1.6605402e-27, 1.0, 1e-6);
}
TEST(TestConstants, kilogramToAmu)
{
    EXPECT_NEAR(constants::_KG_TO_AMU_ * constants::_AMU_TO_KG_, 1.0, 1e-9);
}

// for length units
TEST(TestConstants, angstromToMeter)
{
    EXPECT_NEAR(constants::_ANGSTROM_TO_METER_ / 1.0e-10, 1.0, 1e-9);
}
TEST(TestConstants, meterToAngstrom)
{
    EXPECT_NEAR(constants::_METER_TO_ANGSTROM_ / 1.0e10, 1.0, 1e-9);
}
TEST(TestConstants, bohrRadiusToMeter)
{
    EXPECT_NEAR(
        constants::_BOHR_RADIUS_TO_METER_ / constants::_BOHR_RADIUS_,
        1.0,
        1e-9
    );
}
TEST(TestConstants, meterToBohrRadius)
{
    EXPECT_NEAR(
        constants::_METER_TO_BOHR_RADIUS_ * constants::_BOHR_RADIUS_,
        1.0,
        1e-9
    );
}
TEST(TestConstants, angstromToBohrRadius)
{
    EXPECT_NEAR(
        constants::_ANGSTROM_TO_BOHR_RADIUS_ /
            (constants::_ANGSTROM_TO_METER_ / constants::_BOHR_RADIUS_),
        1.0,
        1e-9
    );
}
TEST(TestConstants, bohrRadiusToAngstrom)
{
    EXPECT_NEAR(
        constants::_BOHR_RADIUS_TO_ANGSTROM_ *
            (constants::_ANGSTROM_TO_METER_ / constants::_BOHR_RADIUS_),
        1.0,
        1e-9
    );
}

// for area units
TEST(TestConstants, angstromSquaredToMeterSquared)
{
    EXPECT_NEAR(
        constants::_ANGSTROM_SQUARED_TO_METER_SQUARED_ /
            (constants::_ANGSTROM_TO_METER_ * constants::_ANGSTROM_TO_METER_),
        1.0,
        1e-9
    );
}
TEST(TestConstants, meterSquaredToAngstromSquared)
{
    EXPECT_NEAR(
        constants::_METER_SQUARED_TO_ANGSTROM_SQUARED_ *
            (constants::_ANGSTROM_TO_METER_ * constants::_ANGSTROM_TO_METER_),
        1.0,
        1e-9
    );
}

// for volume units
TEST(TestConstants, angstromCubicToMeterCubic)
{
    EXPECT_NEAR(
        constants::_ANGSTROM_CUBIC_TO_METER_CUBIC_ /
            (constants::_ANGSTROM_TO_METER_ * constants::_ANGSTROM_TO_METER_ *
             constants::_ANGSTROM_TO_METER_),
        1.0,
        1e-9
    );
}
TEST(TestConstants, meterCubicToAngstromCubic)
{
    EXPECT_NEAR(
        constants::_METER_CUBIC_TO_ANGSTROM_CUBIC_ *
            (constants::_ANGSTROM_TO_METER_ * constants::_ANGSTROM_TO_METER_ *
             constants::_ANGSTROM_TO_METER_),
        1.0,
        1e-9
    );
}
TEST(TestConstants, angstromCubicToLiter)
{
    EXPECT_NEAR(
        constants::_ANGSTROM_CUBIC_TO_LITER_ /
            (constants::_ANGSTROM_TO_METER_ * constants::_ANGSTROM_TO_METER_ *
             constants::_ANGSTROM_TO_METER_ * 1.0e3),
        1.0,
        1e-9
    );
}
TEST(TestConstants, literToAngstromCubic)
{
    EXPECT_NEAR(
        constants::_LITER_TO_ANGSTROM_CUBIC_ *
            (constants::_ANGSTROM_TO_METER_ * constants::_ANGSTROM_TO_METER_ *
             constants::_ANGSTROM_TO_METER_ * 1.0e3),
        1.0,
        1e-9
    );
}
TEST(TestConstants, bohrRadiusCubicToAngstromCubic)
{
    EXPECT_NEAR(
        constants::_BOHR_RADIUS_CUBIC_TO_ANGSTROM_CUBIC_ /
            (constants::_BOHR_RADIUS_TO_ANGSTROM_ *
             constants::_BOHR_RADIUS_TO_ANGSTROM_ *
             constants::_BOHR_RADIUS_TO_ANGSTROM_),
        1.0,
        1e-9
    );
}

// for density units
TEST(TestConstants, kgPerLiterToAmuPerAngstromCubic)
{
    EXPECT_NEAR(
        constants::_KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_ /
            (constants::_KG_TO_AMU_ / constants::_LITER_TO_ANGSTROM_CUBIC_),
        1.0,
        1e-9
    );
}
TEST(TestConstants, amuPerAngstromCubicToKgPerLiter)
{
    EXPECT_NEAR(
        constants::_AMU_PER_ANGSTROM_CUBIC_TO_KG_PER_LITER_CUBIC_ *
            constants::_KG_PER_LITER_TO_AMU_PER_ANGSTROM_CUBIC_,
        1.0,
        1e-9
    );
}

// for energy units
TEST(TestConstants, kcalToJoule)
{
    EXPECT_NEAR(constants::_KCAL_TO_JOULE_ / 4184.0, 1.0, 1e-9);
}
TEST(TestConstants, jouleToKcal)
{
    EXPECT_NEAR(constants::_JOULE_TO_KCAL_ * 4184.0, 1.0, 1e-9);
}
TEST(TestConstants, jouleToKcalPerMol)
{
    EXPECT_NEAR(
        constants::_JOULE_TO_KCAL_PER_MOL_ / constants::_JOULE_TO_KCAL_ /
            constants::_AVOGADRO_NUMBER_,
        1.0,
        1e-9
    );
}
TEST(TestConstants, kcalPerMolToJoule)
{
    EXPECT_NEAR(
        constants::_KCAL_PER_MOL_TO_JOULE_ * constants::_JOULE_TO_KCAL_ *
            constants::_AVOGADRO_NUMBER_,
        1.0,
        1e-9
    );
}
TEST(TestConstants, hartreeToKcalPerMol)
{
    EXPECT_NEAR(
        constants::_HARTREE_TO_KCAL_PER_MOL_ / 627.5096080305927,
        1.0,
        1e-9
    );
}
TEST(TestConstants, boltzmannConstantInKcalPerMol)
{
    EXPECT_NEAR(
        constants::_BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL_ /
            constants::_BOLTZMANN_CONSTANT_ /
            constants::_JOULE_TO_KCAL_PER_MOL_,
        1.0,
        1e-9
    );
}

// for squared energy units
TEST(TestConstants, boltzmannConstantSquared)
{
    EXPECT_NEAR(
        constants::_BOLTZMANN_CONSTANT_SQUARED_ /
            constants::_BOLTZMANN_CONSTANT_ / constants::_BOLTZMANN_CONSTANT_,
        1.0,
        1e-9
    );
}
TEST(TestConstants, reducedPlanckConstantSquared)
{
    EXPECT_NEAR(
        constants::_REDUCED_PLANCK_CONSTANT_SQUARED_ /
            constants::_REDUCED_PLANCK_CONSTANT_ /
            constants::_REDUCED_PLANCK_CONSTANT_,
        1.0,
        1e-9
    );
}

// for force units
TEST(TestConstants, hartreePerBohrToKcalPerMolPerAngstrom)
{
    EXPECT_NEAR(
        constants::_HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM_ /
            (constants::_HARTREE_TO_KCAL_PER_MOL_ /
             constants::_BOHR_RADIUS_TO_ANGSTROM_),
        1.0,
        1e-9
    );
}

// for stress units
TEST(TestConstants, hartreePerBohrCubicToKcalPerMolPerAngstromCubic)
{
    EXPECT_NEAR(
        constants::_HARTREE_PER_BOHR_CUBIC_TO_KCAL_PER_MOL_PER_ANGSTROM_CUBIC_ /
            (constants::_HARTREE_TO_KCAL_PER_MOL_ /
             constants::_BOHR_RADIUS_TO_ANGSTROM_ /
             constants::_BOHR_RADIUS_TO_ANGSTROM_ /
             constants::_BOHR_RADIUS_TO_ANGSTROM_),
        1.0,
        1e-9
    );
}

// for time units
TEST(TestConstants, femtosecondToSecond)
{
    EXPECT_NEAR(constants::_FS_TO_S_ / 1.0e-15, 1.0, 1e-9);
}
TEST(TestConstants, secondToFemtosecond)
{
    EXPECT_NEAR(constants::_S_TO_FS_ / 1.0e15, 1.0, 1e-9);
}
TEST(TestConstants, picosecondToFemtosecond)
{
    EXPECT_NEAR(constants::_PS_TO_FS_ / 1.0e3, 1.0, 1e-9);
}
TEST(TestConstants, femtosecondToPicosecond)
{
    EXPECT_NEAR(constants::_FS_TO_PS_ / 1.0e-3, 1.0, 1e-9);
}

// for pressure units
TEST(TestConstants, barToPascal)
{
    EXPECT_NEAR(constants::_BAR_TO_PASCAL_ / 1.0e5, 1.0, 1e-9);
}
TEST(TestConstants, pascalToBar)
{
    EXPECT_NEAR(constants::_PASCAL_TO_BAR_ * 1.0e5, 1.0, 1e-9);
}

// for velocity units
TEST(TestConstants, meterPerSecondToCentimeterPerPicosecond)
{
    EXPECT_NEAR(constants::_M_PER_S_TO_CM_PER_S_ / 1.0e2, 1.0, 1e-9);
}
TEST(TestConstants, speedOfLightInCentimeterPerSecond)
{
    EXPECT_NEAR(
        constants::_SPEED_OF_LIGHT_IN_CM_PER_S_ /
            (constants::_SPEED_OF_LIGHT_ * constants::_M_PER_S_TO_CM_PER_S_),
        1.0,
        1e-9
    );
}

// for frequency units
TEST(TestConstants, perCentiMeterToHertz)
{
    EXPECT_NEAR(
        constants::_PER_CM_TO_HZ_ / constants::_SPEED_OF_LIGHT_IN_CM_PER_S_,
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
        constants::_FORCE_UNIT_TO_SI_ / (constants::_KCAL_PER_MOL_TO_JOULE_ /
                                         constants::_ANGSTROM_TO_METER_),
        1.0,
        1e-9
    );
}
TEST(TestConstants, massUnitToSI)
{
    EXPECT_NEAR(
        constants::_MASS_UNIT_TO_SI_ / constants::_AMU_TO_KG_,
        1.0,
        1e-9
    );
}
TEST(TestConstants, timeUnitToSI)
{
    EXPECT_NEAR(constants::_TIME_UNIT_TO_SI_ / constants::_FS_TO_S_, 1.0, 1e-9);
}
TEST(TestConstants, velocityUnitToSI)
{
    EXPECT_NEAR(
        constants::_VELOCITY_UNIT_TO_SI_ / constants::_ANGSTROM_TO_METER_,
        1.0,
        1e-9
    );
}
TEST(TestConstants, energyUnitToSI)
{
    EXPECT_NEAR(
        constants::_ENERGY_UNIT_TO_SI_ /
            (constants::_KCAL_TO_JOULE_ / constants::_AVOGADRO_NUMBER_),
        1.0,
        1e-9
    );
}
TEST(TestConstants, volumeUnitToSI)
{
    EXPECT_NEAR(
        constants::_VOLUME_UNIT_TO_SI_ /
            (constants::_ANGSTROM_CUBIC_TO_METER_CUBIC_),
        1.0,
        1e-9
    );
}
TEST(TestConstants, pressureUnitToSI)
{
    EXPECT_NEAR(
        constants::_PRESSURE_UNIT_TO_SI_ / constants::_BAR_TO_PASCAL_,
        1.0,
        1e-9
    );
}
TEST(TestConstants, lengthUnitToSI)
{
    EXPECT_NEAR(
        constants::_LENGTH_UNIT_TO_SI_ / constants::_ANGSTROM_TO_METER_,
        1.0,
        1e-9
    );
}
TEST(TestConstants, momentumUnitToSI)
{
    EXPECT_NEAR(
        constants::_MOMENTUM_UNIT_TO_SI_ /
            (constants::_GRAM_TO_KG_ * constants::_ANGSTROM_TO_METER_ /
             constants::_AVOGADRO_NUMBER_),
        1.0,
        1e-9
    );
}

// for SI to internal units
TEST(TestConstants, siToVelocityUnit)
{
    EXPECT_NEAR(
        constants::_SI_TO_VELOCITY_UNIT_ * constants::_VELOCITY_UNIT_TO_SI_,
        1.0,
        1e-9
    );
}
TEST(TestConstants, siToEnergyUnit)
{
    EXPECT_NEAR(
        constants::_SI_TO_ENERGY_UNIT_ * constants::_ENERGY_UNIT_TO_SI_,
        1.0,
        1e-9
    );
}
TEST(TestConstants, siToPressureUnit)
{
    EXPECT_NEAR(
        constants::_SI_TO_PRESSURE_UNIT_ * constants::_PRESSURE_UNIT_TO_SI_,
        1.0,
        1e-9
    );
}
TEST(TestConstants, siToLengthUnit)
{
    EXPECT_NEAR(
        constants::_SI_TO_LENGTH_UNIT_ * constants::_LENGTH_UNIT_TO_SI_,
        1.0,
        1e-9
    );
}
TEST(TestConstants, siToForceUnit)
{
    EXPECT_NEAR(
        constants::_SI_TO_FORCE_UNIT_ * constants::_FORCE_UNIT_TO_SI_,
        1.0,
        1e-9
    );
}

// for velocity verlet integrator
TEST(TestConstants, vVerletVelocityFactor)
{
    EXPECT_NEAR(
        constants::_V_VERLET_VELOCITY_FACTOR_ /
            (0.5 *
             (constants::_FORCE_UNIT_TO_SI_ / constants::_MASS_UNIT_TO_SI_) *
             constants::_TIME_UNIT_TO_SI_ * constants::_SI_TO_VELOCITY_UNIT_),
        1.0,
        1e-9
    );
}

// for temperature calculation
TEST(TestConstants, temperatureFactor)
{
    EXPECT_NEAR(
        constants::_TEMPERATURE_FACTOR_ /
            (constants::_VELOCITY_UNIT_TO_SI_ *
             constants::_VELOCITY_UNIT_TO_SI_ * constants::_MASS_UNIT_TO_SI_ /
             constants::_BOLTZMANN_CONSTANT_),
        1.0,
        1e-9
    );
}

// for kinetic energy
TEST(TestConstants, kineticEnergyFactor)
{
    EXPECT_NEAR(
        constants::_KINETIC_ENERGY_FACTOR_ /
            (0.5 * constants::_MASS_UNIT_TO_SI_ *
             constants::_VELOCITY_UNIT_TO_SI_ *
             constants::_VELOCITY_UNIT_TO_SI_ * constants::_SI_TO_ENERGY_UNIT_),
        1.0,
        1e-9
    );
}

// for pressure calculation
TEST(TestConstants, pressureFactor)
{
    EXPECT_NEAR(
        constants::_PRESSURE_FACTOR_ /
            (constants::_ENERGY_UNIT_TO_SI_ / constants::_VOLUME_UNIT_TO_SI_ *
             constants::_SI_TO_PRESSURE_UNIT_),
        1.0,
        1e-9
    );
}

// for coulomb prefactor
TEST(TestConstants, coulombPrefactor)
{
    EXPECT_NEAR(
        constants::_COULOMB_PREFACTOR_ /
            (constants::_ELECTRON_CHARGE_ * constants::_ELECTRON_CHARGE_ *
             constants::_SI_TO_LENGTH_UNIT_ * constants::_SI_TO_ENERGY_UNIT_) *
            constants::_PERMITTIVITY_VACUUM_ * 4 * M_PI,
        1.0,
        1e-9
    );
}

// for ring polymer molecular dynamics
TEST(TestConstants, ringPolymerMolecularDynamics)
{
    EXPECT_NEAR(
        constants::_RPMD_PREFACTOR_ /
            (constants::_BOLTZMANN_CONSTANT_SQUARED_ /
             constants::_REDUCED_PLANCK_CONSTANT_SQUARED_ /
             constants::_METER_SQUARED_TO_ANGSTROM_SQUARED_ *
             constants::_GRAM_TO_KG_ * constants::_JOULE_TO_KCAL_),
        1.0,
        1e-9
    );
}

// for momentum to force
TEST(TestConstants, momentumToForce)
{
    EXPECT_NEAR(
        constants::_MOMENTUM_TO_FORCE_ /
            (constants::_MASS_UNIT_TO_SI_ * constants::_VELOCITY_UNIT_TO_SI_ *
             constants::_SI_TO_FORCE_UNIT_),
        1.0,
        1e-9
    );
}