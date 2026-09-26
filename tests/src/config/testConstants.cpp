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

/*********************
 * natural constants *
 *********************/

TEST(TestConstants, avogadroNumber)
{
    EXPECT_NEAR(AVOGADRO_NUMBER / 6.02214076e23, 1.0, 1e-9);
}

TEST(TestConstants, bohrRadius)
{
    EXPECT_NEAR(BOHR_RADIUS / 5.29177210903e-11, 1.0, 1e-9);
}

TEST(TestConstants, planckConstant)
{
    EXPECT_NEAR(PLANCK_CONSTANT / 6.62607015e-34, 1.0, 1e-9);
}
TEST(TestConstants, reducedPlanckConstant)
{
    EXPECT_NEAR(REDUCED_PLANCK_CONSTANT / 1.054571817e-34, 1.0, 1e-9);
}

TEST(TestConstants, boltzmannConstant)
{
    EXPECT_NEAR(BOLTZMANN_CONSTANT / 1.380649e-23, 1.0, 1e-9);
}
TEST(TestConstants, universalGasConstant)
{
    EXPECT_NEAR(UNIVERSAL_GAS_CONSTANT / 8.3144626181532395, 1.0, 1e-9);
}

TEST(TestConstants, electronCharge)
{
    EXPECT_NEAR(ELECTRON_CHARGE / 1.602176634e-19, 1.0, 1e-9);
}
TEST(TestConstants, electronChargeSquared)
{
    EXPECT_NEAR(
        ELECTRON_CHARGE2 / (ELECTRON_CHARGE * ELECTRON_CHARGE),
        1.0,
        1e-9
    );
}

TEST(TestConstants, electronMass)
{
    EXPECT_NEAR(ELECTRON_MASS / 9.109389754e-31, 1.0, 1e-9);
}

TEST(TestConstants, permittivityVacuum)
{
    EXPECT_NEAR(PERMITTIVITY_VACUUM / 8.8541878128e-12, 1.0, 1e-9);
}

TEST(TestConstants, speedOfLight)
{
    EXPECT_NEAR(SPEED_OF_LIGHT / 299792458.0, 1.0, 1e-9);
}

/**********************
 * conversion factors *
 **********************/

// for degree units
TEST(TestConstants, degreesToRadians)
{
    EXPECT_NEAR(DEG_TO_RAD / (M_PI / 180.0), 1.0, 1e-9);
}
TEST(TestConstants, radiansToDegrees)
{
    EXPECT_NEAR(RAD_TO_DEG / (180.0 / M_PI), 1.0, 1e-9);
}

// for mass units
TEST(TestConstants, gramToKilogram)
{
    EXPECT_NEAR(G_TO_KG / 1.0e-3, 1.0, 1e-9);
}
TEST(TestConstants, kilogramToGram)
{
    EXPECT_NEAR(KG_TO_GRAM / 1.0e3, 1.0, 1e-9);
}
TEST(TestConstants, amuToKilogram)
{
    EXPECT_NEAR(AMU_TO_KG / 1.6605402e-27, 1.0, 1e-6);
}
TEST(TestConstants, kilogramToAmu)
{
    EXPECT_NEAR(KG_TO_AMU * AMU_TO_KG, 1.0, 1e-9);
}

// for length units
TEST(TestConstants, angstromToMeter)
{
    EXPECT_NEAR(ANGSTROM_TO_M / 1.0e-10, 1.0, 1e-9);
}
TEST(TestConstants, meterToAngstrom)
{
    EXPECT_NEAR(M_TO_ANGSTROM / 1.0e10, 1.0, 1e-9);
}
TEST(TestConstants, bohrRadiusToMeter)
{
    EXPECT_NEAR(BOHR_TO_M / BOHR_RADIUS, 1.0, 1e-9);
}
TEST(TestConstants, meterToBohrRadius)
{
    EXPECT_NEAR(M_TO_BOHR * BOHR_RADIUS, 1.0, 1e-9);
}
TEST(TestConstants, angstromToBohrRadius)
{
    EXPECT_NEAR(ANGSTROM_TO_BOHR / (ANGSTROM_TO_M / BOHR_RADIUS), 1.0, 1e-9);
}
TEST(TestConstants, bohrRadiusToAngstrom)
{
    EXPECT_NEAR(BOHR_TO_ANGSTROM * (ANGSTROM_TO_M / BOHR_RADIUS), 1.0, 1e-9);
}

// for area units
TEST(TestConstants, angstromSquaredToMeterSquared)
{
    EXPECT_NEAR(ANGSTROM2_TO_M2 / (ANGSTROM_TO_M * ANGSTROM_TO_M), 1.0, 1e-9);
}
TEST(TestConstants, meterSquaredToAngstromSquared)
{
    EXPECT_NEAR(M2_TO_ANGSTROM2 * (ANGSTROM_TO_M * ANGSTROM_TO_M), 1.0, 1e-9);
}

// for volume units
TEST(TestConstants, angstromCubicToMeterCubic)
{
    EXPECT_NEAR(
        ANGSTROM3_TO_M3 / (ANGSTROM_TO_M * ANGSTROM_TO_M * ANGSTROM_TO_M),
        1.0,
        1e-9
    );
}
TEST(TestConstants, meterCubicToAngstromCubic)
{
    EXPECT_NEAR(
        M3_TO_ANGSTROM3 * (ANGSTROM_TO_M * ANGSTROM_TO_M * ANGSTROM_TO_M),
        1.0,
        1e-9
    );
}
TEST(TestConstants, angstromCubicToLiter)
{
    EXPECT_NEAR(
        ANGSTROM3_TO_L /
            (ANGSTROM_TO_M * ANGSTROM_TO_M * ANGSTROM_TO_M * 1.0e3),
        1.0,
        1e-9
    );
}
TEST(TestConstants, literToAngstromCubic)
{
    EXPECT_NEAR(
        L_TO_ANGSTROM3 *
            (ANGSTROM_TO_M * ANGSTROM_TO_M * ANGSTROM_TO_M * 1.0e3),
        1.0,
        1e-9
    );
}
TEST(TestConstants, bohrRadiusCubicToAngstromCubic)
{
    EXPECT_NEAR(
        BOHR3_TO_ANGSTROM3 /
            (BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM * BOHR_TO_ANGSTROM),
        1.0,
        1e-9
    );
}

// for density units
TEST(TestConstants, kgPerLiterToAmuPerAngstromCubic)
{
    EXPECT_NEAR(
        KG_PER_L_TO_AMU_PER_ANGSTROM3 / (KG_TO_AMU / L_TO_ANGSTROM3),
        1.0,
        1e-9
    );
}
TEST(TestConstants, amuPerAngstromCubicToKgPerLiter)
{
    EXPECT_NEAR(
        AMU_PER_ANGSTROM3_TO_KG_PER_L * KG_PER_L_TO_AMU_PER_ANGSTROM3,
        1.0,
        1e-9
    );
}

// for energy units
TEST(TestConstants, kcalToJoule) { EXPECT_NEAR(KCAL_TO_J / 4184.0, 1.0, 1e-9); }
TEST(TestConstants, jouleToKcal) { EXPECT_NEAR(J_TO_KCAL * 4184.0, 1.0, 1e-9); }
TEST(TestConstants, jouleToKcalPerMol)
{
    EXPECT_NEAR(J_TO_KCAL_PER_MOL / J_TO_KCAL / AVOGADRO_NUMBER, 1.0, 1e-9);
}
TEST(TestConstants, kcalPerMolToJoule)
{
    EXPECT_NEAR(KCAL_PER_MOL_TO_J * J_TO_KCAL * AVOGADRO_NUMBER, 1.0, 1e-9);
}
TEST(TestConstants, hartreeToKcalPerMol)
{
    EXPECT_NEAR(HARTREE_TO_KCAL_PER_MOL / 627.5096080305927, 1.0, 1e-9);
}
TEST(TestConstants, boltzmannConstantInKcalPerMol)
{
    EXPECT_NEAR(
        BOLTZMANN_CONSTANT_IN_KCAL_PER_MOL / BOLTZMANN_CONSTANT /
            J_TO_KCAL_PER_MOL,
        1.0,
        1e-9
    );
}

// for squared energy units
TEST(TestConstants, boltzmannConstantSquared)
{
    EXPECT_NEAR(
        BOLTZMANN_CONSTANT2 / BOLTZMANN_CONSTANT / BOLTZMANN_CONSTANT,
        1.0,
        1e-9
    );
}
TEST(TestConstants, reducedPlanckConstantSquared)
{
    EXPECT_NEAR(
        REDUCED_PLANCK_CONSTANT2 / REDUCED_PLANCK_CONSTANT /
            REDUCED_PLANCK_CONSTANT,
        1.0,
        1e-9
    );
}

// for force units
TEST(TestConstants, hartreePerBohrToKcalPerMolPerAngstrom)
{
    EXPECT_NEAR(
        HARTREE_PER_BOHR_TO_KCAL_PER_MOL_PER_ANGSTROM /
            (HARTREE_TO_KCAL_PER_MOL / BOHR_TO_ANGSTROM),
        1.0,
        1e-9
    );
}

// for stress units
TEST(TestConstants, hartreePerBohrCubicToKcalPerMolPerAngstromCubic)
{
    EXPECT_NEAR(
        HARTREE_PER_BOHR3_TO_KCAL_PER_MOL_PER_ANGSTROM3 /
            (HARTREE_TO_KCAL_PER_MOL / BOHR_TO_ANGSTROM / BOHR_TO_ANGSTROM /
             BOHR_TO_ANGSTROM),
        1.0,
        1e-9
    );
}

// for time units
TEST(TestConstants, femtosecondToSecond)
{
    EXPECT_NEAR(FS_TO_S / 1.0e-15, 1.0, 1e-9);
}
TEST(TestConstants, secondToFemtosecond)
{
    EXPECT_NEAR(S_TO_FS / 1.0e15, 1.0, 1e-9);
}
TEST(TestConstants, picosecondToFemtosecond)
{
    EXPECT_NEAR(PS_TO_FS / 1.0e3, 1.0, 1e-9);
}
TEST(TestConstants, femtosecondToPicosecond)
{
    EXPECT_NEAR(FS_TO_PS / 1.0e-3, 1.0, 1e-9);
}

// for pressure units
TEST(TestConstants, barToPascal) { EXPECT_NEAR(BAR_TO_P / 1.0e5, 1.0, 1e-9); }
TEST(TestConstants, pascalToBar) { EXPECT_NEAR(P_TO_BAR * 1.0e5, 1.0, 1e-9); }

// for velocity units
TEST(TestConstants, meterPerSecondToCentimeterPerPicosecond)
{
    EXPECT_NEAR(M_PER_S_TO_CM_PER_S / 1.0e2, 1.0, 1e-9);
}
TEST(TestConstants, speedOfLightInCentimeterPerSecond)
{
    EXPECT_NEAR(
        SPEED_OF_LIGHT_IN_CM_PER_S / (SPEED_OF_LIGHT * M_PER_S_TO_CM_PER_S),
        1.0,
        1e-9
    );
}

// for frequency units
TEST(TestConstants, perCentiMeterToHertz)
{
    EXPECT_NEAR(PER_CM_TO_HZ / SPEED_OF_LIGHT_IN_CM_PER_S, 1.0, 1e-9);
}

/*******************************
 * internal conversion factors *
 *******************************/

// for internal to SI units
TEST(TestConstants, forceUnitToSI)
{
    EXPECT_NEAR(
        FORCE_UNIT_TO_SI / (KCAL_PER_MOL_TO_J / ANGSTROM_TO_M),
        1.0,
        1e-9
    );
}
TEST(TestConstants, massUnitToSI)
{
    EXPECT_NEAR(MASS_UNIT_TO_SI / AMU_TO_KG, 1.0, 1e-9);
}
TEST(TestConstants, timeUnitToSI)
{
    EXPECT_NEAR(TIME_UNIT_TO_SI / FS_TO_S, 1.0, 1e-9);
}
TEST(TestConstants, velocityUnitToSI)
{
    EXPECT_NEAR(VELOCITY_UNIT_TO_SI / ANGSTROM_TO_M, 1.0, 1e-9);
}
TEST(TestConstants, energyUnitToSI)
{
    EXPECT_NEAR(ENERGY_UNIT_TO_SI / (KCAL_TO_J / AVOGADRO_NUMBER), 1.0, 1e-9);
}
TEST(TestConstants, volumeUnitToSI)
{
    EXPECT_NEAR(VOLUME_UNIT_TO_SI / (ANGSTROM3_TO_M3), 1.0, 1e-9);
}
TEST(TestConstants, pressureUnitToSI)
{
    EXPECT_NEAR(PRESSURE_UNIT_TO_SI / BAR_TO_P, 1.0, 1e-9);
}
TEST(TestConstants, lengthUnitToSI)
{
    EXPECT_NEAR(LENGTH_UNIT_TO_SI / ANGSTROM_TO_M, 1.0, 1e-9);
}
TEST(TestConstants, momentumUnitToSI)
{
    EXPECT_NEAR(
        MOMENTUM_UNIT_TO_SI / (G_TO_KG * ANGSTROM_TO_M / AVOGADRO_NUMBER),
        1.0,
        1e-9
    );
}

// for SI to internal units
TEST(TestConstants, siToVelocityUnit)
{
    EXPECT_NEAR(SI_TO_VELOCITY_UNIT * VELOCITY_UNIT_TO_SI, 1.0, 1e-9);
}
TEST(TestConstants, siToEnergyUnit)
{
    EXPECT_NEAR(SI_TO_ENERGY_UNIT * ENERGY_UNIT_TO_SI, 1.0, 1e-9);
}
TEST(TestConstants, siToPressureUnit)
{
    EXPECT_NEAR(SI_TO_PRESSURE_UNIT * PRESSURE_UNIT_TO_SI, 1.0, 1e-9);
}
TEST(TestConstants, siToLengthUnit)
{
    EXPECT_NEAR(SI_TO_LENGTH_UNIT * LENGTH_UNIT_TO_SI, 1.0, 1e-9);
}
TEST(TestConstants, siToForceUnit)
{
    EXPECT_NEAR(SI_TO_FORCE_UNIT * FORCE_UNIT_TO_SI, 1.0, 1e-9);
}

// for velocity verlet integrator
TEST(TestConstants, vVerletVelocityFactor)
{
    EXPECT_NEAR(
        V_VERLET_VELOCITY_FACTOR / (0.5 * (FORCE_UNIT_TO_SI / MASS_UNIT_TO_SI) *
                                    TIME_UNIT_TO_SI * SI_TO_VELOCITY_UNIT),
        1.0,
        1e-9
    );
}

// for temperature calculation
TEST(TestConstants, temperatureFactor)
{
    EXPECT_NEAR(
        TEMPERATURE_FACTOR / (VELOCITY_UNIT_TO_SI * VELOCITY_UNIT_TO_SI *
                              MASS_UNIT_TO_SI / BOLTZMANN_CONSTANT),
        1.0,
        1e-9
    );
}

// for kinetic energy
TEST(TestConstants, kineticEnergyFactor)
{
    EXPECT_NEAR(
        KINETIC_ENERGY_FACTOR / (0.5 * MASS_UNIT_TO_SI * VELOCITY_UNIT_TO_SI *
                                 VELOCITY_UNIT_TO_SI * SI_TO_ENERGY_UNIT),
        1.0,
        1e-9
    );
}

// for pressure calculation
TEST(TestConstants, pressureFactor)
{
    EXPECT_NEAR(
        PRESSURE_FACTOR /
            (ENERGY_UNIT_TO_SI / VOLUME_UNIT_TO_SI * SI_TO_PRESSURE_UNIT),
        1.0,
        1e-9
    );
}

// for coulomb prefactor
TEST(TestConstants, coulombPrefactor)
{
    EXPECT_NEAR(
        COULOMB_PREFACTOR /
            (ELECTRON_CHARGE * ELECTRON_CHARGE * SI_TO_LENGTH_UNIT *
             SI_TO_ENERGY_UNIT) *
            PERMITTIVITY_VACUUM * 4 * M_PI,
        1.0,
        1e-9
    );
}

// for ring polymer molecular dynamics
TEST(TestConstants, ringPolymerMolecularDynamics)
{
    EXPECT_NEAR(
        RPMD_PREFACTOR / (BOLTZMANN_CONSTANT2 / REDUCED_PLANCK_CONSTANT2 /
                          M2_TO_ANGSTROM2 * G_TO_KG * J_TO_KCAL),
        1.0,
        1e-9
    );
}

// for momentum to force
TEST(TestConstants, momentumToForce)
{
    EXPECT_NEAR(
        MOMENTUM_TO_FORCE /
            (MASS_UNIT_TO_SI * VELOCITY_UNIT_TO_SI * SI_TO_FORCE_UNIT),
        1.0,
        1e-9
    );
}
