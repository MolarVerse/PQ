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

#include "simulationBox_API.hpp"

#include "constants.hpp"
#include "linearAlgebra.hpp"
#include "orthorhombicBox.inl"
#include "settings.hpp"
#include "simulationBox.hpp"   // IWYU pragma: keep
#include "triclinicBox.inl"

using namespace simulationBox;
using namespace linearAlgebra;
using namespace settings;
using namespace constants;

/**
 * @brief Get the minimum, maximum, sum and mean of the positions
 *
 * @param simBox
 *
 * @return std::tuple<Real, Real, Real, Real>
 */
std::tuple<Real, Real, Real, Real> simulationBox::posMinMaxSumMean(
    pq::SimBox& simBox
)
{
    bool useDevice = false;
#ifdef __PQ_GPU__
    useDevice = Settings::useDevice();
#endif

    auto* const pos    = simBox.getPosPtr();
    const auto  nAtoms = simBox.getNumberOfAtoms();

    return minMaxSumMean(pos, nAtoms * 3, useDevice);
}

/**
 * @brief Get the minimum, maximum, sum and mean of the velocities
 *
 * @param simBox
 *
 * @return std::tuple<Real, Real, Real, Real>
 */
std::tuple<Real, Real, Real, Real> simulationBox::velMinMaxSumMean(
    pq::SimBox& simBox
)
{
    bool useDevice = false;
#ifdef __PQ_GPU__
    useDevice = Settings::useDevice();
#endif

    auto* const vel    = simBox.getVelPtr();
    const auto  nAtoms = simBox.getNumberOfAtoms();

    return minMaxSumMean(vel, nAtoms * 3, useDevice);
}

/**
 * @brief Get the minimum, maximum, sum and mean of the forces
 *
 * @param simBox
 *
 * @return std::tuple<Real, Real, Real, Real>
 */
std::tuple<Real, Real, Real, Real> simulationBox::forcesMinMaxSumMean(
    pq::SimBox& simBox
)
{
    bool useDevice = false;
#ifdef __PQ_GPU__
    useDevice = Settings::useDevice();
#endif

    auto* const forces = simBox.getForcesPtr();
    const auto  nAtoms = simBox.getNumberOfAtoms();

    return minMaxSumMean(forces, nAtoms * 3, useDevice);
}

/**
 * @brief Get the minimum, maximum, sum and mean of the shift forces
 *
 * @param simBox
 *
 * @return std::tuple<Real, Real, Real, Real>
 */
std::tuple<Real, Real, Real, Real> simulationBox::shiftForcesMinMaxSumMean(
    pq::SimBox& simBox
)
{
    bool useDevice = false;
#ifdef __PQ_GPU__
    useDevice = Settings::useDevice();
#endif

    auto* const shiftForces = simBox.getShiftForcesPtr();
    const auto  nAtoms      = simBox.getNumberOfAtoms();

    return minMaxSumMean(shiftForces, nAtoms * 3, useDevice);
}

/**
 * @brief Get the minimum, maximum, sum and mean of the center of mass of the
 * molecules
 *
 * @param simBox
 *
 * @return std::tuple<Real, Real, Real, Real>
 */
std::tuple<Real, Real, Real, Real> simulationBox::comMoleculesMinMaxSumMean(
    pq::SimBox& simBox
)
{
    bool useDevice = false;
#ifdef __PQ_GPU__
    useDevice = Settings::useDevice();
#endif

    auto* const comMolecules = simBox.getComMoleculesPtr();
    const auto  nMolecules   = simBox.getNumberOfMolecules();

    return minMaxSumMean(comMolecules, nMolecules * 3, useDevice);
}

/**
 * @brief Get the minimum, maximum, sum and mean of the molecular masses
 *
 * @param simBox
 *
 * @return std::tuple<Real, Real, Real, Real>
 */
std::tuple<Real, Real, Real, Real> simulationBox::molMassesMinMaxSumMean(
    pq::SimBox& simBox
)
{
    bool useDevice = false;
#ifdef __PQ_GPU__
    useDevice = Settings::useDevice();
#endif

    auto* const molMasses  = simBox.getMolMassesPtr();
    const auto  nMolecules = simBox.getNumberOfMolecules();

    return minMaxSumMean(molMasses, nMolecules, useDevice);
}

/**
 * @brief calculate temperature of simulationBox
 *
 */
double simulationBox::calculateTemperature(pq::SimBox& simBox)
{
    __DEBUG_ENTER_FUNCTION__("Temperature calculation");

    auto temperature = 0.0;

    const auto* const velPtr  = simBox.getVelPtr();
    const auto* const massPtr = simBox.getMassesPtr();
    const auto        nAtoms  = simBox.getNumberOfAtoms();

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for \
                collapse(2)                          \
                is_device_ptr(velPtr,massPtr)        \
                reduction(+:temperature)             \
                map(temperature)
#else
    #pragma omp parallel for                         \
                collapse(2)                          \
                reduction(+:temperature)
#endif
    // clang-format on
    for (size_t i = 0; i < nAtoms; ++i)
        for (size_t j = 0; j < 3; ++j)
            temperature += massPtr[i] * velPtr[i * 3 + j] * velPtr[i * 3 + j];

    const auto dof  = simBox.getDegreesOfFreedom();
    temperature    *= _TEMPERATURE_FACTOR_ / double(dof);

    __DEBUG_TEMPERATURE__(temperature);
    __DEBUG_EXIT_FUNCTION__("Temperature calculation");

    return temperature;
}

/**
 * @brief calculate momentum of simulationBox
 *
 * @return Vec3D
 */
Vec3D simulationBox::calculateMomentum(pq::SimBox& simBox)
{
    __DEBUG_ENTER_FUNCTION__("Momentum calculation");

    auto momentumX = 0.0;
    auto momentumY = 0.0;
    auto momentumZ = 0.0;

    const auto* const velPtr    = simBox.getVelPtr();
    const auto* const massesPtr = simBox.getMassesPtr();
    const auto        nAtoms    = simBox.getNumberOfAtoms();

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for         \
                is_device_ptr(velPtr, massesPtr)             \
                map(momentumX, momentumY, momentumZ)         \
                reduction(+:momentumX, momentumY, momentumZ)
#else
    #pragma omp parallel for                                 \
                reduction(+:momentumX, momentumY, momentumZ)
#endif
    // clang-format on
    for (size_t i = 0; i < nAtoms; ++i)
    {
        momentumX += massesPtr[i] * velPtr[i * 3];
        momentumY += massesPtr[i] * velPtr[i * 3 + 1];
        momentumZ += massesPtr[i] * velPtr[i * 3 + 2];
    }

    const auto momentum = Vec3D{momentumX, momentumY, momentumZ};

    __DEBUG_MOMENTUM__(momentum);
    __DEBUG_EXIT_FUNCTION__("Momentum calculation");

    return momentum;
}

/**
 * @brief calculate angular momentum of simulationBox
 *
 */
Vec3D simulationBox::calculateAngularMomentum(
    SimulationBox& simBox,
    const Vec3D&   momentum
)
{
    __DEBUG_ENTER_FUNCTION__("Angular momentum calculation");

    Real angularMomX = 0.0;
    Real angularMomY = 0.0;
    Real angularMomZ = 0.0;

    const auto* const massesPtr = simBox.getMassesPtr();
    const auto* const posPtr    = simBox.getPosPtr();
    const auto* const velPtr    = simBox.getVelPtr();
    const auto        nAtoms    = simBox.getNumberOfAtoms();

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for                \
                is_device_ptr(massesPtr, posPtr, velPtr)            \
                map(angularMomX, angularMomY, angularMomZ)          \
                reduction(+:angularMomX, angularMomY, angularMomZ)
#else
    #pragma omp parallel for                                        \
                reduction(+:angularMomX, angularMomY, angularMomZ)
#endif
    // clang-format on
    for (size_t i = 0; i < nAtoms; ++i)
    {
        Real       angularMom_[3];
        const auto mass = massesPtr[i];

        cross(angularMom_, posPtr + 3 * i, velPtr + 3 * i);

        angularMomX += angularMom_[0] * mass;
        angularMomY += angularMom_[1] * mass;
        angularMomZ += angularMom_[2] * mass;
    }

    auto angularMom = Vec3D{angularMomX, angularMomY, angularMomZ};

    const auto centerOfMass = calculateCenterOfMass(simBox);
    const auto totalMass    = simBox.getTotalMass();

    const auto correction = cross(centerOfMass, momentum / totalMass) *
                            totalMass;   // TODO: remove totalMass?

    angularMom[0] -= correction[0];
    angularMom[1] -= correction[1];
    angularMom[2] -= correction[2];

    __DEBUG_ANGULAR_MOMENTUM__(angularMom);
    __DEBUG_EXIT_FUNCTION__("Angular momentum calculation");

    return angularMom;
}

/**
 * @brief calculate total mass of simulationBox
 *
 * @param simBox
 * @param updateSimulationBox
 *
 * @return Real
 */
Real simulationBox::calculateTotalMass(
    SimulationBox& simBox,
    const bool     updateSimulationBox
)
{
    Real totalMass = 0.0;

    const auto* const massPtr = simBox.getMassesPtr();
    const auto        nAtoms  = simBox.getNumberOfAtoms();

#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for is_device_ptr(massPtr) \
        map(totalMass) reduction(+ : totalMass)
#else
    #pragma omp parallel for reduction(+ : totalMass)
#endif
    for (size_t i = 0; i < nAtoms; ++i)
        totalMass += massPtr[i];

    if (updateSimulationBox)
        simBox.setTotalMass(totalMass);

    return totalMass;
}

/**
 * @brief calculate total mass of simulationBox
 *
 * @param simBox
 *
 * @return Real
 */
Real simulationBox::calculateTotalMass(SimulationBox& simBox)
{
    return calculateTotalMass(simBox, true);
}

/**
 * @brief calculate total charge of simulationBox
 *
 * @param simBox
 * @param updateSimulationBox
 *
 * @return Real
 */
Real simulationBox::calculateTotalCharge(
    SimulationBox& simBox,
    const bool     updateSimulationBox
)
{
    Real totalCharge = 0.0;

    const auto* const chargesPtr = simBox.getChargesPtr();
    const auto        nAtoms     = simBox.getNumberOfAtoms();

#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for is_device_ptr(chargesPtr) \
        map(totalCharge) reduction(+ : totalCharge)
#else
    #pragma omp parallel for reduction(+ : totalCharge)
#endif
    for (size_t i = 0; i < nAtoms; ++i)
        totalCharge += chargesPtr[i];

    if (updateSimulationBox)
        simBox.setTotalCharge(totalCharge);

    return totalCharge;
}

/**
 * @brief calculate total charge of simulationBox
 *
 * @param simBox
 *
 * @return Real
 */
Real simulationBox::calculateTotalCharge(SimulationBox& simBox)
{
    return calculateTotalCharge(simBox, true);
}

/**
 * @brief calculate center of mass of simulationBox
 *
 * @param simBox
 * @param update
 *
 * @return Vec3D center of mass
 */
Vec3D simulationBox::calculateCenterOfMass(
    SimulationBox& simBox,
    const bool     update
)
{
    auto comX = 0.0;
    auto comY = 0.0;
    auto comZ = 0.0;

    const auto* const posPtr    = simBox.getPosPtr();
    const auto* const massesPtr = simBox.getMassesPtr();
    const auto        nAtoms    = simBox.getNumberOfAtoms();

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for \
                is_device_ptr(posPtr, massesPtr)     \
                map(comX, comY, comZ)                \
                reduction(+:comX, comY, comZ)
#else
    #pragma omp parallel for                         \
                reduction(+:comX, comY, comZ)
#endif
    // clang-format on
    for (size_t i = 0; i < nAtoms; ++i)
    {
        comX += massesPtr[i] * posPtr[i * 3];
        comY += massesPtr[i] * posPtr[i * 3 + 1];
        comZ += massesPtr[i] * posPtr[i * 3 + 2];
    }

    const auto totalMass = simBox.getTotalMass();

    const auto centerOfMass = Vec3D{comX, comY, comZ} / totalMass;

    if (update)
        simBox.setCenterOfMass(centerOfMass);

    return centerOfMass;
}

/**
 * @brief calculate center of mass of simulationBox
 *
 * @param simBox
 *
 * @return Vec3D center of mass
 */
Vec3D simulationBox::calculateCenterOfMass(SimulationBox& simBox)
{
    return calculateCenterOfMass(simBox, true);
}

/**
 * @brief calculate center of mass of all molecules
 *
 */
void simulationBox::calculateCenterOfMassMolecules(SimulationBox& simBox)
{
    __DEBUG_ENTER_FUNCTION__("COM of molecules calculation");

    const auto nMolecules = simBox.getNumberOfMolecules();

    const auto* const posPtr         = simBox.getPosPtr();
    const auto* const massesPtr      = simBox.getMassesPtr();
    const auto* const molMassesPtr   = simBox.getMolMassesPtr();
    const auto* const atomsPerMolPtr = simBox.getAtomsPerMoleculePtr();
    const auto* const boxParams      = simBox.getBox().getBoxParamsPtr();
    const auto* const molOffsetPtr   = simBox.getMoleculeOffsetsPtr();

    auto* const comMolecules   = simBox.getComMoleculesPtr();
    const auto  isOrthorhombic = simBox.getBox().isOrthoRhombic();

    // TODO: implement a shorter version to only calculate shift vector

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for             \
                is_device_ptr(posPtr, massesPtr, molMassesPtr,   \
                    atomsPerMolPtr, boxParams, molOffsetPtr,     \
                    comMolecules)
#else
    #pragma omp parallel for
#endif
    // clang-format on
    for (size_t i = 0; i < nMolecules; ++i)
    {
        auto comX = 0.0;
        auto comY = 0.0;
        auto comZ = 0.0;

        const auto nAtoms    = atomsPerMolPtr[i];
        const auto molOffset = molOffsetPtr[i];
        const auto posAtomX  = posPtr[molOffset * 3];
        const auto posAtomY  = posPtr[molOffset * 3 + 1];
        const auto posAtomZ  = posPtr[molOffset * 3 + 2];

        for (size_t j = 0; j < nAtoms; ++j)
        {
            const auto atomIndex = molOffset + j;
            const auto mass      = massesPtr[atomIndex];
            const auto posX      = posPtr[atomIndex * 3];
            const auto posY      = posPtr[atomIndex * 3 + 1];
            const auto posZ      = posPtr[atomIndex * 3 + 2];

            auto dx = posX - posAtomX;
            auto dy = posY - posAtomY;
            auto dz = posZ - posAtomZ;

            auto tx = 0.0;
            auto ty = 0.0;
            auto tz = 0.0;

            if (isOrthorhombic)
                imageOrthoRhombic(boxParams, dx, dy, dz, tx, ty, tz);
            else
                imageTriclinic(boxParams, dx, dy, dz, tx, ty, tz);

            comX += mass * (posX + tx);
            comY += mass * (posY + ty);
            comZ += mass * (posZ + tz);
        }

        const auto molMass = molMassesPtr[i];

        comX /= molMass;
        comY /= molMass;
        comZ /= molMass;

        if (isOrthorhombic)
            imageOrthoRhombic(boxParams, comX, comY, comZ);
        else
            imageTriclinic(boxParams, comX, comY, comZ);

        comMolecules[i * 3]     = comX;
        comMolecules[i * 3 + 1] = comY;
        comMolecules[i * 3 + 2] = comZ;
    }

#ifdef __PQ_LEGACY__
    simBox.deFlattenComMolecules();
#endif

    __MOL_MASSES_MIN_MAX_SUM_MEAN__(simBox);
    __COM_MIN_MAX_SUM_MEAN__(simBox);
    __DEBUG_EXIT_FUNCTION__("COM of molecules calculation");
}

/**
 * @brief calculate mol masses of simulationBox
 *
 * @param simBox
 */
void simulationBox::calculateMolMasses(SimulationBox& simBox)
{
    const auto nMolecules = simBox.getNumberOfMolecules();

    const auto* const massesPtr      = simBox.getMassesPtr();
    const auto* const atomsPerMolPtr = simBox.getAtomsPerMoleculePtr();
    const auto* const molOffsetPtr   = simBox.getMoleculeOffsetsPtr();

    auto* const molMasses = simBox.getMolMassesPtr();

    // clang-format off
#ifdef __PQ_GPU__
    #pragma omp target teams distribute parallel for     \
                is_device_ptr(massesPtr, atomsPerMolPtr, \
                    molOffsetPtr, molMasses)
#else
    #pragma omp parallel for
#endif
    // clang-format on
    for (size_t i = 0; i < nMolecules; ++i)
    {
        const auto nAtoms    = atomsPerMolPtr[i];
        const auto molOffset = molOffsetPtr[i];

        Real molMass = 0.0;

        for (size_t j = 0; j < nAtoms; ++j)
            molMass += massesPtr[molOffset + j];

        molMasses[i] = molMass;
    }
}
