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

#include "physicalData.hpp"

#include <algorithm>   // for __for_each_fn
#include <cstddef>     // for size_t
#include <utility>

#include "constants/conversionFactors.hpp"           // for _FS_TO_S_
#include "constants/internalConversionFactors.hpp"   // for _KINETIC_ENERGY_FACTOR_
#include "simulationBox.hpp"                         // for SimulationBox

using namespace physicalData;
using namespace simulationBox;
using namespace linearAlgebra;
using namespace constants;

/**
 * @brief get the virial tensor, either atomic or molecular depending on the
 * configured virial type
 *
 * @param virialType - the virial type to get the virial tensor for
 *
 * @return const linearAlgebra::tensor3D&
 */
const linearAlgebra::tensor3D& KineticEnergyVirialTensor::getVirialTensor(
    settings::VirialType virialType
) const
{
    switch (virialType)
    {
        case settings::VirialType::ATOMIC: return atomic;
        case settings::VirialType::MOLECULAR: return molecular;
    }

    std::unreachable();
}

/**
 * @brief Calculates kinetic energy and momentum of the system
 *
 * @param simulationBox
 */
void PhysicalData::calculateKinetics(SimulationBox& simulationBox)
{
    auto _ = scoped("Calc Kinetics");

    _momentum = Vec3D();
    tensor3D kineticEnergyAtomicTensor{};
    tensor3D kineticEnergyMolecularTensor{};

    auto kinEnergyAndMomOfMol = [&kineticEnergyAtomicTensor,
                                 &kineticEnergyMolecularTensor,
                                 this](auto& molecule)
    {
        const auto numberOfAtoms   = molecule.getNumberOfAtoms();
        auto       momentumSquared = tensor3D();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            const auto velocities = molecule.getAtomVelocity(i);

            const auto momentum = velocities * molecule.getAtomMass(i);

            _momentum                 += momentum;
            kineticEnergyAtomicTensor += tensorProduct(momentum, velocities);
            momentumSquared           += tensorProduct(momentum, momentum);
        }

        kineticEnergyMolecularTensor += momentumSquared / molecule.getMolMass();
    };

    std::ranges::for_each(simulationBox.getMolecules(), kinEnergyAndMomOfMol);

    kineticEnergyAtomicTensor    *= KINETIC_ENERGY_FACTOR;
    kineticEnergyMolecularTensor *= KINETIC_ENERGY_FACTOR;

    _kinEnergyVirialTensor.atomic    = kineticEnergyAtomicTensor;
    _kinEnergyVirialTensor.molecular = kineticEnergyMolecularTensor;

    _kineticEnergy = trace(kineticEnergyAtomicTensor);

    _angularMomentum  = simulationBox.calculateAngularMomentum(_momentum);
    _angularMomentum *= FS_TO_S;

    _momentum *= FS_TO_S;
}

/**
 * @brief clones the physicalData
 *
 * @return std::shared_ptr<PhysicalData>
 */
std::shared_ptr<PhysicalData> PhysicalData::clone() const
{
    return std::make_shared<PhysicalData>(*this);
}

/**
 * @brief copies one physicalData to another
 *
 * @param data - physicalData to copy from
 */
void PhysicalData::copy(const PhysicalData& data)
{
    reset();

    updateAverages(data);
}

/**
 * @brief calculates the sum of all physicalData of last steps
 *
 * @param physicalData
 */
void PhysicalData::updateAverages(const PhysicalData& physicalData)
{
    _numberOfQMAtoms += physicalData.getNumberOfQMAtoms();
    _loopTime        += physicalData.getLoopTime();

    _coulombEnergy         += physicalData.getCoulombEnergy();
    _nonCoulombEnergy      += physicalData.getNonCoulombEnergy();
    _intraCoulombEnergy    += physicalData.getIntraCoulombEnergy();
    _intraNonCoulombEnergy += physicalData.getIntraNonCoulombEnergy();

    _bondEnergy     += physicalData.getBondEnergy();
    _angleEnergy    += physicalData.getAngleEnergy();
    _dihedralEnergy += physicalData.getDihedralEnergy();
    _improperEnergy += physicalData.getImproperEnergy();

    _temperature   += physicalData.getTemperature();
    _kineticEnergy += physicalData.getKineticEnergy();
    _volume        += physicalData.getVolume();
    _density       += physicalData.getDensity();
    _virial        += physicalData.getVirial();
    _pressure      += physicalData.getPressure();

    _qmEnergy += physicalData.getQMEnergy();

    _momentum        += physicalData.getMomentum();
    _angularMomentum += physicalData.getAngularMomentum();

    _noseHooverMomentumEnergy += physicalData.getNoseHooverMomentumEnergy();
    _noseHooverFrictionEnergy += physicalData.getNoseHooverFrictionEnergy();

    _lowerDistanceConstraints += physicalData.getLowerDistanceConstraints();
    _upperDistanceConstraints += physicalData.getUpperDistanceConstraints();

    _ringPolymerEnergy += physicalData.getRingPolymerEnergy();

    const auto& kinEnergyVirialTensor = physicalData._kinEnergyVirialTensor;

    _kinEnergyVirialTensor.atomic    += kinEnergyVirialTensor.atomic;
    _kinEnergyVirialTensor.molecular += kinEnergyVirialTensor.molecular;
}

/**
 * @brief calculates the average of all physicalData of last steps
 *
 * @param outputFrequency
 */
void PhysicalData::makeAverages(const double outputFrequency)
{
    _numberOfQMAtoms /= outputFrequency;
    _loopTime        /= outputFrequency;

    _kineticEnergy         /= outputFrequency;
    _coulombEnergy         /= outputFrequency;
    _nonCoulombEnergy      /= outputFrequency;
    _intraCoulombEnergy    /= outputFrequency;
    _intraNonCoulombEnergy /= outputFrequency;

    _bondEnergy     /= outputFrequency;
    _angleEnergy    /= outputFrequency;
    _dihedralEnergy /= outputFrequency;
    _improperEnergy /= outputFrequency;

    _temperature /= outputFrequency;
    _volume      /= outputFrequency;
    _density     /= outputFrequency;
    _virial      /= outputFrequency;
    _pressure    /= outputFrequency;

    _qmEnergy /= outputFrequency;

    _momentum        /= outputFrequency;
    _angularMomentum /= outputFrequency;

    _noseHooverMomentumEnergy /= outputFrequency;
    _noseHooverFrictionEnergy /= outputFrequency;

    _lowerDistanceConstraints /= outputFrequency;
    _upperDistanceConstraints /= outputFrequency;

    _ringPolymerEnergy /= outputFrequency;

    _kinEnergyVirialTensor.atomic    /= outputFrequency;
    _kinEnergyVirialTensor.molecular /= outputFrequency;
}

/**
 * @brief clear all physicalData in order to call add functions
 *
 */
void PhysicalData::reset()
{
    _numberOfQMAtoms = 0.0;
    _loopTime        = 0.0;

    _kineticEnergy         = 0.0;
    _coulombEnergy         = 0.0;
    _nonCoulombEnergy      = 0.0;
    _intraCoulombEnergy    = 0.0;
    _intraNonCoulombEnergy = 0.0;

    _bondEnergy     = 0.0;
    _angleEnergy    = 0.0;
    _dihedralEnergy = 0.0;
    _improperEnergy = 0.0;

    _temperature = 0.0;
    _volume      = 0.0;
    _density     = 0.0;
    _pressure    = 0.0;
    _virial      = {0.0};

    _qmEnergy = 0.0;

    _momentum        = {0.0, 0.0, 0.0};
    _angularMomentum = {0.0, 0.0, 0.0};

    _noseHooverMomentumEnergy = 0.0;
    _noseHooverFrictionEnergy = 0.0;

    _lowerDistanceConstraints = 0.0;
    _upperDistanceConstraints = 0.0;

    _ringPolymerEnergy = 0.0;

    // reset kinetic energy virial tensor, but make sure to keep the
    // isVirialAtomic flag as it is
    _kinEnergyVirialTensor.atomic    = {0.0};
    _kinEnergyVirialTensor.molecular = {0.0};
}

/**
 * @brief calculate temperature
 *
 * @param simulationBox
 */
void PhysicalData::calculateTemperature(SimulationBox& simulationBox)
{
    _temperature = simulationBox.calculateTemperature();
}

/**
 * @brief calculate potential energy
 *
 * @return double
 */
double PhysicalData::getTotalEnergy() const
{
    auto potentialEnergy = 0.0;

    potentialEnergy += _bondEnergy;
    potentialEnergy += _angleEnergy;
    potentialEnergy += _dihedralEnergy;
    potentialEnergy += _improperEnergy;

    potentialEnergy += _coulombEnergy;      // intra + inter
    potentialEnergy += _nonCoulombEnergy;   // intra + inter

    potentialEnergy += _kineticEnergy;

    potentialEnergy += _qmEnergy;

    return potentialEnergy;
}

/**
 * @brief add intra coulomb energy
 *
 * @details This function is used to add intra coulomb energy to the total
 * coulomb energy
 *
 * @param intraCoulombEnergy
 */
void PhysicalData::addIntraCoulombEnergy(const double intraCoulombEnergy)
{
    _intraCoulombEnergy += intraCoulombEnergy;
    _coulombEnergy      += intraCoulombEnergy;
}

/**
 * @brief add intra non coulomb energy
 *
 * @details This function is used to add intra non coulomb energy to the total
 * non coulomb energy
 *
 * @param intraNonCoulombEnergy
 */
void PhysicalData::addIntraNonCoulombEnergy(const double intraNonCoulombEnergy)
{
    _intraNonCoulombEnergy += intraNonCoulombEnergy;
    _nonCoulombEnergy      += intraNonCoulombEnergy;
}

/**
 * @brief calculate the mean of a vector of physicalData
 *
 * @param dataVec - vector of physicalData
 * @return PhysicalData
 */
PhysicalData physicalData::mean(std::vector<PhysicalData>& dataVec)
{
    PhysicalData meanData;

    std::ranges::for_each(
        dataVec,
        [&meanData](auto& physicalData)
        { meanData.updateAverages(physicalData); }
    );

    meanData.makeAverages(static_cast<double>(dataVec.size()));

    return meanData;
}
