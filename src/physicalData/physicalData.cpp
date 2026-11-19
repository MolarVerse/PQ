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

#include <algorithm>
#include <utility>

#include "globalTimer.hpp"
#include "simulationBox.hpp"
#include "unit_types.hpp"

namespace physicalData
{
    /**
     * @brief get the virial tensor, either atomic or molecular depending on the
     * configured virial type
     *
     * @param virialType - the virial type to get the virial tensor for
     *
     * @return const linalg::tensor3D&
     */
    const VirialPerMol& KineticEnergyVirialTensor::getVirialTensor(
        VirialType virialType
    ) const
    {
        switch (virialType)
        {
            case VirialType::ATOMIC: return atomic;
            case VirialType::MOLECULAR: return molecular;
        }

        std::unreachable();
    }

    /**
     * @brief Calculates kinetic energy and momentum of the system
     *
     * @param simulationBox
     */
    void PhysicalData::calculateKinetics(molsys::SimulationBox& simulationBox)
    {
        auto _ = scopedTimer(TimerId::PhysicalData, "Calc Kinetics");

        _momentum = Momentum::zero();
        MassVelSquaredTensor3D kineticEnergyAtomicTensor =
            MassVelSquared::zero();
        MassVelSquaredTensor3D kineticEnergyMolecularTensor =
            MassVelSquared::zero();

        auto kinEnergyAndMomOfMol = [&kineticEnergyAtomicTensor,
                                     &kineticEnergyMolecularTensor,
                                     this](auto& molecule)
        {
            const auto numberOfAtoms     = molecule.getNumberOfAtoms();
            Momentum3D molecularMomentum = Momentum::zero();

            for (AtomIndex i{0}; i.get() < numberOfAtoms; ++i)
            {
                const auto& velocities = molecule.getAtomVelocity(i);

                const auto momentum = velocities * molecule.getAtomMass(i);

                _momentum         += momentum;
                molecularMomentum += momentum;

                kineticEnergyAtomicTensor +=
                    tensorProduct(momentum, velocities);
            }

            kineticEnergyMolecularTensor +=
                tensorProduct(molecularMomentum, molecularMomentum) /
                molecule.getMolMass();
        };

        std::ranges::for_each(
            simulationBox.getMolecules(),
            kinEnergyAndMomOfMol
        );

        const auto atomicEnergyTensor    = kineticEnergyAtomicTensor / 2.0;
        const auto molecularEnergyTensor = kineticEnergyMolecularTensor / 2.0;

        _kinEnergyVirialTensor.atomic    = atomicEnergyTensor;
        _kinEnergyVirialTensor.molecular = molecularEnergyTensor;

        _kineticEnergy = trace(atomicEnergyTensor);

        _angularMomentum = simulationBox.calculateAngularMomentum(_momentum);
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
     * @param other - physicalData to copy from
     */
    void PhysicalData::copy(const PhysicalData& other)
    {
        reset();

        updateAverages(other);
    }

    /**
     * @brief calculates the sum of all physicalData of last steps
     *
     * @param other - physicalData to update averages from
     */
    void PhysicalData::updateAverages(const PhysicalData& other)
    {
        _numberOfQMAtoms += other.getNumberOfQMAtoms();
        _loopTime        += other.getLoopTime();

        _coulombEnergy         += other.getCoulombEnergy();
        _nonCoulombEnergy      += other.getNonCoulombEnergy();
        _intraCoulombEnergy    += other.getIntraCoulombEnergy();
        _intraNonCoulombEnergy += other.getIntraNonCoulombEnergy();

        _bondEnergy     += other.getBondEnergy();
        _angleEnergy    += other.getAngleEnergy();
        _dihedralEnergy += other.getDihedralEnergy();
        _improperEnergy += other.getImproperEnergy();

        _temperature     += other.getTemperature().quantity_from_zero();
        _kineticEnergy   += other.getKineticEnergy();
        _volume          += other.getVolume();
        _density         += other.getDensity();
        _virial          += other.getVirial();
        _pressure        += other.getPressure();
        _coupledPressure += other.getCoupledPressure();

        _qmEnergy          += other.getQMEnergy();
        _qmReferenceEnergy += other.getQMReferenceEnergy();

        _numberOfSmoothingMol += other.getNumberOfSmoothingMolecules();

        _momentum        += other.getMomentum();
        _angularMomentum += other.getAngularMomentum();

        _noseHooverMomentumEnergy += other.getNoseHooverMomentumEnergy();
        _noseHooverFrictionEnergy += other.getNoseHooverFrictionEnergy();

        _lowerDistanceConstraints += other.getLowerDistanceConstraints();
        _upperDistanceConstraints += other.getUpperDistanceConstraints();

        _ringPolymerEnergy += other.getRingPolymerEnergy();

        const auto& kinEnergyVirialTensor = other._kinEnergyVirialTensor;

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

        _temperature =
            Temperature{_temperature.quantity_from_zero() / outputFrequency};
        _volume          /= outputFrequency;
        _density         /= outputFrequency;
        _virial          /= outputFrequency;
        _pressure        /= outputFrequency;
        _coupledPressure /= outputFrequency;

        _qmEnergy          /= outputFrequency;
        _qmReferenceEnergy /= outputFrequency;

        _numberOfSmoothingMol /= outputFrequency;

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
        _loopTime        = Time::zero();

        _volume          = Volume::zero();
        _density         = Density::zero();
        _temperature     = Temperature::zero();
        _pressure        = Pressure::zero();
        _coupledPressure = Pressure::zero();

        _kineticEnergy         = EnergyPerMol::zero();
        _coulombEnergy         = EnergyPerMol::zero();
        _nonCoulombEnergy      = EnergyPerMol::zero();
        _intraCoulombEnergy    = EnergyPerMol::zero();
        _intraNonCoulombEnergy = EnergyPerMol::zero();

        _bondEnergy     = EnergyPerMol::zero();
        _angleEnergy    = EnergyPerMol::zero();
        _dihedralEnergy = EnergyPerMol::zero();
        _improperEnergy = EnergyPerMol::zero();

        _qmEnergy          = EnergyPerMol::zero();
        _qmReferenceEnergy = EnergyPerMol::zero();

        _numberOfSmoothingMol = 0.0;

        _momentum        = Momentum::zero();
        _angularMomentum = AngularMomentum::zero();

        _noseHooverMomentumEnergy = EnergyPerMol::zero();
        _noseHooverFrictionEnergy = 0.0;

        _lowerDistanceConstraints = EnergyPerMol::zero();
        _upperDistanceConstraints = EnergyPerMol::zero();

        _virial       = VirialPerMol::zero();
        _stressTensor = Stress::zero();

        _ringPolymerEnergy = EnergyPerMol::zero();

        // reset kinetic energy virial tensor, but make sure to keep the
        // isVirialAtomic flag as it is
        _kinEnergyVirialTensor.atomic    = EnergyPerMol::zero();
        _kinEnergyVirialTensor.molecular = EnergyPerMol::zero();
    }

    /**
     * @brief Clear all energies in PhysicalData. Used in QM/MM exact smoothing.
     *
     */
    void PhysicalData::resetEnergies()
    {
        _kineticEnergy         = EnergyPerMol::zero();
        _coulombEnergy         = EnergyPerMol::zero();
        _nonCoulombEnergy      = EnergyPerMol::zero();
        _intraCoulombEnergy    = EnergyPerMol::zero();
        _intraNonCoulombEnergy = EnergyPerMol::zero();

        _bondEnergy     = EnergyPerMol::zero();
        _angleEnergy    = EnergyPerMol::zero();
        _dihedralEnergy = EnergyPerMol::zero();
        _improperEnergy = EnergyPerMol::zero();

        _qmEnergy          = EnergyPerMol::zero();
        _qmReferenceEnergy = EnergyPerMol::zero();

        _kinEnergyVirialTensor.atomic    = EnergyPerMol::zero();
        _kinEnergyVirialTensor.molecular = EnergyPerMol::zero();
    }

    /**
     * @brief calculate temperature
     *
     * @param simulationBox
     */
    void PhysicalData::calculateTemperature(
        molsys::SimulationBox& simulationBox
    )
    {
        _temperature = simulationBox.calculateTemperature();
    }

    /**
     * @brief calculate total energy
     *
     * @return EnergyPerMol
     */
    EnergyPerMol PhysicalData::getTotalEnergy() const
    {
        auto totalEnergy = EnergyPerMol::zero();

        totalEnergy += _bondEnergy;
        totalEnergy += _angleEnergy;
        totalEnergy += _dihedralEnergy;
        totalEnergy += _improperEnergy;

        totalEnergy += _coulombEnergy;      // intra + inter
        totalEnergy += _nonCoulombEnergy;   // intra + inter

        totalEnergy += _kineticEnergy;

        totalEnergy += _qmEnergy;
        totalEnergy -= _qmReferenceEnergy;

        return totalEnergy;
    }

    /**
     * @brief add intra coulomb energy
     *
     * @details This function is used to add intra coulomb energy to the total
     * coulomb energy
     *
     * @param intraCoulombEnergy
     */
    void PhysicalData::addIntraCoulombEnergy(
        const EnergyPerMol& intraCoulombEnergy
    )
    {
        _intraCoulombEnergy += intraCoulombEnergy;
        _coulombEnergy      += intraCoulombEnergy;
    }

    /**
     * @brief add intra non coulomb energy
     *
     * @details This function is used to add intra non coulomb energy to the
     * total non coulomb energy
     *
     * @param intraNonCoulombEnergy
     */
    void PhysicalData::addIntraNonCoulombEnergy(
        const EnergyPerMol& intraNonCoulombEnergy
    )
    {
        _intraNonCoulombEnergy += intraNonCoulombEnergy;
        _nonCoulombEnergy      += intraNonCoulombEnergy;
    }

    /**
     * @brief calculate the mean of a vector of physicalData
     *
     * @param physicalDataVector - vector of physicalData
     * @return PhysicalData
     */
    PhysicalData mean(std::vector<PhysicalData>& physicalDataVector)
    {
        PhysicalData meanData;

        std::ranges::for_each(
            physicalDataVector,
            [&meanData](auto& physicalData)
            { meanData.updateAverages(physicalData); }
        );

        meanData.makeAverages(static_cast<double>(physicalDataVector.size()));

        return meanData;
    }

}   // namespace physicalData
