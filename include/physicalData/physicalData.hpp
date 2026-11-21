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

#ifndef PHYSICAL_DATA__PHYSICAL_DATA_HPP
#define PHYSICAL_DATA__PHYSICAL_DATA_HPP

#include <memory>
#include <mstd/utils/members.hpp>
#include <vector>

#include "enums/virial.hpp"
#include "unit_types.hpp"

namespace molsys
{
    class SimulationBox;   // forward declaration

}   // namespace molsys

namespace physicalData
{
    class PhysicalData;   // forward declaration

    PhysicalData mean(std::vector<PhysicalData>& physicalDataVector);

    /**
     * @struct KineticEnergyVirialTensor
     *
     * @brief KineticEnergyVirialTensor is a struct for storing kinetic energy
     * and virial tensors
     *
     */
    struct KineticEnergyVirialTensor
    {
        VirialPerMol atomic    = EnergyPerMol::zero();
        VirialPerMol molecular = EnergyPerMol::zero();

        [[nodiscard]]
        const VirialPerMol& getVirialTensor(VirialType virialType) const;
    };

    /**
     * @class PhysicalData
     *
     * @brief PhysicalData is a class for output data storage
     *
     */
    class PhysicalData
    {
       private:
        double _nQMAtoms = 0.0;
        Time   _loopTime = Time::zero();

        Volume      _volume  = Volume::zero();
        Density     _density = Density::zero();
        Temperature _temperature{Temperature::zero()};

        Pressure _pressure{Pressure::zero()};
        Pressure _coupledPressure{Pressure::zero()};

        EnergyPerMol _kineticEnergy         = EnergyPerMol::zero();
        EnergyPerMol _coulombEnergy         = EnergyPerMol::zero();
        EnergyPerMol _nonCoulombEnergy      = EnergyPerMol::zero();
        EnergyPerMol _intraCoulombEnergy    = EnergyPerMol::zero();
        EnergyPerMol _intraNonCoulombEnergy = EnergyPerMol::zero();

        EnergyPerMol _bondEnergy     = EnergyPerMol::zero();
        EnergyPerMol _angleEnergy    = EnergyPerMol::zero();
        EnergyPerMol _dihedralEnergy = EnergyPerMol::zero();
        EnergyPerMol _improperEnergy = EnergyPerMol::zero();

        EnergyPerMol _qmEnergy          = EnergyPerMol::zero();
        EnergyPerMol _qmReferenceEnergy = EnergyPerMol::zero();

        double _numberOfSmoothingMol = 0.0;

        EnergyPerMol _noseHooverMomentumEnergy = EnergyPerMol::zero();
        Quantity     _noseHooverFrictionEnergy = Quantity::zero();

        EnergyPerMol _lowerDistanceConstraints = EnergyPerMol::zero();
        EnergyPerMol _upperDistanceConstraints = EnergyPerMol::zero();

        // Vec3D and tensor3D do not initialise their components, a default
        // constructed PhysicalData (also the one that resets the averages after
        // every output) would otherwise write garbage, sometimes inf, into the
        // first energy and momentum line
        Momentum3D        _momentum        = Momentum::zero();
        AngularMomentum3D _angularMomentum = AngularMomentum::zero();

        KineticEnergyVirialTensor _kinEnergyVirialTensor;

        VirialPerMol _virial       = VirialPerMol::zero();
        Stress       _stressTensor = Stress::zero();

        EnergyPerMol _ringPolymerEnergy = EnergyPerMol::zero();

       public:
        void calculateTemperature(molsys::SimulationBox&);
        void calculateKinetics(molsys::SimulationBox&);

        [[nodiscard]] std::shared_ptr<PhysicalData> clone() const;

        void copy(const PhysicalData&);
        void updateAverages(const PhysicalData&);
        void makeAverages(double);
        void reset();
        void resetEnergies();

        void addIntraCoulombEnergy(const EnergyPerMol& intraCoulombEnergy);
        void addIntraNonCoulombEnergy(
            const EnergyPerMol& intraNonCoulombEnergy
        );

        [[nodiscard]] EnergyPerMol getTotalEnergy() const;
        [[nodiscard]] EnergyPerMol getPotentialEnergy() const;

        /*************************
         * standard add methods  *
         *************************/

        void addVirial(const VirialPerMol& virial);
        void addQMEnergy(const EnergyPerMol& qmEnergy);
        void addQMReferenceEnergy(const EnergyPerMol& qmReferenceEnergy);
        void addCoulombEnergy(const EnergyPerMol& coulombEnergy);
        void addNonCoulombEnergy(const EnergyPerMol& nonCoulombEnergy);

        void addBondEnergy(const EnergyPerMol& bondEnergy);
        void addAngleEnergy(const EnergyPerMol& angleEnergy);
        void addDihedralEnergy(const EnergyPerMol& dihedralEnergy);
        void addImproperEnergy(const EnergyPerMol& improperEnergy);

        void addRingPolymerEnergy(const EnergyPerMol& ringPolymerEnergy);
        void addLowerDistanceConstraints(
            const EnergyPerMol& lowerDistanceConstraints
        );
        void addUpperDistanceConstraints(
            const EnergyPerMol& upperDistanceConstraints
        );

        /****************************
         * auto getters and setters *
         ****************************/

        CONST_GETTER_SETTER(_nQMAtoms);
        CONST_GETTER_SETTER(_loopTime);

        CONST_GETTER_SETTER(_volume);
        CONST_GETTER_SETTER(_density);
        CONST_GETTER_SETTER(_temperature);
        CONST_GETTER_SETTER(_pressure);
        CONST_GETTER_SETTER(_coupledPressure);

        CONST_GETTER_SETTER(_virial);
        CONST_GETTER_SETTER(_stressTensor);

        CONST_GETTER_SETTER(_momentum);
        CONST_GETTER_SETTER(_angularMomentum);

        CONST_GETTER_SETTER(_kineticEnergy);

        CONST_GETTER_SETTER(_coulombEnergy);
        CONST_GETTER_SETTER(_nonCoulombEnergy);
        CONST_GETTER_SETTER(_intraCoulombEnergy);
        CONST_GETTER_SETTER(_intraNonCoulombEnergy);

        CONST_GETTER_SETTER(_bondEnergy);
        CONST_GETTER_SETTER(_angleEnergy);
        CONST_GETTER_SETTER(_dihedralEnergy);
        CONST_GETTER_SETTER(_improperEnergy);

        CONST_GETTER_SETTER(_qmEnergy);
        CONST_GETTER_SETTER(_qmReferenceEnergy);

        // special stuff
        CONST_GETTER_SETTER(_noseHooverMomentumEnergy);
        CONST_GETTER_SETTER(_noseHooverFrictionEnergy);
        CONST_GETTER_SETTER(_lowerDistanceConstraints);
        CONST_GETTER_SETTER(_upperDistanceConstraints);
        CONST_GETTER_SETTER(_ringPolymerEnergy);
        CONST_GETTER_SETTER(_numberOfSmoothingMol);

        /********************
         * standard setters *
         ********************/

        void setKineticEnergyAtomicVector(const VirialPerMol& vec);
        void setKineticEnergyMolecularVector(const VirialPerMol& vec);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] EnergyPerMol getIntraEnergy() const;

        [[nodiscard]] const VirialPerMol& getKinEnergyAtomTensor() const;
        [[nodiscard]] const VirialPerMol& getKinEnergyMolTensor() const;

        [[nodiscard]]
        const VirialPerMol& getKinEnergyVirialTensor(
            VirialType virialType
        ) const;
    };

}   // namespace physicalData

#endif   // PHYSICAL_DATA__PHYSICAL_DATA_HPP
