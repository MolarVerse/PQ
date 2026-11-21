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
        double _numberOfQMAtoms = 0.0;
        Time   _loopTime        = Time::zero();

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

        /********************
         * standard setters *
         ********************/

        void setNumberOfQMAtoms(double nQMAtoms);
        void setLoopTime(const Time& loopTime);

        void setVolume(const Volume& volume);
        void setDensity(const Density& density);
        void setTemperature(const Temperature& temperature);
        void setPressure(const Pressure& pressure);
        void setCoupledPressure(const Pressure& coupledPressure);

        void setVirial(const VirialPerMol& virial);
        void setStressTensor(const Stress& stressTensor);

        void setMomentum(const Momentum3D& momentum);
        void setAngularMomentum(const AngularMomentum3D& angularMomentum);

        void setKineticEnergy(const EnergyPerMol& kineticEnergy);
        void setKineticEnergyAtomicVector(const VirialPerMol& vec);
        void setKineticEnergyMolecularVector(const VirialPerMol& vec);

        void setCoulombEnergy(const EnergyPerMol& coulombEnergy);
        void setNonCoulombEnergy(const EnergyPerMol& nonCoulombEnergy);
        void setIntraCoulombEnergy(const EnergyPerMol& intraCoulombEnergy);
        void setIntraNonCoulombEnergy(
            const EnergyPerMol& intraNonCoulombEnergy
        );

        void setBondEnergy(const EnergyPerMol& bondEnergy);
        void setAngleEnergy(const EnergyPerMol& angleEnergy);
        void setDihedralEnergy(const EnergyPerMol& dihedralEnergy);
        void setImproperEnergy(const EnergyPerMol& improperEnergy);

        void setQMEnergy(const EnergyPerMol& qmEnergy);
        void setQMReferenceEnergy(const EnergyPerMol& qmReferenceEnergy);

        void setNumberOfSmoothingMolecules(double numberSmMol);

        void setNoseHooverMomentumEnergy(const EnergyPerMol& momentumEnergy);
        void setNoseHooverFrictionEnergy(const Quantity& frictionEnergy);

        void setLowerDistanceConstraints(
            const EnergyPerMol& lowerDistanceConstraints
        );
        void setUpperDistanceConstraints(
            const EnergyPerMol& upperDistanceConstraints
        );

        void setRingPolymerEnergy(const EnergyPerMol& ringPolymerEnergy);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] double      getNumberOfActiveAtoms() const;
        [[nodiscard]] const Time& getLoopTime() const;

        [[nodiscard]] const Volume&      getVolume() const;
        [[nodiscard]] const Density&     getDensity() const;
        [[nodiscard]] const Temperature& getTemperature() const;
        [[nodiscard]] const Pressure&    getPressure() const;
        [[nodiscard]] const Pressure&    getCoupledPressure() const;

        [[nodiscard]] const EnergyPerMol& getKineticEnergy() const;
        [[nodiscard]] const EnergyPerMol& getNonCoulombEnergy() const;
        [[nodiscard]] const EnergyPerMol& getCoulombEnergy() const;
        [[nodiscard]] const EnergyPerMol& getIntraCoulombEnergy() const;
        [[nodiscard]] const EnergyPerMol& getIntraNonCoulombEnergy() const;
        [[nodiscard]] EnergyPerMol        getIntraEnergy() const;

        [[nodiscard]] const EnergyPerMol& getBondEnergy() const;
        [[nodiscard]] const EnergyPerMol& getAngleEnergy() const;
        [[nodiscard]] const EnergyPerMol& getDihedralEnergy() const;
        [[nodiscard]] const EnergyPerMol& getImproperEnergy() const;

        [[nodiscard]] const EnergyPerMol& getQMEnergy() const;
        [[nodiscard]] const EnergyPerMol& getQMReferenceEnergy() const;

        [[nodiscard]] double getNumberOfSmoothingMolecules() const;

        [[nodiscard]] const EnergyPerMol& getNoseHooverMomentumEnergy() const;
        [[nodiscard]] const Quantity&     getNoseHooverFrictionEnergy() const;

        [[nodiscard]] const EnergyPerMol& getLowerDistanceConstraints() const;
        [[nodiscard]] const EnergyPerMol& getUpperDistanceConstraints() const;

        [[nodiscard]] const EnergyPerMol& getRingPolymerEnergy() const;

        [[nodiscard]] const VirialPerMol& getKinEnergyAtomTensor() const;
        [[nodiscard]] const VirialPerMol& getKinEnergyMolTensor() const;

        [[nodiscard]]
        const VirialPerMol& getKinEnergyVirialTensor(
            VirialType virialType
        ) const;

        [[nodiscard]] const VirialPerMol& getVirial() const;
        [[nodiscard]] const Stress&       getStressTensor() const;

        [[nodiscard]] const Momentum3D&        getMomentum() const;
        [[nodiscard]] const AngularMomentum3D& getAngularMomentum() const;
    };

}   // namespace physicalData

#endif   // PHYSICAL_DATA__PHYSICAL_DATA_HPP
