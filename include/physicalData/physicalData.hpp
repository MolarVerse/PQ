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

#ifndef _PHYSICAL_DATA_HPP_

#define _PHYSICAL_DATA_HPP_

#include <memory>
#include <vector>   // for vector

#include "settings.hpp"
#include "staticMatrix.hpp"

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
        linalg::tensor3D atomic;
        linalg::tensor3D molecular;

        [[nodiscard]]
        const linalg::tensor3D& getVirialTensor(
            settings::VirialType virialType
        ) const;
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
        double _loopTime        = 0.0;

        double _volume          = 0.0;
        double _density         = 0.0;
        double _temperature     = 0.0;
        double _pressure        = 0.0;
        double _coupledPressure = 0.0;

        double _kineticEnergy         = 0.0;
        double _coulombEnergy         = 0.0;
        double _nonCoulombEnergy      = 0.0;
        double _intraCoulombEnergy    = 0.0;
        double _intraNonCoulombEnergy = 0.0;

        double _bondEnergy     = 0.0;
        double _angleEnergy    = 0.0;
        double _dihedralEnergy = 0.0;
        double _improperEnergy = 0.0;

        double _qmEnergy = 0.0;

        double _numberOfSmoothingMol = 0.0;

        double _noseHooverMomentumEnergy = 0.0;
        double _noseHooverFrictionEnergy = 0.0;

        double _lowerDistanceConstraints = 0.0;
        double _upperDistanceConstraints = 0.0;

        linalg::Vec3D _momentum;
        linalg::Vec3D _angularMomentum;

        KineticEnergyVirialTensor _kinEnergyVirialTensor;

        linalg::tensor3D _virial;
        linalg::tensor3D _stressTensor;

        double _ringPolymerEnergy = 0.0;

       public:
        void calculateTemperature(molsys::SimulationBox&);
        void calculateKinetics(molsys::SimulationBox&);

        [[nodiscard]] std::shared_ptr<PhysicalData> clone() const;

        void copy(const PhysicalData&);
        void updateAverages(const PhysicalData&);
        void makeAverages(double);
        void reset();
        void resetEnergies();

        void addIntraCoulombEnergy(double intraCoulombEnergy);
        void addIntraNonCoulombEnergy(double intraNonCoulombEnergy);

        [[nodiscard]] double getTotalEnergy() const;

        /*************************
         * standard add methods  *
         *************************/

        void addVirial(const linalg::tensor3D& virial);
        void addQMEnergy(double qmEnergy);
        void addCoulombEnergy(double coulombEnergy);
        void addNonCoulombEnergy(double nonCoulombEnergy);

        void addBondEnergy(double bondEnergy);
        void addAngleEnergy(double angleEnergy);
        void addDihedralEnergy(double dihedralEnergy);
        void addImproperEnergy(double improperEnergy);

        void addRingPolymerEnergy(double ringPolymerEnergy);

        /********************
         * standard setters *
         ********************/

        void setNumberOfQMAtoms(double nQMAtoms);
        void setLoopTime(double loopTime);

        void setVolume(double volume);
        void setDensity(double density);
        void setTemperature(double temperature);
        void setPressure(double pressure);
        void setCoupledPressure(double coupledPressure);

        void setVirial(const linalg::tensor3D& virial);
        void setStressTensor(const linalg::tensor3D& stressTensor);

        void setMomentum(const linalg::Vec3D& momentum);
        void setAngularMomentum(const linalg::Vec3D& angularMomentum);

        void setKineticEnergy(double kineticEnergy);
        void setKineticEnergyAtomicVector(const linalg::tensor3D& vec);
        void setKineticEnergyMolecularVector(const linalg::tensor3D& vec);

        void setCoulombEnergy(double coulombEnergy);
        void setNonCoulombEnergy(double nonCoulombEnergy);
        void setIntraCoulombEnergy(double intraCoulombEnergy);
        void setIntraNonCoulombEnergy(double intraNonCoulombEnergy);

        void setBondEnergy(double bondEnergy);
        void setAngleEnergy(double angleEnergy);
        void setDihedralEnergy(double dihedralEnergy);
        void setImproperEnergy(double improperEnergy);

        void setQMEnergy(double qmEnergy);

        void setNumberOfSmoothingMolecules(double numberSmMol);

        void setNoseHooverMomentumEnergy(double momentumEnergy);
        void setNoseHooverFrictionEnergy(double frictionEnergy);

        void setLowerDistanceConstraints(double lowerDistanceConstraints);
        void setUpperDistanceConstraints(double upperDistanceConstraints);

        void setRingPolymerEnergy(double ringPolymerEnergy);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] double getNumberOfQMAtoms() const;
        [[nodiscard]] double getLoopTime() const;

        [[nodiscard]] double getVolume() const;
        [[nodiscard]] double getDensity() const;
        [[nodiscard]] double getTemperature() const;
        [[nodiscard]] double getPressure() const;
        [[nodiscard]] double getCoupledPressure() const;

        [[nodiscard]] double getKineticEnergy() const;
        [[nodiscard]] double getNonCoulombEnergy() const;
        [[nodiscard]] double getCoulombEnergy() const;
        [[nodiscard]] double getIntraCoulombEnergy() const;
        [[nodiscard]] double getIntraNonCoulombEnergy() const;
        [[nodiscard]] double getIntraEnergy() const;

        [[nodiscard]] double getBondEnergy() const;
        [[nodiscard]] double getAngleEnergy() const;
        [[nodiscard]] double getDihedralEnergy() const;
        [[nodiscard]] double getImproperEnergy() const;

        [[nodiscard]] double getQMEnergy() const;

        [[nodiscard]] double getNumberOfSmoothingMolecules() const;

        [[nodiscard]] double getNoseHooverMomentumEnergy() const;
        [[nodiscard]] double getNoseHooverFrictionEnergy() const;

        [[nodiscard]] double getLowerDistanceConstraints() const;
        [[nodiscard]] double getUpperDistanceConstraints() const;

        [[nodiscard]] double getRingPolymerEnergy() const;

        [[nodiscard]] linalg::tensor3D getKinEnergyAtomTensor() const;
        [[nodiscard]] linalg::tensor3D getKinEnergyMolTensor() const;

        [[nodiscard]]
        const linalg::tensor3D& getKinEnergyVirialTensor(
            settings::VirialType virialType
        ) const;

        [[nodiscard]] linalg::tensor3D getVirial() const;
        [[nodiscard]] linalg::tensor3D getStressTensor() const;

        [[nodiscard]] linalg::Vec3D getMomentum() const;
        [[nodiscard]] linalg::Vec3D getAngularMomentum() const;
    };

}   // namespace physicalData

#endif   // _PHYSICAL_DATA_HPP_
