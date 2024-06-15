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

#include <functional>   // for _Bind_front_t, bind_front, function
#include <vector>       // for vector

#include "staticMatrix3x3.hpp"   // for StaticMatrix3x3
#include "timer.hpp"             // for Timer
#include "vector3d.hpp"          // for Vec3D

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace physicalData
{
    class PhysicalData;   // forward declaration

    PhysicalData mean(std::vector<PhysicalData> &physicalDataVector);

    /**
     * @class PhysicalData
     *
     * @brief PhysicalData is a class for output data storage
     *
     */
    class PhysicalData : public timings::Timer
    {
       private:
        double _numberOfQMAtoms = 0.0;
        double _loopTime        = 0.0;

        double _volume      = 0.0;
        double _density     = 0.0;
        double _temperature = 0.0;
        double _pressure    = 0.0;

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

        double _noseHooverMomentumEnergy = 0.0;
        double _noseHooverFrictionEnergy = 0.0;

        double _lowerDistanceConstraints = 0.0;
        double _upperDistanceConstraints = 0.0;

        linearAlgebra::Vec3D    _momentum                     = {0.0, 0.0, 0.0};
        linearAlgebra::Vec3D    _angularMomentum              = {0.0, 0.0, 0.0};
        linearAlgebra::tensor3D _kineticEnergyAtomicTensor    = {0.0};
        linearAlgebra::tensor3D _kineticEnergyMolecularTensor = {0.0};

        linearAlgebra::tensor3D _virial       = {0.0};
        linearAlgebra::tensor3D _stressTensor = {0.0};

        double _ringPolymerEnergy = 0.0;

       public:
        void calculateTemperature(simulationBox::SimulationBox &);
        void calculateKinetics(simulationBox::SimulationBox &);
        void changeKineticVirialToAtomic();

        std::function<linearAlgebra::StaticMatrix3x3<double>()>
            getKineticEnergyVirialVector = std::bind_front(
                &PhysicalData::getKineticEnergyMolecularVector,
                this
            );

        void copy(const PhysicalData &);
        void updateAverages(const PhysicalData &);
        void makeAverages(const double);
        void reset();

        void addIntraCoulombEnergy(const double intraCoulombEnergy);
        void addIntraNonCoulombEnergy(const double intraNonCoulombEnergy);

        [[nodiscard]] double getTotalEnergy() const;

        /*************************
         * standard add methods  *
         *************************/

        void addVirial(const linearAlgebra::tensor3D &virial);
        void addCoulombEnergy(const double coulombEnergy);
        void addNonCoulombEnergy(const double nonCoulombEnergy);

        void addBondEnergy(const double bondEnergy);
        void addAngleEnergy(const double angleEnergy);
        void addDihedralEnergy(const double dihedralEnergy);
        void addImproperEnergy(const double improperEnergy);

        void addRingPolymerEnergy(const double ringPolymerEnergy);

        /********************
         * standard setters *
         ********************/

        void setNumberOfQMAtoms(const double nQMAtoms);
        void setLoopTime(const double loopTime);

        void setVolume(const double volume);
        void setDensity(const double density);
        void setTemperature(const double temperature);
        void setPressure(const double pressure);

        void setVirial(const linearAlgebra::tensor3D &virial);
        void setStressTensor(const linearAlgebra::tensor3D &stressTensor);

        void setMomentum(const linearAlgebra::Vec3D &vec);
        void setAngularMomentum(const linearAlgebra::Vec3D &vec);

        void setKineticEnergy(const double kineticEnergy);
        void setKineticEnergyAtomicVector(const linearAlgebra::tensor3D &vec);
        void setKineticEnergyMolecularVector(const linearAlgebra::tensor3D &vec
        );

        void setCoulombEnergy(const double coulombEnergy);
        void setNonCoulombEnergy(const double nonCoulombEnergy);
        void setIntraCoulombEnergy(const double intraCoulombEnergy);
        void setIntraNonCoulombEnergy(const double intraNonCoulombEnergy);

        void setBondEnergy(const double bondEnergy);
        void setAngleEnergy(const double angleEnergy);
        void setDihedralEnergy(const double dihedralEnergy);
        void setImproperEnergy(const double improperEnergy);

        void setQMEnergy(const double qmEnergy);

        void setNoseHooverMomentumEnergy(const double momentumEnergy);
        void setNoseHooverFrictionEnergy(const double frictionEnergy);

        void setLowerDistanceConstraints(const double lowerDistanceConstraints);
        void setUpperDistanceConstraints(const double upperDistanceConstraints);

        void setRingPolymerEnergy(const double ringPolymerEnergy);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] double getNumberOfQMAtoms() const;
        [[nodiscard]] double getLoopTime() const;

        [[nodiscard]] double getVolume() const;
        [[nodiscard]] double getDensity() const;
        [[nodiscard]] double getTemperature() const;
        [[nodiscard]] double getPressure() const;

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

        [[nodiscard]] double getNoseHooverMomentumEnergy() const;
        [[nodiscard]] double getNoseHooverFrictionEnergy() const;

        [[nodiscard]] double getLowerDistanceConstraints() const;
        [[nodiscard]] double getUpperDistanceConstraints() const;

        [[nodiscard]] double getRingPolymerEnergy() const;

        [[nodiscard]] linearAlgebra::tensor3D getKineticEnergyAtomicVector(
        ) const;
        [[nodiscard]] linearAlgebra::tensor3D getKineticEnergyMolecularVector(
        ) const;
        [[nodiscard]] linearAlgebra::tensor3D getVirial() const;
        [[nodiscard]] linearAlgebra::tensor3D getStressTensor() const;

        [[nodiscard]] linearAlgebra::Vec3D getMomentum() const;
        [[nodiscard]] linearAlgebra::Vec3D getAngularMomentum() const;
    };

}   // namespace physicalData

#endif   // _PHYSICAL_DATA_HPP_
