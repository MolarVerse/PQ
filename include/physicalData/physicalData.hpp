#ifndef _PHYSICAL_DATA_HPP_

#define _PHYSICAL_DATA_HPP_

#include "vector3d.hpp"   // for Vec3D

#include <functional>   // for _Bind_front_t, bind_front, function

namespace simulationBox
{
    class SimulationBox;
}   // namespace simulationBox

namespace physicalData
{
    /**
     * @class PhysicalData
     *
     * @brief PhysicalData is a class for output data storage
     *
     */
    class PhysicalData
    {
      private:
        double _volume      = 0.0;
        double _density     = 0.0;
        double _temperature = 0.0;
        double _pressure    = 0.0;
        double _momentum    = 0.0;

        double _kineticEnergy         = 0.0;
        double _coulombEnergy         = 0.0;
        double _nonCoulombEnergy      = 0.0;
        double _intraCoulombEnergy    = 0.0;
        double _intraNonCoulombEnergy = 0.0;

        double _bondEnergy     = 0.0;
        double _angleEnergy    = 0.0;
        double _dihedralEnergy = 0.0;
        double _improperEnergy = 0.0;

        linearAlgebra::Vec3D _virial                       = {0.0, 0.0, 0.0};
        linearAlgebra::Vec3D _momentumVector               = {0.0, 0.0, 0.0};
        linearAlgebra::Vec3D _kineticEnergyAtomicVector    = {0.0, 0.0, 0.0};
        linearAlgebra::Vec3D _kineticEnergyMolecularVector = {0.0, 0.0, 0.0};

      public:
        void calculateTemperature(simulationBox::SimulationBox &);

        void calculateKineticEnergyAndMomentum(simulationBox::SimulationBox &);

        void updateAverages(const PhysicalData &);
        void makeAverages(const double);
        void clearData();   // TODO: refactor this to reset

        std::function<linearAlgebra::Vec3D()> getKineticEnergyVirialVector =
            std::bind_front(&PhysicalData::getKineticEnergyMolecularVector, this);

        void changeKineticVirialToAtomic()
        {
            getKineticEnergyVirialVector = std::bind_front(&PhysicalData::getKineticEnergyAtomicVector, this);
        }

        void addVirial(const linearAlgebra::Vec3D virial) { _virial += virial; }

        void addCoulombEnergy(const double coulombEnergy) { _coulombEnergy += coulombEnergy; }
        void addNonCoulombEnergy(const double nonCoulombEnergy) { _nonCoulombEnergy += nonCoulombEnergy; }
        void addIntraCoulombEnergy(const double intraCoulombEnergy) { _intraCoulombEnergy += intraCoulombEnergy; }
        void addIntraNonCoulombEnergy(const double intraNonCoulombEnergy) { _intraNonCoulombEnergy += intraNonCoulombEnergy; }

        void addBondEnergy(const double bondEnergy) { _bondEnergy += bondEnergy; }
        void addAngleEnergy(const double angleEnergy) { _angleEnergy += angleEnergy; }
        void addDihedralEnergy(const double dihedralEnergy) { _dihedralEnergy += dihedralEnergy; }
        void addImproperEnergy(const double improperEnergy) { _improperEnergy += improperEnergy; }

        /********************
         * standard setters *
         ********************/

        void setVolume(const double volume) { _volume = volume; }
        void setDensity(const double density) { _density = density; }
        void setTemperature(const double temperature) { _temperature = temperature; }
        void setPressure(const double pressure) { _pressure = pressure; }
        void setVirial(const linearAlgebra::Vec3D &virial) { _virial = virial; }

        void setMomentum(const double momentum) { _momentum = momentum; }
        void setMomentumVector(const linearAlgebra::Vec3D &vec) { _momentumVector = vec; }

        void setKineticEnergy(const double kineticEnergy) { _kineticEnergy = kineticEnergy; }
        void setKineticEnergyAtomicVector(const linearAlgebra::Vec3D &vec) { _kineticEnergyAtomicVector = vec; }
        void setKineticEnergyMolecularVector(const linearAlgebra::Vec3D &vec) { _kineticEnergyMolecularVector = vec; }
        void setCoulombEnergy(const double coulombEnergy) { _coulombEnergy = coulombEnergy; }
        void setNonCoulombEnergy(const double nonCoulombEnergy) { _nonCoulombEnergy = nonCoulombEnergy; }
        void setIntraCoulombEnergy(const double intraCoulombEnergy) { _intraCoulombEnergy = intraCoulombEnergy; }
        void setIntraNonCoulombEnergy(const double intraNonCoulombEnergy) { _intraNonCoulombEnergy = intraNonCoulombEnergy; }

        void setBondEnergy(const double bondEnergy) { _bondEnergy = bondEnergy; }
        void setAngleEnergy(const double angleEnergy) { _angleEnergy = angleEnergy; }
        void setDihedralEnergy(const double dihedralEnergy) { _dihedralEnergy = dihedralEnergy; }
        void setImproperEnergy(const double improperEnergy) { _improperEnergy = improperEnergy; }

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] double getVolume() const { return _volume; }
        [[nodiscard]] double getDensity() const { return _density; }
        [[nodiscard]] double getTemperature() const { return _temperature; }
        [[nodiscard]] double getPressure() const { return _pressure; }
        [[nodiscard]] double getMomentum() const { return _momentum; }

        [[nodiscard]] double getNonCoulombEnergy() const { return _nonCoulombEnergy; }
        [[nodiscard]] double getCoulombEnergy() const { return _coulombEnergy; }
        [[nodiscard]] double getIntraCoulombEnergy() const { return _intraCoulombEnergy; }
        [[nodiscard]] double getIntraNonCoulombEnergy() const { return _intraNonCoulombEnergy; }
        [[nodiscard]] double getKineticEnergy() const { return _kineticEnergy; }

        [[nodiscard]] double getBondEnergy() const { return _bondEnergy; }
        [[nodiscard]] double getAngleEnergy() const { return _angleEnergy; }
        [[nodiscard]] double getDihedralEnergy() const { return _dihedralEnergy; }
        [[nodiscard]] double getImproperEnergy() const { return _improperEnergy; }

        [[nodiscard]] linearAlgebra::Vec3D getKineticEnergyAtomicVector() const { return _kineticEnergyAtomicVector; }
        [[nodiscard]] linearAlgebra::Vec3D getKineticEnergyMolecularVector() const { return _kineticEnergyMolecularVector; }
        [[nodiscard]] linearAlgebra::Vec3D getVirial() const { return _virial; }
        [[nodiscard]] linearAlgebra::Vec3D getMomentumVector() const { return _momentumVector; }
    };

}   // namespace physicalData

#endif   // _PHYSICAL_DATA_HPP_
