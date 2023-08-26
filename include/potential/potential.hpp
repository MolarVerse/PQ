#ifndef _POTENTIAL_HPP_

#define _POTENTIAL_HPP_

#include "vector3d.hpp"   // for Vec3D

#include <cstddef>   // for size_t
#include <memory>    // for shared_ptr, __shared_ptr_access, make_shared
#include <utility>   // for pair

namespace physicalData
{
    class PhysicalData;
}

namespace simulationBox
{
    class CellList;
    class Molecule;
    class SimulationBox;
}   // namespace simulationBox

namespace potential
{
    class CoulombPotential;       // forward declaration
    class NonCoulombPotential;    // forward declaration
    class ForceFieldNonCoulomb;   // forward declaration

    /**
     * @class Potential
     *
     * @brief base class for all potential routines
     *
     * @details
     * possible options:
     * - brute force
     * - cell list
     *
     * @note _nonCoulombicPairsVector is just a container to store the nonCoulombicPairs for later processing
     *
     */
    class Potential
    {
      protected:
        std::shared_ptr<CoulombPotential>    _coulombPotential;
        std::shared_ptr<NonCoulombPotential> _nonCoulombPotential;

      public:
        virtual ~Potential() = default;

        std::pair<double, double> calculateSingleInteraction(
            const linearAlgebra::Vec3D &, simulationBox::Molecule &, simulationBox::Molecule &, const size_t, const size_t) const;
        virtual void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) = 0;

        template <typename T>
        void makeCoulombPotential(T coulombPotential)
        {
            _coulombPotential = std::make_shared<T>(coulombPotential);
        }
        template <typename T>
        void makeNonCoulombPotential(T nonCoulombPotential)
        {
            _nonCoulombPotential = std::make_shared<T>(nonCoulombPotential);
        }

        void setNonCoulombPotential(std::shared_ptr<NonCoulombPotential> nonCoulombPotential)
        {
            _nonCoulombPotential = nonCoulombPotential;
        }

        [[nodiscard]] CoulombPotential                    &getCoulombPotential() const { return *_coulombPotential; }
        [[nodiscard]] NonCoulombPotential                 &getNonCoulombPotential() const { return *_nonCoulombPotential; }
        [[nodiscard]] std::shared_ptr<CoulombPotential>    getCoulombPotentialSharedPtr() const { return _coulombPotential; }
        [[nodiscard]] std::shared_ptr<NonCoulombPotential> getNonCoulombPotentialSharedPtr() const
        {
            return _nonCoulombPotential;
        }
    };

    /**
     * @class PotentialBruteForce
     *
     * @brief brute force implementation of the potential
     *
     */
    class PotentialBruteForce : public Potential
    {
      public:
        void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) override;
    };

    /**
     * @class PotentialCellList
     *
     * @brief cell list implementation of the potential
     *
     */
    class PotentialCellList : public Potential
    {
      public:
        void calculateForces(simulationBox::SimulationBox &, physicalData::PhysicalData &, simulationBox::CellList &) override;
    };

}   // namespace potential

#endif   // _POTENTIAL_HPP_