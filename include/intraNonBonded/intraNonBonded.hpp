#ifndef _INTRA_NON_BONDED_HPP_

#define _INTRA_NON_BONDED_HPP_

#include "intraNonBondedContainer.hpp"   // for IntraNonBondedContainer
#include "intraNonBondedMap.hpp"         // for IntraNonBondedMap

#include <cstddef>   // for size_t
#include <memory>    // for shared_ptr
#include <vector>    // for vector

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace potential
{
    class CoulombPotential;      // forward declaration
    class NonCoulombPotential;   // forward declaration
}   // namespace potential

namespace intraNonBonded
{
    using vec_intra_container = std::vector<IntraNonBondedContainer>;

    /**
     * @brief enum class for the different types of intra non bonded interactions
     *
     */
    enum class IntraNonBondedType : size_t
    {
        NONE,
        GUFF,
        FORCE_FIELD
    };

    /**
     * @class IntraNonBonded
     *
     * @brief base class for intra non bonded interactions
     */
    class IntraNonBonded
    {
      protected:
        IntraNonBondedType _intraNonBondedType = IntraNonBondedType::NONE;
        bool               _isActivated        = false;

        std::shared_ptr<potential::NonCoulombPotential> _nonCoulombPotential;
        std::shared_ptr<potential::CoulombPotential>    _coulombPotential;
        std::vector<IntraNonBondedMap>                  _intraNonBondedMaps;
        vec_intra_container                             _intraNonBondedContainers;

      public:
        void calculate(const simulationBox::SimulationBox &, physicalData::PhysicalData &);
        void fillIntraNonBondedMaps(simulationBox::SimulationBox &);

        [[nodiscard]] IntraNonBondedContainer *findIntraNonBondedContainerByMolType(const size_t);

        /*************************
         * standard add methods  *
         *************************/

        void addIntraNonBondedContainer(const IntraNonBondedContainer &type) { _intraNonBondedContainers.push_back(type); }
        void addIntraNonBondedMap(const IntraNonBondedMap &interaction) { _intraNonBondedMaps.push_back(interaction); }

        /*****************************
         * standard activate methods *
         *****************************/

        void               activate() { _isActivated = true; }
        void               deactivate() { _isActivated = false; }
        [[nodiscard]] bool isActivated() const { return _isActivated; }

        /***************************
         * standard setter methods *
         ***************************/

        void setNonCoulombPotential(const std::shared_ptr<potential::NonCoulombPotential> &pot) { _nonCoulombPotential = pot; }
        void setCoulombPotential(const std::shared_ptr<potential::CoulombPotential> &pot) { _coulombPotential = pot; }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] IntraNonBondedType             getIntraNonBondedType() const { return _intraNonBondedType; }
        [[nodiscard]] vec_intra_container            getIntraNonBondedContainers() const { return _intraNonBondedContainers; }
        [[nodiscard]] std::vector<IntraNonBondedMap> getIntraNonBondedMaps() const { return _intraNonBondedMaps; }
    };

}   // namespace intraNonBonded

#endif   // _INTRA_NON_BONDED_HPP_