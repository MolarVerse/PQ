#ifndef _INTRA_NON_BONDED_HPP_

#define _INTRA_NON_BONDED_HPP_

#include "forceField.hpp"
#include "intraNonBondedContainer.hpp"
#include "intraNonBondedMap.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"

namespace intraNonBonded
{
    class IntraNonBonded;
    class IntraNonBondedGuff;
    class IntraNonBondedForceField;
    enum class IntraNonBondedType : size_t;
}   // namespace intraNonBonded

/**
 * @brief enum class for the different types of intra non bonded interactions
 *
 */
enum class intraNonBonded::IntraNonBondedType : size_t
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
class intraNonBonded::IntraNonBonded
{
  protected:
    IntraNonBondedType _intraNonBondedType = IntraNonBondedType::NONE;
    bool               _isActivated        = false;

    double _scale14Coulomb     = defaults::_SCALE_14_COULOMB_DEFAULT_;
    double _scale14VanDerWaals = defaults::_SCALE_14_VAN_DER_WAALS_DEFAULT_;

    std::vector<IntraNonBondedContainer> _intraNonBondedContainers;
    std::vector<IntraNonBondedMap>       _intraNonBondedMaps;

  public:
    void calculateEnergyAndForces(simulationBox::SimulationBox &, forceField::ForceField &, physicalData::PhysicalData &);
    IntraNonBondedContainer *findIntraNonBondedContainerByMolType(const size_t);

    void fillIntraNonBondedMaps(simulationBox::SimulationBox &);

    void addIntraNonBondedContainer(const IntraNonBondedContainer &intraNonBondedType)
    {
        _intraNonBondedContainers.push_back(intraNonBondedType);
    }
    void addIntraNonBondedMap(const IntraNonBondedMap &intraNonBondedInteraction)
    {
        _intraNonBondedMaps.push_back(intraNonBondedInteraction);
    }

    void               activate() { _isActivated = true; }
    void               deactivate() { _isActivated = false; }
    [[nodiscard]] bool isActivated() const { return _isActivated; }

    void setScale14Coulomb(const double scale14Coulomb) { _scale14Coulomb = scale14Coulomb; }
    void setScale14VanDerWaals(const double scale14VanDerWaals) { _scale14VanDerWaals = scale14VanDerWaals; }

    [[nodiscard]] double                               getScale14Coulomb() const { return _scale14Coulomb; }
    [[nodiscard]] double                               getScale14VanDerWaals() const { return _scale14VanDerWaals; }
    [[nodiscard]] IntraNonBondedType                   getIntraNonBondedType() const { return _intraNonBondedType; }
    [[nodiscard]] std::vector<IntraNonBondedContainer> getIntraNonBondedContainers() const { return _intraNonBondedContainers; }
    [[nodiscard]] std::vector<IntraNonBondedMap>       getIntraNonBondedMaps() const { return _intraNonBondedMaps; }
};

/**
 * @class IntraNonBondedGuff
 *
 * @brief inherits from IntraNonBonded
 */
class intraNonBonded::IntraNonBondedGuff : public IntraNonBonded
{
  public:
    IntraNonBondedGuff() : IntraNonBonded() { _intraNonBondedType = IntraNonBondedType::GUFF; }
};

/**
 * @class IntraNonBondedForceField
 *
 * @brief inherits from IntraNonBonded
 */
class intraNonBonded::IntraNonBondedForceField : public IntraNonBonded
{
  public:
    IntraNonBondedForceField() : IntraNonBonded() { _intraNonBondedType = IntraNonBondedType::FORCE_FIELD; }
};

#endif   // _INTRA_NON_BONDED_HPP_