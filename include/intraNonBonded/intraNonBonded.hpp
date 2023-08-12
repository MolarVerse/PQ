#ifndef _INTRA_NON_BONDED_HPP_

#define _INTRA_NON_BONDED_HPP_

#include "intraNonBondedContainer.hpp"
#include "intraNonBondedMap.hpp"

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

    std::vector<IntraNonBondedContainer> _intraNonBondedTypes;
    std::vector<IntraNonBondedMap>       _intraNonBondedInteractions;

  public:
    void addIntraNonBondedContainer(const IntraNonBondedContainer &intraNonBondedType);
    void addIntraNonBondedMap(const IntraNonBondedMap &intraNonBondedInteraction);

    void               activate() { _isActivated = true; }
    void               deactivate() { _isActivated = false; }
    [[nodiscard]] bool isActivated() const { return _isActivated; }

    [[nodiscard]] IntraNonBondedType                   getIntraNonBondedType() const { return _intraNonBondedType; }
    [[nodiscard]] std::vector<IntraNonBondedContainer> getIntraNonBondedContainers() const { return _intraNonBondedTypes; }
    [[nodiscard]] std::vector<IntraNonBondedMap>       getIntraNonBondedMaps() const { return _intraNonBondedInteractions; }
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