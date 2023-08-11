#ifndef _INTRA_NON_BONDED_HPP_

#define _INTRA_NON_BONDED_HPP_

#include "intraNonBondedInteraction.hpp"
#include "intraNonBondedType.hpp"

namespace intraNonBonded
{
    class IntraNonBonded;
    class IntraNonBondedGuff;
    class IntraNonBondedForceField;
}   // namespace intraNonBonded

/**
 * @class IntraNonBonded
 *
 * @brief base class for intra non bonded interactions
 */
class intraNonBonded::IntraNonBonded
{
  protected:
    bool _isActivated = false;

    std::vector<IntraNonBondedType>        _intraNonBondedTypes;
    std::vector<IntraNonBondedInteraction> _intraNonBondedInteractions;

  public:
    void addIntraNonBondedType(const IntraNonBondedType &intraNonBondedType);
    void addIntraNonBondedInteraction(const IntraNonBondedInteraction &intraNonBondedInteraction);

    void               activate() { _isActivated = true; }
    void               deactivate() { _isActivated = false; }
    [[nodiscard]] bool isActivated() const { return _isActivated; }

    [[nodiscard]] std::vector<IntraNonBondedType>        getIntraNonBondedTypes() const;
    [[nodiscard]] std::vector<IntraNonBondedInteraction> getIntraNonBondedInteractions() const;
};

/**
 * @class IntraNonBondedGuff
 *
 * @brief inherits from IntraNonBonded
 */
class intraNonBonded::IntraNonBondedGuff : public IntraNonBonded
{
  public:
    using IntraNonBonded::IntraNonBonded;
};

/**
 * @class IntraNonBondedForceField
 *
 * @brief inherits from IntraNonBonded
 */
class intraNonBonded::IntraNonBondedForceField : public IntraNonBonded
{
  public:
    using IntraNonBonded::IntraNonBonded;
};

#endif   // _INTRA_NON_BONDED_HPP_