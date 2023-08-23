#ifndef _INTRA_NON_BONDED_HPP_

#define _INTRA_NON_BONDED_HPP_

#include "coulombPotential.hpp"
#include "forceField.hpp"
#include "intraNonBondedContainer.hpp"
#include "intraNonBondedMap.hpp"
#include "nonCoulombPotential.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"

namespace intraNonBonded
{
    class IntraNonBonded;
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

    std::shared_ptr<potential::NonCoulombPotential> _nonCoulombPotential;
    std::shared_ptr<potential::CoulombPotential>    _coulombPotential;
    std::vector<IntraNonBondedContainer>            _intraNonBondedContainers;
    std::vector<IntraNonBondedMap>                  _intraNonBondedMaps;

  public:
    void                                   calculate(simulationBox::SimulationBox &, physicalData::PhysicalData &);
    [[nodiscard]] IntraNonBondedContainer *findIntraNonBondedContainerByMolType(const size_t);

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

    void setNonCoulombPotential(const std::shared_ptr<potential::NonCoulombPotential> &nonCoulombPotential)
    {
        _nonCoulombPotential = nonCoulombPotential;
    }
    void setCoulombPotential(const std::shared_ptr<potential::CoulombPotential> &coulombPotential)
    {
        _coulombPotential = coulombPotential;
    }

    [[nodiscard]] IntraNonBondedType                   getIntraNonBondedType() const { return _intraNonBondedType; }
    [[nodiscard]] std::vector<IntraNonBondedContainer> getIntraNonBondedContainers() const { return _intraNonBondedContainers; }
    [[nodiscard]] std::vector<IntraNonBondedMap>       getIntraNonBondedMaps() const { return _intraNonBondedMaps; }
};

#endif   // _INTRA_NON_BONDED_HPP_