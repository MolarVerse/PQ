#include "bondConstraint.hpp"

#include "molecule.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"
#include "vector3d.hpp"

#include <cmath>
#include <vector>

using namespace constraints;

/**
 * @brief calculates the reference bond data of a bond constraint
 *
 * @param simulationBox
 */
void BondConstraint::calculateConstraintBondRef(const simulationBox::SimulationBox &simBox)
{
    _shakeDistanceRef = _molecules[0]->getAtomPosition(_atomIndices[0]) - _molecules[1]->getAtomPosition(_atomIndices[1]);

    simBox.applyPBC(_shakeDistanceRef);
}

/**
 * @brief calculates the distance delta of a bond constraint
 *
 */
double BondConstraint::calculateDistanceDelta(const simulationBox::SimulationBox &simBox) const
{
    const auto pos1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    const auto pos2 = _molecules[1]->getAtomPosition(_atomIndices[1]);

    auto dPosition = pos1 - pos2;
    simBox.applyPBC(dPosition);

    const auto distanceSquared = normSquared(dPosition);
    const auto delta           = 0.5 * (_targetBondLength * _targetBondLength - distanceSquared);

    return delta;
}

/**
 * @brief applies the shake algorithm to a bond constraint
 *
 * @details if delta is not smaller than tolerance, the shake algorithm is applied
 *
 */
bool BondConstraint::applyShake(const simulationBox::SimulationBox &simBox, double tolerance)
{

    if (const auto delta = calculateDistanceDelta(simBox); std::fabs(delta / (_targetBondLength * _targetBondLength)) > tolerance)
    {
        const auto invMass1 = 1 / _molecules[0]->getAtom(_atomIndices[0])->getMass();
        const auto invMass2 = 1 / _molecules[1]->getAtom(_atomIndices[1])->getMass();

        const auto shakeForce = delta / (invMass1 + invMass2) / normSquared(_shakeDistanceRef);

        const auto dPosition = shakeForce * _shakeDistanceRef;

        _molecules[0]->getAtom(_atomIndices[0])->addPosition(+invMass1 * dPosition);
        _molecules[1]->getAtom(_atomIndices[1])->addPosition(-invMass2 * dPosition);

        const auto dVelocity = dPosition / settings::TimingsSettings::getTimeStep();

        _molecules[0]->getAtom(_atomIndices[0])->addVelocity(+invMass1 * dVelocity);
        _molecules[1]->getAtom(_atomIndices[1])->addVelocity(-invMass2 * dVelocity);

        return false;
    }

    return true;
}

/**
 * @brief calculates the velocity delta of a bond constraint
 *
 */
[[nodiscard]] double BondConstraint::calculateVelocityDelta() const
{
    const auto dVelocity =
        _molecules[0]->getAtom(_atomIndices[0])->getVelocity() - _molecules[1]->getAtom(_atomIndices[1])->getVelocity();

    const auto scalarProduct = dot(dVelocity, _shakeDistanceRef);

    const auto invMass1 = 1 / _molecules[0]->getAtom(_atomIndices[0])->getMass();
    const auto invMass2 = 1 / _molecules[1]->getAtom(_atomIndices[1])->getMass();

    const auto delta = -scalarProduct / (invMass1 + invMass2) / normSquared(_shakeDistanceRef);

    return delta;
}

/**
 * @brief applies the rattle algorithm to a bond constraint
 *
 * @details if delta is not smaller than tolerance, the rattle algorithm is applied
 *
 */
bool BondConstraint::applyRattle(double tolerance)
{
    if (const auto delta = calculateVelocityDelta(); std::fabs(delta) > tolerance)
    {
        const auto dVelocity = delta * _shakeDistanceRef;

        const auto invMass1 = 1 / _molecules[0]->getAtom(_atomIndices[0])->getMass();
        const auto invMass2 = 1 / _molecules[1]->getAtom(_atomIndices[1])->getMass();

        _molecules[0]->getAtom(_atomIndices[0])->addVelocity(+invMass1 * dVelocity);
        _molecules[1]->getAtom(_atomIndices[1])->addVelocity(-invMass2 * dVelocity);

        return false;
    }

    return true;
}