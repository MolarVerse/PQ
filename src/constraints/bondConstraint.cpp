#include "bondConstraint.hpp"

using namespace constraints;

/**
 * @brief calculates the reference bond data of a bond constraint
 *
 * @param simulationBox
 */
void BondConstraint::calculateConstraintBondRef(const simulationBox::SimulationBox &simBox)
{
    _shakeDistanceRef = _molecule1->getAtomPosition(_atomIndex1) - _molecule2->getAtomPosition(_atomIndex2);

    simBox.applyPBC(_shakeDistanceRef);
}

/**
 * @brief calculates the distance delta of a bond constraint
 *
 */
double BondConstraint::calculateDistanceDelta(const simulationBox::SimulationBox &simBox) const
{
    auto pos1 = _molecule1->getAtomPosition(_atomIndex1);
    auto pos2 = _molecule2->getAtomPosition(_atomIndex2);

    auto dpos = pos1 - pos2;
    simBox.applyPBC(dpos);

    const auto distanceSquared = normSquared(dpos);
    const auto delta = 0.5 * (_targetBondLength * _targetBondLength - distanceSquared) / (_targetBondLength * _targetBondLength);

    return delta;
}

/**
 * @brief applies the shake algorithm to a bond constraint
 *
 */
bool BondConstraint::applyShake(const simulationBox::SimulationBox &simBox, double tolerance, double timestep)
{
    const auto delta = calculateDistanceDelta(simBox);

    if (std::fabs(delta) > tolerance)
    {
        auto invMass1 = 1 / _molecule1->getAtomMass(_atomIndex1);
        auto invMass2 = 1 / _molecule2->getAtomMass(_atomIndex2);

        const auto shakeForce = delta / (invMass1 + invMass2) / normSquared(_shakeDistanceRef);

        auto dpos = shakeForce * _shakeDistanceRef;

        _molecule1->addAtomPosition(_atomIndex1, +invMass1 * dpos);
        _molecule2->addAtomPosition(_atomIndex2, -invMass2 * dpos);

        dpos = dpos / timestep;

        _molecule1->addAtomVelocity(_atomIndex1, +invMass1 * dpos);
        _molecule2->addAtomVelocity(_atomIndex2, -invMass2 * dpos);

        return false;
    }

    return true;
}

/**
 * @brief calculates the velocity delta of a bond constraint
 *
 */
double BondConstraint::calculateVelocityDelta() const
{
    auto dv = _molecule1->getAtomVelocity(_atomIndex1) - _molecule2->getAtomVelocity(_atomIndex2);

    auto scalarProduct = dot(dv, _shakeDistanceRef);

    const auto invMass1 = 1 / _molecule1->getAtomMass(_atomIndex1);
    const auto invMass2 = 1 / _molecule2->getAtomMass(_atomIndex2);

    const auto delta = -scalarProduct / (invMass1 + invMass2) / normSquared(_shakeDistanceRef);

    return delta;
}

/**
 * @brief applies the rattle algorithm to a bond constraint
 *
 */
bool BondConstraint::applyRattle(double tolerance)
{
    const auto delta = calculateVelocityDelta();

    if (std::fabs(delta) > tolerance)
    {
        const auto dv = delta * _shakeDistanceRef;

        const auto invMass1 = 1 / _molecule1->getAtomMass(_atomIndex1);
        const auto invMass2 = 1 / _molecule2->getAtomMass(_atomIndex2);

        _molecule1->addAtomVelocity(_atomIndex1, +invMass1 * dv);
        _molecule2->addAtomVelocity(_atomIndex2, -invMass2 * dv);

        return false;
    }

    return true;
}