#include "bondConstraint.hpp"

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
    auto pos1 = _molecules[0]->getAtomPosition(_atomIndices[0]);
    auto pos2 = _molecules[1]->getAtomPosition(_atomIndices[1]);

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
        auto invMass1 = 1 / _molecules[0]->getAtomMass(_atomIndices[0]);
        auto invMass2 = 1 / _molecules[1]->getAtomMass(_atomIndices[1]);

        const auto shakeForce = delta / (invMass1 + invMass2) / normSquared(_shakeDistanceRef);

        auto dpos = shakeForce * _shakeDistanceRef;

        _molecules[0]->addAtomPosition(_atomIndices[0], +invMass1 * dpos);
        _molecules[1]->addAtomPosition(_atomIndices[1], -invMass2 * dpos);

        dpos = dpos / timestep;

        _molecules[0]->addAtomVelocity(_atomIndices[0], +invMass1 * dpos);
        _molecules[1]->addAtomVelocity(_atomIndices[1], -invMass2 * dpos);

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
    auto dv = _molecules[0]->getAtomVelocity(_atomIndices[0]) - _molecules[1]->getAtomVelocity(_atomIndices[1]);

    auto scalarProduct = dot(dv, _shakeDistanceRef);

    const auto invMass1 = 1 / _molecules[0]->getAtomMass(_atomIndices[0]);
    const auto invMass2 = 1 / _molecules[1]->getAtomMass(_atomIndices[1]);

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

        const auto invMass1 = 1 / _molecules[0]->getAtomMass(_atomIndices[0]);
        const auto invMass2 = 1 / _molecules[1]->getAtomMass(_atomIndices[1]);

        _molecules[0]->addAtomVelocity(_atomIndices[0], +invMass1 * dv);
        _molecules[1]->addAtomVelocity(_atomIndices[1], -invMass2 * dv);

        return false;
    }

    return true;
}