#include "bondConstraint.hpp"

#include "simulationBox.hpp"
#include "vector3d.hpp"

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

    auto dPosition = pos1 - pos2;
    simBox.applyPBC(dPosition);

    const auto distanceSquared = normSquared(dPosition);
    const auto delta = 0.5 * (_targetBondLength * _targetBondLength - distanceSquared) / (_targetBondLength * _targetBondLength);

    return delta;
}

/**
 * @brief applies the shake algorithm to a bond constraint
 *
 */
bool BondConstraint::applyShake(const simulationBox::SimulationBox &simBox, double tolerance, double timestep)
{

    if (const auto delta = calculateDistanceDelta(simBox); std::fabs(delta) > tolerance)
    {
        auto invMass1 = 1 / _molecules[0]->getAtomMass(_atomIndices[0]);
        auto invMass2 = 1 / _molecules[1]->getAtomMass(_atomIndices[1]);

        const auto shakeForce = delta / (invMass1 + invMass2) / normSquared(_shakeDistanceRef);
        std::cout << invMass1 << " " << invMass2 << std::endl;

        auto dPosition = shakeForce * _shakeDistanceRef;

        _molecules[0]->addAtomPosition(_atomIndices[0], +invMass1 * dPosition);
        _molecules[1]->addAtomPosition(_atomIndices[1], -invMass2 * dPosition);

        dPosition = dPosition / timestep;

        _molecules[0]->addAtomVelocity(_atomIndices[0], +invMass1 * dPosition);
        _molecules[1]->addAtomVelocity(_atomIndices[1], -invMass2 * dPosition);

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

    if (const auto delta = calculateVelocityDelta(); std::fabs(delta) > tolerance)
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