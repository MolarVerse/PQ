#include "bondConstraint.hpp"

using namespace constraints;

/**
 * @brief calculates the reference bond data of a bond constraint
 *
 * @param simulationBox
 */
void BondConstraint::calculateConstraintBondRef(simulationBox::SimulationBox &simBox)
{
    _shakeDistanceRef = _molecule1->getAtomPosition(_atomIndex1) - _molecule2->getAtomPosition(_atomIndex2);

    simBox.applyPBC(_shakeDistanceRef);
}

/**
 * @brief applies the shake algorithm to a bond constraint
 *
 */
bool BondConstraint::applyShake(simulationBox::SimulationBox &simBox, double tolerance, double timestep)
{
    auto pos1 = _molecule1->getAtomPosition(_atomIndex1);
    auto pos2 = _molecule2->getAtomPosition(_atomIndex2);

    auto dpos = pos1 - pos2;
    simBox.applyPBC(dpos);

    const auto distanceSquared = normSquared(dpos);
    const auto delta = 0.5 * (_targetBondLength * _targetBondLength - distanceSquared) / (_targetBondLength * _targetBondLength);

    if (std::fabs(delta) < tolerance)
    {
        auto invMass1 = 1 / _molecule1->getAtomMass(_atomIndex1);
        auto invMass2 = 1 / _molecule2->getAtomMass(_atomIndex2);

        const auto shakeForce = delta / (invMass1 + invMass2) / normSquared(_shakeDistanceRef);

        dpos = shakeForce * _shakeDistanceRef;

        _molecule1->setAtomPosition(_atomIndex1, pos1 + invMass1 * dpos);
        _molecule2->setAtomPosition(_atomIndex2, pos2 - invMass2 * dpos);

        dpos = dpos / timestep;

        _molecule1->setAtomVelocity(_atomIndex1, _molecule1->getAtomVelocity(_atomIndex1) + invMass1 * dpos);
        _molecule2->setAtomVelocity(_atomIndex2, _molecule2->getAtomVelocity(_atomIndex2) - invMass2 * dpos);

        return false;
    }

    return true;
}