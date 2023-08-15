#include "potential.hpp"

#include <cmath>
#include <iostream>

using namespace std;
using namespace simulationBox;
using namespace potential;
using namespace physicalData;
using namespace linearAlgebra;

/**
 * @brief determines internal global van der Waals types and sets them in the NonCoulombPair objects
 *
 * @param _externalToInternalGlobalVDWTypes
 */
void Potential::determineInternalGlobalVdwTypes(const map<size_t, size_t> &externalToInternalGlobalVDWTypes)
{
    auto setInternalVanDerWaalsType = [&externalToInternalGlobalVDWTypes](auto &nonCoulombicPair)
    {
        nonCoulombicPair->setInternalType1(externalToInternalGlobalVDWTypes.at(nonCoulombicPair->getVanDerWaalsType1()));
        nonCoulombicPair->setInternalType2(externalToInternalGlobalVDWTypes.at(nonCoulombicPair->getVanDerWaalsType2()));
    };

    ranges::for_each(_nonCoulombicPairsVector, setInternalVanDerWaalsType);
}

/**
 * @brief fills the diagonal elements of the non-coulombic pairs matrix
 *
 * @param diagonalElements
 */
void Potential::fillDiagonalElementsOfNonCoulombicPairsMatrix(vector<shared_ptr<NonCoulombPair>> &diagonalElements)
{
    ranges::sort(diagonalElements,
                 [](const auto &nonCoulombicPair1, const auto &nonCoulombicPair2)
                 { return nonCoulombicPair1->getInternalType1() < nonCoulombicPair2->getInternalType1(); });

    auto numberOfDiagonalElements = diagonalElements.size();
    initNonCoulombicPairsMatrix(numberOfDiagonalElements);

    for (size_t i = 0; i < numberOfDiagonalElements; ++i)
        _nonCoulombicPairsMatrix[i][i] = diagonalElements[i];
}

/**
 * @brief fills the off-diagonal elements of the non-coulombic pairs matrix
 *
 * @details if no type is found check if mixing rules are used
 * if mixing rules are used, then add a new non-coulombic pair with the mixing rule (TODO: not yet implemented)
 * if no mixing rules are used, then throw an exception
 * if a type is found, then add the non-coulombic pair to the matrix
 * if a type is found with both index combinations and it has the same parameters, then add the non-coulombic pair
 * if a type is found with both index combinations and it has different parameters, then throw an exception
 *
 * @throws customException::ParameterFileException if mixing rules are not used and not all combinations of global van der Waals
 * types are defined in the parameter file
 * @throws customException::ParameterFileException if type is found with both index combinations and it has different parameters
 */
void Potential::fillNonDiagonalElementsOfNonCoulombicPairsMatrix()
{
    const auto &[rows, cols] = _nonCoulombicPairsMatrix.shape();

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = i + 1; j < cols; ++j)
        {
            auto nonCoulombicPair1 = findNonCoulombicPairByInternalTypes(i, j);
            auto nonCoulombicPair2 = findNonCoulombicPairByInternalTypes(j, i);

            if (nonCoulombicPair1 == nullopt && nonCoulombicPair2 == nullopt)
            {
                if (_mixingRule == MixingRule::NONE)
                {
                    throw customException::ParameterFileException(
                        "Not all combinations of global van der Waals types are defined in the parameter file - and no mixing "
                        "rules were chosen");
                }
            }
            else if (nonCoulombicPair1 != nullopt && nonCoulombicPair2 != nullopt)
            {
                if (**nonCoulombicPair1 != **nonCoulombicPair2)
                {
                    const auto vdwType1 = (*nonCoulombicPair1)->getVanDerWaalsType1();
                    const auto vdwType2 = (*nonCoulombicPair1)->getVanDerWaalsType2();
                    throw customException::ParameterFileException(
                        format("Non-coulombic pairs with global van der Waals types {}, {} and {}, {} in the parameter file have "
                               "different parameters",
                               vdwType1,
                               vdwType2,
                               vdwType2,
                               vdwType1));
                }

                _nonCoulombicPairsMatrix[i][j] = *nonCoulombicPair1;
                _nonCoulombicPairsMatrix[j][i] = *nonCoulombicPair1;
            }
            else if (nonCoulombicPair1 != nullopt)
            {
                _nonCoulombicPairsMatrix[i][j] = *nonCoulombicPair1;
                _nonCoulombicPairsMatrix[j][i] = *nonCoulombicPair1;
            }
            else
            {
                _nonCoulombicPairsMatrix[i][j] = *nonCoulombicPair2;
                _nonCoulombicPairsMatrix[j][i] = *nonCoulombicPair2;
            }
        }
}

/**
 * @brief finds all non-coulombic pairs that are self-interaction pairs
 *
 * @details self interacting means that both internal or external types are the same
 *
 * @return vector<shared_ptr<NonCoulombPair>>
 */
vector<shared_ptr<NonCoulombPair>> Potential::getSelfInteractionNonCoulombicPairs() const
{
    vector<shared_ptr<NonCoulombPair>> selfInteractionNonCoulombicPairs;

    auto addSelfInteractionPairs = [&selfInteractionNonCoulombicPairs](const auto &nonCoulombicPair)
    {
        if (nonCoulombicPair->getInternalType1() == nonCoulombicPair->getInternalType2())
            selfInteractionNonCoulombicPairs.push_back(nonCoulombicPair);
    };

    ranges::for_each(_nonCoulombicPairsVector, addSelfInteractionPairs);

    return selfInteractionNonCoulombicPairs;
}

/**
 * @brief finds a non coulombic pair by internal types
 *
 * @details if the non coulombic pair is not found, an empty optional is returned
 *          if the non coulombic pair is found twice an exception is thrown
 *
 * @param internalType1
 * @param internalType2
 * @return optional<shared_ptr<NonCoulombPair>>
 *
 * @throws if the non coulombic pair is found twice
 */
optional<shared_ptr<NonCoulombPair>> Potential::findNonCoulombicPairByInternalTypes(size_t internalType1,
                                                                                    size_t internalType2) const
{
    auto findByInternalAtomTypes = [internalType1, internalType2](const auto &nonCoulombicPair)
    { return nonCoulombicPair->getInternalType1() == internalType1 && nonCoulombicPair->getInternalType2() == internalType2; };

    if (auto firstNonCoulombicPair = ranges::find_if(_nonCoulombicPairsVector, findByInternalAtomTypes);
        firstNonCoulombicPair != _nonCoulombicPairsVector.end())
    {
        if (auto secondCoulombicPair =
                ranges::find_if(firstNonCoulombicPair + 1, _nonCoulombicPairsVector.end(), findByInternalAtomTypes);
            secondCoulombicPair != _nonCoulombicPairsVector.end())
        {
            auto vdwType1 = (*firstNonCoulombicPair)->getVanDerWaalsType1();
            auto vdwType2 = (*firstNonCoulombicPair)->getVanDerWaalsType2();
            throw customException::ParameterFileException(
                format("Non coulombic pair with global van der waals types {} and {} is defined twice in the parameter file.",
                       vdwType1,
                       vdwType2));
        }
        else
        {
            return *firstNonCoulombicPair;
        }
    }
    else
        return nullopt;
}

/**
 * @brief inner part of the double loop to calculate non-bonded inter molecular interactions
 *
 * @param box
 * @param molecule1
 * @param molecule2
 * @param atom1
 * @param atom2
 * @return std::pair<double, double>
 */
std::pair<double, double> Potential::calculateSingleInteraction(const linearAlgebra::Vec3D &box,
                                                                simulationBox::Molecule    &molecule1,
                                                                simulationBox::Molecule    &molecule2,
                                                                const size_t                atom1,
                                                                const size_t                atom2)
{
    auto coulombEnergy    = 0.0;
    auto nonCoulombEnergy = 0.0;

    const auto xyz_i = molecule1.getAtomPosition(atom1);
    const auto xyz_j = molecule2.getAtomPosition(atom2);

    auto dxyz = xyz_i - xyz_j;

    const auto txyz = -box * round(dxyz / box);

    // dxyz += txyz;
    dxyz[0] += txyz[0];
    dxyz[1] += txyz[1];
    dxyz[2] += txyz[2];

    const double distanceSquared = normSquared(dxyz);

    if (const auto RcCutOff = CoulombPotential::getCoulombRadiusCutOff(); distanceSquared < RcCutOff * RcCutOff)
    {
        const double distance   = ::sqrt(distanceSquared);
        const size_t atomType_i = molecule1.getAtomType(atom1);
        const size_t atomType_j = molecule2.getAtomType(atom2);

        // TODO: think of a clever solution for guff routine
        //  const size_t externalGlobalVdwType_i = molecule1.getExternalGlobalVDWType(atom1);
        //  const size_t externalGlobalVdwType_j = molecule2.getExternalGlobalVDWType(atom2);

        // const size_t globalVdwType_i =
        // simBox.getExternalToInternalGlobalVDWTypes().at(externalGlobalVdwType_i); const size_t globalVdwType_j
        // = simBox.getExternalToInternalGlobalVDWTypes().at(externalGlobalVdwType_j);

        const size_t globalVdwType_i = 0;
        const size_t globalVdwType_j = 0;

        const auto moltype_i = molecule1.getMoltype();
        const auto moltype_j = molecule2.getMoltype();

        const auto combinedIndices = {moltype_i, moltype_j, atomType_i, atomType_j, globalVdwType_i, globalVdwType_j};

        const auto coulombPreFactor = 1.0;   // TODO: implement for force field

        auto [energy, force] = _coulombPotential->calculate(combinedIndices, distance, coulombPreFactor);
        coulombEnergy        = energy;

        const auto nonCoulombicPair = _nonCoulombPotential->getNonCoulombPair(combinedIndices);

        if (const auto rncCutOff = nonCoulombicPair->getRadialCutOff(); distance < rncCutOff)
        {
            const auto &[energy, nonCoulombForce] = nonCoulombicPair->calculateEnergyAndForce(distance);
            nonCoulombEnergy                      = energy;

            force += nonCoulombForce;
        }

        force /= distance;

        const auto forcexyz = force * dxyz;

        const auto shiftForcexyz = forcexyz * txyz;

        molecule1.addAtomForce(atom1, forcexyz);
        molecule2.addAtomForce(atom2, -forcexyz);

        molecule1.addAtomShiftForce(atom1, shiftForcexyz);
    }

    return {coulombEnergy, nonCoulombEnergy};
}