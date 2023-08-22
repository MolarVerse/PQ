#include "forceFieldNonCoulomb.hpp"

#include "exceptions.hpp"

#include <format>
#include <map>
#include <ranges>

using namespace potential;
using namespace std;

/**
 * @brief determines internal global van der Waals types and sets them in the NonCoulombPair objects
 *
 * @param _externalToInternalGlobalVDWTypes
 */
void ForceFieldNonCoulomb::determineInternalGlobalVdwTypes(const map<size_t, size_t> &externalToInternalGlobalVDWTypes)
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
void ForceFieldNonCoulomb::fillDiagonalElementsOfNonCoulombicPairsMatrix(vector<shared_ptr<NonCoulombPair>> &diagonalElements)
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
void ForceFieldNonCoulomb::fillNonDiagonalElementsOfNonCoulombicPairsMatrix()
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
vector<shared_ptr<NonCoulombPair>> ForceFieldNonCoulomb::getSelfInteractionNonCoulombicPairs() const
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
optional<shared_ptr<NonCoulombPair>> ForceFieldNonCoulomb::findNonCoulombicPairByInternalTypes(size_t internalType1,
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