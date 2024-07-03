/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "forceFieldNonCoulomb.hpp"

#include <algorithm>     // for copy, max
#include <format>        // for std::format
#include <functional>    // for identity
#include <map>           // for map
#include <ranges>        // for __find_if_fn, find_if
#include <string>        // for string
#include <string_view>   // for string_view

#include "exceptions.hpp"       // for ParameterFileException
#include "nonCoulombPair.hpp"   // for NonCoulombPair

using namespace potential;

/**
 * @brief gets a shared pointer to a NonCoulombPair object
 *
 * @details the indices vector contains the indices of the atoms in the system,
 * the last two indices are the global van der Waals
 *
 * @param indices
 * @return std::shared_ptr<NonCoulombPair>
 */
std::shared_ptr<NonCoulombPair> ForceFieldNonCoulomb::getNonCoulPair(
    const std::vector<size_t> &indices
)
{
    return _nonCoulombPairsMatrix(
        getGlobalVdwType1(indices),
        getGlobalVdwType2(indices)
    );
}

/**
 * @brief calculates and sets energy and force cutoff for all non-coulombic
 * pairs in _nonCoulombPairsVector
 *
 */
void ForceFieldNonCoulomb::setupNonCoulombicCutoffs()
{
    auto setEnergyAndForceCutOff = [](auto &nonCoulombPair)
    {
        const auto &[energy, force] =
            nonCoulombPair->calculate(nonCoulombPair->getRadialCutOff());
        nonCoulombPair->setEnergyCutOff(energy);
        nonCoulombPair->setForceCutOff(force);
    };

    std::ranges::for_each(_nonCoulombPairsVector, setEnergyAndForceCutOff);
}

/**
 * @brief determines internal global van der Waals types and sets them in the
 * NonCoulombPair objects
 *
 * @param _externalToInternalGlobalVDWTypes
 *
 */
void ForceFieldNonCoulomb::determineInternalGlobalVdwTypes(
    const std::map<size_t, size_t> &externalToInternalGlobalVDWTypes
)
{
    auto setInternalVanDerWaalsType =
        [&externalToInternalGlobalVDWTypes](auto &nonCoulombPair)
    {
        nonCoulombPair->setInternalType1(externalToInternalGlobalVDWTypes.at(
            nonCoulombPair->getVanDerWaalsType1()
        ));
        nonCoulombPair->setInternalType2(externalToInternalGlobalVDWTypes.at(
            nonCoulombPair->getVanDerWaalsType2()
        ));
    };

    std::ranges::for_each(_nonCoulombPairsVector, setInternalVanDerWaalsType);
}

/**
 * @brief sorts the elements of a non-coulombic pairs vector
 *
 * @param nonCoulombicPairsVector
 *
 * @throw ParameterFileException if non-coulombic pairs with the same global van
 * der Waals types are defined twice
 */
void ForceFieldNonCoulomb::sortNonCoulombicsPairs(
    std::vector<std::shared_ptr<NonCoulombPair>> &nonCoulombicPairsVector
)
{
    auto isLess =
        [](const auto &nonCoulombicPair1, const auto &nonCoulombicPair2)
    {
        if (nonCoulombicPair1->getInternalType1() <
            nonCoulombicPair2->getInternalType1())
            return true;
        else if (nonCoulombicPair1->getInternalType1() ==
                 nonCoulombicPair2->getInternalType1())
            return nonCoulombicPair1->getInternalType2() <
                   nonCoulombicPair2->getInternalType2();
        else
            return false;
    };

    std::ranges::sort(nonCoulombicPairsVector, isLess);

    auto compareSharedPointers =
        [](const auto &nonCoulombicPair1, const auto &nonCoulombicPair2)
    { return *nonCoulombicPair1 == *nonCoulombicPair2; };

    const auto iter = std::ranges::adjacent_find(
        nonCoulombicPairsVector,
        compareSharedPointers
    );

    if (iter != nonCoulombicPairsVector.end())
        throw customException::ParameterFileException(std::format(
            "Non-coulombic pairs with global van der Waals types {} and {} in "
            "the parameter file are defined twice",
            (*iter)->getVanDerWaalsType1(),
            (*iter)->getVanDerWaalsType2()
        ));
}

/**
 * @brief fills the diagonal elements of the non-coulombic pairs matrix
 *
 * @param diagonalElements
 */
void ForceFieldNonCoulomb::fillDiagonalElementsOfNonCoulombPairsMatrix(
    std::vector<std::shared_ptr<NonCoulombPair>> &diagonalElements
)
{
    sortNonCoulombicsPairs(diagonalElements);

    _nonCoulombPairsMatrix =
        linearAlgebra::Matrix<std::shared_ptr<NonCoulombPair>>(
            diagonalElements.size()
        );

    for (size_t i = 0, numberOfDiagonalElements = diagonalElements.size();
         i < numberOfDiagonalElements;
         ++i)
        _nonCoulombPairsMatrix(i, i) = diagonalElements[i];
}

/**
 * @brief fills one off-diagonal element of the non-coulombic pairs matrix
 *
 * @details if no type is found check if mixing rules are used
 * if mixing rules are used, then add a new non-coulombic pair with the mixing
 * rule (TODO: not yet implemented) if no mixing rules are used, then throw an
 * exception if a type is found, then add the non-coulombic pair to the matrix
 * if a type is found with both index combinations and it has the same
 * parameters, then add the non-coulombic pair if a type is found with both
 * index combinations and it has different parameters, then throw an exception
 *
 * @param atomType1
 * @param atomType2
 *
 * @throws customException::ParameterFileException if mixing rules are not used
 * and not all combinations of global van der Waals types are defined in the
 * parameter file
 * @throws customException::ParameterFileException if type is found with both
 * index combinations and it has different parameters
 */
void ForceFieldNonCoulomb::setOffDiagonalElement(
    const size_t atomType1,
    const size_t atomType2
)
{
    auto nonCoulombicPair1 =
        findNonCoulombicPairByInternalTypes(atomType1, atomType2);
    auto nonCoulombicPair2 =
        findNonCoulombicPairByInternalTypes(atomType2, atomType1);

    if (nonCoulombicPair1 == std::nullopt && nonCoulombicPair2 == std::nullopt)
    {
        if (_mixingRule == MixingRule::NONE)
        {
            throw customException::ParameterFileException(
                "Not all combinations of global van der Waals types are "
                "defined in the parameter file - and no mixing "
                "rules were chosen"
            );
        }
    }
    else if (nonCoulombicPair1 != std::nullopt &&
             nonCoulombicPair2 != std::nullopt)
    {
        if (**nonCoulombicPair1 != **nonCoulombicPair2)
        {
            const auto vdwType1 = (*nonCoulombicPair1)->getVanDerWaalsType1();
            const auto vdwType2 = (*nonCoulombicPair1)->getVanDerWaalsType2();
            throw customException::ParameterFileException(std::format(
                "Non-coulombic pairs with global van der Waals types {}, {} "
                "and {}, {} in the parameter file have "
                "different parameters",
                vdwType1,
                vdwType2,
                vdwType2,
                vdwType1
            ));
        }

        _nonCoulombPairsMatrix(atomType1, atomType2) = *nonCoulombicPair1;
        _nonCoulombPairsMatrix(atomType2, atomType1) = *nonCoulombicPair1;
    }
    else if (nonCoulombicPair1 != std::nullopt)
    {
        _nonCoulombPairsMatrix(atomType1, atomType2) = *nonCoulombicPair1;
        _nonCoulombPairsMatrix(atomType2, atomType1) = *nonCoulombicPair1;
    }
    else
    {
        _nonCoulombPairsMatrix(atomType1, atomType2) = *nonCoulombicPair2;
        _nonCoulombPairsMatrix(atomType2, atomType1) = *nonCoulombicPair2;
    }
}

/**
 * @brief fills the off-diagonal elements of the non-coulombic pairs matrix
 *
 */
void ForceFieldNonCoulomb::fillOffDiagonalElementsOfNonCoulombPairsMatrix()
{
    const auto &[rows, cols] = _nonCoulombPairsMatrix.shape();

    for (size_t i = 0; i < rows; ++i)
        for (size_t j = i + 1; j < cols; ++j) setOffDiagonalElement(i, j);
}

/**
 * @brief finds all non-coulombic pairs that are self-interaction pairs
 *
 * @details self interacting means that both internal or external types are the
 * same
 *
 * @return std::vector<std::shared_ptr<NonCoulombPair>>
 */
std::vector<std::shared_ptr<NonCoulombPair>> ForceFieldNonCoulomb::
    getSelfInteractionNonCoulombicPairs() const
{
    auto isSelfInteractionElement = [](const auto &nonCoulombPair)
    {
        return nonCoulombPair->getInternalType1() ==
               nonCoulombPair->getInternalType2();
    };

    auto view =
        _nonCoulombPairsVector | std::views::filter(isSelfInteractionElement);

    return std::vector(view.begin(), view.end());
}

/**
 * @brief finds a non coulombic pair by internal types
 *
 * @details if the non coulombic pair is not found, an empty optional is
 * returned if the non coulombic pair is found twice an exception is thrown
 *
 * @param internalType1
 * @param internalType2
 * @return optional<std::shared_ptr<NonCoulombPair>>
 *
 * @throws if the non coulombic pair is found twice
 */
std::optional<std::shared_ptr<NonCoulombPair>> ForceFieldNonCoulomb::
    findNonCoulombicPairByInternalTypes(
        size_t internalType1,
        size_t internalType2
    ) const
{
    auto findByInternalAtomTypes =
        [internalType1, internalType2](const auto &nonCoulombPair)
    {
        return nonCoulombPair->getInternalType1() == internalType1 &&
               nonCoulombPair->getInternalType2() == internalType2;
    };

    if (auto firstNonCoulombicPair = std::ranges::find_if(
            _nonCoulombPairsVector,
            findByInternalAtomTypes
        );
        firstNonCoulombicPair != _nonCoulombPairsVector.end())
    {
        if (auto secondCoulombicPair = std::ranges::find_if(
                firstNonCoulombicPair + 1,
                _nonCoulombPairsVector.end(),
                findByInternalAtomTypes
            );
            secondCoulombicPair != _nonCoulombPairsVector.end())
        {
            auto vdwType1 = (*firstNonCoulombicPair)->getVanDerWaalsType1();
            auto vdwType2 = (*firstNonCoulombicPair)->getVanDerWaalsType2();
            throw customException::ParameterFileException(std::format(
                "Non coulombic pair with global van der waals types {} and {} "
                "is defined twice in the parameter file.",
                vdwType1,
                vdwType2
            ));
        }

        return *firstNonCoulombicPair;
    }
    else
        return std::nullopt;
}