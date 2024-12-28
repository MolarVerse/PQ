#ifndef __SETUP_NON_LEGACY_INL__
#define __SETUP_NON_LEGACY_INL__

#include "buckingham.hpp"
#include "buckinghamPair.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "lennardJones.hpp"
#include "lennardJonesPair.hpp"
#include "morse.hpp"
#include "morsePair.hpp"
#include "potential.hpp"
#include "setup.hpp"

/**
 * @brief setup the flattened Lennard-Jones potential
 *
 * @tparam T the type of the potential
 * @param potential the potential to setup
 */
template <typename T>
void setup::setupFlattenedNonCoulPot(potential::Potential *const pot)
{
    auto &nonCoulPot = dynamic_cast<potential::ForceFieldNonCoulomb &>(
        pot->getNonCoulombPotential()
    );

    auto &nonCoulPairs    = nonCoulPot.getNonCoulombPairsMatrix();
    const auto [row, col] = nonCoulPairs.shape();

    assert(row == col);

    auto tempPot = T();
    tempPot.resize(row);

    // clang-format off
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < row; ++i)
            for (size_t j = 0; j < col; ++j)
            {
                if constexpr (std::is_same_v<T, potential::LennardJones>)
                {
                    const auto &pair =
                        dynamic_cast<potential::LennardJonesPair &>(
                            *nonCoulPairs(i, j)
                        );
                    tempPot.addPair(pair, i, j);
                }
                else if constexpr (std::is_same_v<T, potential::Buckingham>)
                {
                    const auto &pair =
                        dynamic_cast<potential::BuckinghamPair &>(
                            *nonCoulPairs(i, j)
                        );
                    tempPot.addPair(pair, i, j);
                }
                else if constexpr (std::is_same_v<T, potential::Morse>)
                {
                    const auto &pair = dynamic_cast<potential::MorsePair &>(
                        *nonCoulPairs(i, j)
                    );
                    tempPot.addPair(pair, i, j);
                }
                else
                    static_assert(false, "Unknown potential type");
            }
    // clang-format on

    const auto params  = tempPot.copyParams();
    const auto cutOffs = tempPot.copyCutOffs();
    const auto size    = tempPot.getSize();
    const auto nParams = T::getNParams();

    pot->setNonCoulombParamVectors(params, cutOffs, nParams, size);
}

#endif   // __SETUP_NON_LEGACY_INL__
