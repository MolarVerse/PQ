#ifndef _NON_COULOMB_PAIR_HPP_

#define _NON_COULOMB_PAIR_HPP_

#include <cstddef>   // for size_t
#include <utility>   // for pair

namespace potential
{
    /**
     * @class NonCoulombPair
     *
     * @brief base class representing a pair of non-coulombic types
     *
     * @details constructor with van der Waals types and cut-off radius is for force field parameters
     *         constructor with cut-off radius only is for guff representation
     *
     */
    class NonCoulombPair
    {
      protected:
        size_t _vanDerWaalsType1 = 0;
        size_t _vanDerWaalsType2 = 0;
        size_t _internalType1    = 0;
        size_t _internalType2    = 0;

        double _radialCutOff;
        double _energyCutOff = 0.0;
        double _forceCutOff  = 0.0;

      public:
        explicit NonCoulombPair(const size_t vanDerWaalsType1, const size_t vanDerWaalsType2, const double cutOff)
            : _vanDerWaalsType1(vanDerWaalsType1), _vanDerWaalsType2(vanDerWaalsType2), _radialCutOff(cutOff){};

        explicit NonCoulombPair(const double cutOff) : _radialCutOff(cutOff){};

        explicit NonCoulombPair(const double cutoff, const double energyCutoff, const double forceCutoff)
            : _radialCutOff(cutoff), _energyCutOff(energyCutoff), _forceCutOff(forceCutoff){};

        virtual ~NonCoulombPair() = default;

        [[nodiscard]] bool operator==(const NonCoulombPair &other) const;

        [[nodiscard]] virtual std::pair<double, double> calculateEnergyAndForce(const double distance) const = 0;

        void setInternalType1(const size_t internalType1) { _internalType1 = internalType1; }
        void setInternalType2(const size_t internalType2) { _internalType2 = internalType2; }

        void setEnergyCutOff(const double energyCutoff) { _energyCutOff = energyCutoff; }
        void setForceCutOff(const double forceCutoff) { _forceCutOff = forceCutoff; }

        [[nodiscard]] size_t getVanDerWaalsType1() const { return _vanDerWaalsType1; }
        [[nodiscard]] size_t getVanDerWaalsType2() const { return _vanDerWaalsType2; }
        [[nodiscard]] size_t getInternalType1() const { return _internalType1; }
        [[nodiscard]] size_t getInternalType2() const { return _internalType2; }
        [[nodiscard]] double getRadialCutOff() const { return _radialCutOff; }
    };

}   // namespace potential

#endif   // _NON_COULOMB_PAIR_HPP_