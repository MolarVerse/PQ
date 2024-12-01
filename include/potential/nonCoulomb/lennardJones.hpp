#include <vector>   // vector

#include "lennardJonesPair.hpp"
#include "typeAliases.hpp"

namespace potential
{
    class LennardJones
    {
       private:
        std::vector<Real> _c6;
        std::vector<Real> _c12;
        std::vector<Real> _cutOff;
        std::vector<Real> _energyCutOff;
        std::vector<Real> _forceCutOff;

        size_t _size;

#ifdef __PQ_GPU__
        Real* _c6Device;
        Real* _c12Device;
        Real* _cutOffDevice;
        Real* _energyCutOffDevice;
        Real* _forceCutOffDevice;
#endif

       public:
        LennardJones();
        explicit LennardJones(const size_t size);

        void addPair(
            const LennardJonesPair& pair,
            const size_t            index1,
            const size_t            index2
        );

        const Real* getC6() const;
        const Real* getC12() const;
        const Real* getCutOff() const;
        const Real* getEnergyCutOff() const;
        const Real* getForceCutOff() const;
    };
}   // namespace potential