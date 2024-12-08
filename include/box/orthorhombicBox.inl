#ifndef __ORTHORHOMBIC_BOX_INL__
#define __ORTHORHOMBIC_BOX_INL__

#include "orthorhombicBox.hpp"

#ifdef __PQ_DEBUG__
#include "debug.hpp"
#endif

namespace simulationBox
{

    /**
     * @brief image orthorhombic
     *
     * @param boxDimensions
     * @param x
     * @param y
     * @param z
     * @param tx
     * @param ty
     * @param tz
     */
    static inline void imageOrthoRhombic(
        const Real* const boxDimensions,
        Real&             x,
        Real&             y,
        Real&             z,
        Real&             tx,
        Real&             ty,
        Real&             tz
    )
    {
#ifdef __PQ_DEBUG__
        if (config::Debug::useDebug(config::DebugLevel::BOX_DEBUG))
        {
            std::cout << std::format(
                "Orthorhombic box: x = {}, y = {}, z = {}\n",
                boxDimensions[0],
                boxDimensions[1],
                boxDimensions[2]
            );
        }
#endif
        const auto boxX = boxDimensions[0];
        const auto boxY = boxDimensions[1];
        const auto boxZ = boxDimensions[2];

        tx = -boxX * ::round(x / boxX);
        ty = -boxY * ::round(y / boxY);
        tz = -boxZ * ::round(z / boxZ);

        x += tx;
        y += ty;
        z += tz;
    }

}   // namespace simulationBox

#endif   // __ORTHORHOMBIC_BOX_INL__