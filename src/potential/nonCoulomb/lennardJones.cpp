#include "lennardJones.hpp"

#include "settings.hpp"

using namespace potential;
using namespace settings;

/**
 * @brief Construct a new Lennard Jones object
 *
 * @param size
 */
LennardJones::LennardJones(const size_t size)
    : _c6(size * size),
      _c12(size * size),
      _cutOff(size * size),
      _energyCutOff(size * size),
      _forceCutOff(size * size),
      _size(size)
{
}

/**
 * @brief Construct a new Lennard Jones object
 *
 */
LennardJones::LennardJones() { *this = LennardJones(0); }

/**
 * @brief Add a pair of Lennard-Jones types to the LennardJones object
 *
 * @param pair
 * @param index1
 * @param index2
 */
void LennardJones::addPair(
    const LennardJonesPair& pair,
    const size_t            index1,
    const size_t            index2
)
{
    _c6[index1 * _size + index2]           = pair.getC6();
    _c6[index2 * _size + index1]           = pair.getC6();
    _c12[index1 * _size + index2]          = pair.getC12();
    _c12[index2 * _size + index1]          = pair.getC12();
    _cutOff[index1 * _size + index2]       = pair.getRadialCutOff();
    _cutOff[index2 * _size + index1]       = pair.getRadialCutOff();
    _energyCutOff[index1 * _size + index2] = pair.getEnergyCutOff();
    _energyCutOff[index2 * _size + index1] = pair.getEnergyCutOff();
    _forceCutOff[index1 * _size + index2]  = pair.getForceCutOff();
    _forceCutOff[index2 * _size + index1]  = pair.getForceCutOff();
}

/**
 * @brief Get the C6 object
 *
 * @return const Real*
 */
const Real* LennardJones::getC6() const
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _c6Device;
    else
#endif
        return _c6.data();
}

/**
 * @brief Get the C12 object
 *
 * @return const Real*
 */
const Real* LennardJones::getC12() const
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _c12Device;
    else
#endif
        return _c12.data();
}

/**
 * @brief Get the CutOff object
 *
 * @return const Real*
 */
const Real* LennardJones::getCutOff() const
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _cutOffDevice;
    else
#endif
        return _cutOff.data();
}

/**
 * @brief Get the Energy CutOff object
 *
 * @return const Real*
 */
const Real* LennardJones::getEnergyCutOff() const
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _energyCutOffDevice;
    else
#endif
        return _energyCutOff.data();
}

/**
 * @brief Get the Force CutOff object
 *
 * @return const Real*
 */
const Real* LennardJones::getForceCutOff() const
{
#ifdef __PQ_GPU__
    if (Settings::useDevice())
        return _forceCutOffDevice;
    else
#endif
        return _forceCutOff.data();
}