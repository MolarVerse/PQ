#ifndef _SETUP_HPP_

#define _SETUP_HPP_

#include "engine.hpp"

#include <string>

/**
 * @namespace setup
 *
 * @note
 *  This namespace contains all the functions that are used to setup the
 *  simulation. This includes reading the input file, the moldescriptor,
 *  the rst file, the guff.dat file, and post processing the setup.
 *
 */
namespace setup
{
    void setupEngine(const std::string &, engine::Engine &);
}

#endif   // _SETUP_HPP_