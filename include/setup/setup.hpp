#ifndef _SETUP_HPP_

#define _SETUP_HPP_

#include <string>   // for string

namespace engine
{
    class Engine;
}   // namespace engine

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
    void readFiles(const std::string &inputFileName, engine::Engine &);
    void setupEngine(engine::Engine &);
    void setupSimulation(const std::string &inputFileName, engine::Engine &);
}   // namespace setup

#endif   // _SETUP_HPP_