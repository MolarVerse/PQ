#include "setup.hpp"

#include "guffDatReader.hpp"
#include "inputFileReader.hpp"
#include "moldescriptorReader.hpp"
#include "postProcessSetup.hpp"
#include "rstFileReader.hpp"

#include <iostream>

using namespace std;
using namespace engine;

/**
 * @brief setup the engine
 *
 * @param filename
 * @param engine
 */
void setup::setupEngine(const string &filename, Engine &engine)
{
    cout << "Reading input file..." << endl;
    readInputFile(filename, engine);

    cout << "Reading moldescriptor..." << endl;
    readMolDescriptor(engine);

    cout << "Reading rst file..." << endl;
    readRstFile(engine);

    cout << "Post processing setup..." << endl;
    postProcessSetup(engine);

    cout << "Reading guff.dat..." << endl;
    setup::readGuffDat(engine);
}