#include "intraNonBondedReader.hpp"

using namespace readInput;

/**
 * @brief construct IntraNonBondedReader object and read the file
 *
 * @param engine
 */
void readInput::readIntraNonBondedFile(engine::Engine &engine)
{
    IntraNonBondedReader reader(engine.getSettings().getIntraNonBondedFilename(), engine);
    reader.read();
}

/**
 * @brief reads the intra non bonded interactions from the intraNonBonded file
 *
 */
void IntraNonBondedReader::read()
{
    if (!isNeeded())
        return;
}