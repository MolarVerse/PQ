#ifndef _RST_FILE_READER_H_

#define _RST_FILE_READER_H_

#include <string>
#include <vector>
#include <fstream>

#include "simulationBox.hpp"
#include "engine.hpp"
#include "rstFileSection.hpp"

namespace Setup::RstFileReader
{
    /**
     * @class RstFileReader
     *
     * @brief Reads a .rst file and returns a SimulationBox object
     *
     */
    class RstFileReader
    {
    private:
        const std::string _filename;
        std::ifstream _fp;
        Engine &_engine;
        std::vector<RstFileSection *> _sections;
        RstFileSection *_atomSection = new AtomSection;

    public:
        RstFileReader(const std::string &, Engine &);
        ~RstFileReader();

        void read();
        RstFileSection *determineSection(std::vector<std::string> &);
    };
}

/**
 * @brief Reads a .rst file and returns a SimulationBox object
 *
 * @param engine
 */
void read_rst(Engine &);

#endif