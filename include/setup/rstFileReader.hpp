#ifndef _RST_FILE_READER_H_

#define _RST_FILE_READER_H_

#include <string>
#include <memory>
#include <vector>

#include "simulationBox.hpp"
#include "settings.hpp"
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
        Settings &_settings;
        std::vector<RstFileSection *> _sections;
        RstFileSection *_atomSection = new AtomSection;

    public:
        RstFileReader(const std::string &, Settings &);
        ~RstFileReader();

        std::unique_ptr<SimulationBox> read();
        RstFileSection *determineSection(std::vector<std::string> &);
    };
}

/**
 * @brief Reads a .rst file and returns a SimulationBox object
 * 
 * @return std::unique_ptr<SimulationBox> 
 */
std::unique_ptr<SimulationBox> read_rst(const std::string &, Settings &);

#endif