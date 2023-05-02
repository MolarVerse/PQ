#ifndef _RST_FILE_SECTION_H_

#define _RST_FILE_SECTION_H_

#include <string>
#include <vector>

#include "simulationBox.hpp"
#include "settings.hpp"

namespace Setup::RstFileReader
{
    /**
     * @class RstFileSection
     * 
     * @brief Base class for all sections of a .rst file
     * 
     */
    class RstFileSection
    {
    public:
        virtual ~RstFileSection() = default;

        int _lineNumber;
        virtual std::string keyword() = 0;
        virtual bool isHeader() = 0;
        virtual void process(std::vector<std::string> &, Settings &, SimulationBox &) = 0;
    };

    /**
     * @class BoxSection
     * 
     * @brief Reads the box section of a .rst file
     * 
     */
    class BoxSection : public RstFileSection
    {
    public:
        std::string keyword() override;
        bool isHeader() override;
        void process(std::vector<std::string> &, Settings &, SimulationBox &) override;
    };

    // class CellSection : public RstFileSection
    // {
    // public:
    //     std::string keyword() override;
    //     bool isHeader() override;
    //     void process(std::vector<std::string>, SimulationBox &) override;
    // };

    /**
     * @class NoseHooverSection
     * 
     * @brief Reads the Nose-Hoover section of a .rst file
     * 
     */
    class NoseHooverSection : public RstFileSection
    {
    public:
        std::string keyword() override;
        bool isHeader() override;
        void process(std::vector<std::string> &, Settings &, SimulationBox &) override;
    };

    /**
     * @class StepCountSection
     * 
     * @brief Reads the step count section of a .rst file
     * 
     */
    class StepCountSection : public RstFileSection
    {
    public:
        std::string keyword() override;
        bool isHeader() override;
        void process(std::vector<std::string> &, Settings &, SimulationBox &) override;
    };

    /**
     * @class AtomSection
     * 
     * @brief Reads the atom section of a .rst file
     * 
     */
    class AtomSection : public RstFileSection
    {
    public:
        std::string keyword() override;
        bool isHeader() override;
        void process(std::vector<std::string> &, Settings &, SimulationBox &) override;
    };
}

#endif