#ifndef _RST_FILE_SECTION_H_

#define _RST_FILE_SECTION_H_

#include <string>
#include <vector>

#include "simulationBox.hpp"
#include "settings.hpp"

namespace Setup::RstFileReader
{
    class RstFileSection
    {
    public:
        virtual ~RstFileSection() = default;

        int _lineNumber;
        virtual std::string keyword() = 0;
        virtual bool isHeader() = 0;
        virtual void process(std::vector<std::string> &, Settings &, SimulationBox &) = 0;
    };

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

    class NoseHooverSection : public RstFileSection
    {
    public:
        std::string keyword() override;
        bool isHeader() override;
        void process(std::vector<std::string> &, Settings &, SimulationBox &) override;
    };

    class StepCountSection : public RstFileSection
    {
    public:
        std::string keyword() override;
        bool isHeader() override;
        void process(std::vector<std::string> &, Settings &, SimulationBox &) override;
    };

    class AtomSection : public RstFileSection
    {
    public:
        std::string keyword() override;
        bool isHeader() override;
        void process(std::vector<std::string> &, Settings &, SimulationBox &) override;
    };
}

#endif