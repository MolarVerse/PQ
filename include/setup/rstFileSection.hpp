#ifndef _RST_FILE_SECTION_H_

#define _RST_FILE_SECTION_H_

#include <string>
#include <vector>

#include "simulationBox.hpp"
#include "settings.hpp"

namespace Setup::RstFileReader
{
    using namespace std;

    class RstFileSection
    {
    public:
        int _lineNumber;
        virtual string keyword() = 0;
        virtual bool isHeader() = 0;
        virtual void process(vector<string>, Settings &, SimulationBox &) = 0;
    };

    class BoxSection : public RstFileSection
    {
    public:
        string keyword() override;
        bool isHeader() override;
        void process(vector<string>, Settings &, SimulationBox &) override;
    };

    // class CellSection : public RstFileSection
    // {
    // public:
    //     string keyword() override;
    //     bool isHeader() override;
    //     void process(vector<string>, SimulationBox &) override;
    // };

    class NoseHooverSection : public RstFileSection
    {
    public:
        string keyword() override;
        bool isHeader() override;
        void process(vector<string>, Settings &, SimulationBox &) override;
    };

    class StepCountSection : public RstFileSection
    {
    public:
        string keyword() override;
        bool isHeader() override;
        void process(vector<string>, Settings &, SimulationBox &) override;
    };

    class AtomSection : public RstFileSection
    {
    public:
        string keyword() override;
        bool isHeader() override;
        void process(vector<string>, Settings &, SimulationBox &) override;
    };
}

#endif