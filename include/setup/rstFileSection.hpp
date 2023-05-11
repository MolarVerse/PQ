#ifndef _RST_FILE_SECTION_H_

#define _RST_FILE_SECTION_H_

#include <string>
#include <vector>
#include <fstream>

#include "simulationBox.hpp"
#include "settings.hpp"
#include "engine.hpp"

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
        std::ifstream *_fp;
        virtual std::string keyword() = 0;
        virtual bool isHeader() = 0;
        virtual void process(std::vector<std::string> &, Engine &) = 0;
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
        std::string keyword() override { return "box"; };
        bool isHeader() override;
        void process(std::vector<std::string> &, Engine &) override;
    };

    /**
     * @class NoseHooverSection
     *
     * @brief Reads the Nose-Hoover section of a .rst file
     *
     */
    class NoseHooverSection : public RstFileSection
    {
    public:
        std::string keyword() override { return "chi"; };
        bool isHeader() override;
        void process(std::vector<std::string> &, Engine &) override;
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
        std::string keyword() override { return "step"; };
        bool isHeader() override;
        void process(std::vector<std::string> &, Engine &) override;
    };

    /**
     * @class AtomSection
     *
     * @brief Reads the atom section of a .rst file
     *
     */
    class AtomSection : public RstFileSection
    {
    private:
        void processAtomLine(std::vector<std::string> &, Molecule &);

    public:
        std::string keyword() override { return ""; }
        bool isHeader() override;
        void process(std::vector<std::string> &, Engine &) override;
    };
}

#endif