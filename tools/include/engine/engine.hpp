#ifndef _TOOLENGINE_HPP_

#define _TOOLENGINE_HPP_

#include <string>
#include <memory>

#include "analysisRunner.hpp"
class Engine
{
private:
    std::string _executableName;
    std::string _inputFilename;

    std::vector<std::unique_ptr<AnalysisRunner>> _analysisRunners;

public:
    Engine(const std::string_view &executableName, const std::string_view &inputFilename) : _executableName(executableName), _inputFilename(inputFilename){};

    void run();
};

#endif // _ENGINE_HPP_