#include "engine.hpp"
#include "trajectoryToCom.hpp"

using namespace std;

void Engine::run()
{
    // TODO: add input file reading
    _analysisRunners.push_back(make_unique<TrajToCom>());
}