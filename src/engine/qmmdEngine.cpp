#include "qmmdEngine.hpp"

using engine::QMMDEngine;

void QMMDEngine::takeStep() { _qmRunner->run(_simulationBox); }