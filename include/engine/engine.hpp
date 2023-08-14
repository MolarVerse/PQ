#ifndef _ENGINE_HPP_

#define _ENGINE_HPP_

#include "celllist.hpp"
#include "constraints.hpp"
#include "engineOutput.hpp"
#include "forceField.hpp"
#include "integrator.hpp"
#include "intraNonBonded.hpp"
#include "manostat.hpp"
#include "physicalData.hpp"
#include "potential.hpp"
#include "potential_new.hpp"
#include "resetKinetics.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "thermostat.hpp"
#include "timings.hpp"
#include "virial.hpp"

#include <memory>

namespace engine
{
    class Engine;
}

/**
 * @class Engine
 *
 * @brief Contains all the information needed to run the simulation
 *
 */
class engine::Engine
{
  private:
    size_t _step = 1;

    settings::Settings           _settings;
    timings::Timings             _timings;
    simulationBox::CellList      _cellList;
    simulationBox::SimulationBox _simulationBox;
    physicalData::PhysicalData   _physicalData;
    physicalData::PhysicalData   _averagePhysicalData;
    constraints::Constraints     _constraints;
    forceField::ForceField       _forceField;
    engine::EngineOutput         _engineOutput;

    std::unique_ptr<integrator::Integrator>         _integrator     = std::make_unique<integrator::VelocityVerlet>();
    std::unique_ptr<thermostat::Thermostat>         _thermostat     = std::make_unique<thermostat::Thermostat>();
    std::unique_ptr<manostat::Manostat>             _manostat       = std::make_unique<manostat::Manostat>();
    std::unique_ptr<virial::Virial>                 _virial         = std::make_unique<virial::VirialMolecular>();
    std::unique_ptr<resetKinetics::ResetKinetics>   _resetKinetics  = std::make_unique<resetKinetics::ResetKinetics>();
    std::unique_ptr<intraNonBonded::IntraNonBonded> _intraNonBonded = std::make_unique<intraNonBonded::IntraNonBondedGuff>();

    std::unique_ptr<potential::Potential>     _potential    = std::make_unique<potential::PotentialBruteForce>();
    std::unique_ptr<potential_new::Potential> _potentialNew = std::make_unique<potential_new::PotentialBruteForce>();   // TODO:

  public:
    void run();
    void takeStep();
    void writeOutput();

    [[nodiscard]] bool isForceFieldActivated() const { return _forceField.isActivated(); }
    [[nodiscard]] bool isForceFieldNonCoulombicsActivated() const { return _forceField.isNonCoulombicActivated(); }
    [[nodiscard]] bool isGuffActivated() const { return !_forceField.isNonCoulombicActivated(); }
    [[nodiscard]] bool isCellListActivated() const { return _cellList.isActivated(); }
    [[nodiscard]] bool isConstraintsActivated() const { return _constraints.isActivated(); }
    [[nodiscard]] bool isIntraNonBondedActivated() const { return _intraNonBonded->isActivated(); }

    /************************************
     *                                  *
     * standard make unique_ptr methods *
     *                                  *
     ************************************/

    template <typename T>
    void makeIntegrator(T integrator)
    {
        _integrator = std::make_unique<T>(integrator);
    }
    template <typename T>
    void makePotential(T potential)
    {
        _potential = std::make_unique<T>(potential);
    }
    template <typename T>
    void makePotentialNew(T potentialNew)   // TODO:
    {
        _potentialNew = std::make_unique<T>(potentialNew);
    }
    template <typename T>
    void makeThermostat(T thermostat)
    {
        _thermostat = std::make_unique<T>(thermostat);
    }
    template <typename T>
    void makeManostat(T manostat)
    {
        _manostat = std::make_unique<T>(manostat);
    }
    template <typename T>
    void makeVirial(T virial)
    {
        _virial = std::make_unique<T>(virial);
    }
    template <typename T>
    void makeResetKinetics(T resetKinetics)
    {
        _resetKinetics = std::make_unique<T>(resetKinetics);
    }
    template <typename T>
    void makeIntraNonBonded(T intraNonBonded)
    {
        _intraNonBonded = std::make_unique<T>(intraNonBonded);
    }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    [[nodiscard]] settings::Settings           &getSettings() { return _settings; }
    [[nodiscard]] timings::Timings             &getTimings() { return _timings; }
    [[nodiscard]] simulationBox::CellList      &getCellList() { return _cellList; }
    [[nodiscard]] simulationBox::SimulationBox &getSimulationBox() { return _simulationBox; }
    [[nodiscard]] physicalData::PhysicalData   &getPhysicalData() { return _physicalData; }
    [[nodiscard]] physicalData::PhysicalData   &getAveragePhysicalData() { return _averagePhysicalData; }
    [[nodiscard]] constraints::Constraints     &getConstraints() { return _constraints; }
    [[nodiscard]] forceField::ForceField       &getForceField() { return _forceField; }

    [[nodiscard]] virial::Virial                 &getVirial() { return *_virial; }
    [[nodiscard]] integrator::Integrator         &getIntegrator() { return *_integrator; }
    [[nodiscard]] potential::Potential           &getPotential() { return *_potential; }
    [[nodiscard]] potential_new::Potential       &getPotentialNew() { return *_potentialNew; }   // TODO:
    [[nodiscard]] thermostat::Thermostat         &getThermostat() { return *_thermostat; }
    [[nodiscard]] manostat::Manostat             &getManostat() { return *_manostat; }
    [[nodiscard]] resetKinetics::ResetKinetics   &getResetKinetics() { return *_resetKinetics; }
    [[nodiscard]] intraNonBonded::IntraNonBonded &getIntraNonBonded() { return *_intraNonBonded; }

    [[nodiscard]] engine::EngineOutput     &getEngineOutput() { return _engineOutput; }
    [[nodiscard]] output::EnergyOutput     &getEnergyOutput() { return _engineOutput.getEnergyOutput(); }
    [[nodiscard]] output::TrajectoryOutput &getXyzOutput() { return _engineOutput.getXyzOutput(); }
    [[nodiscard]] output::TrajectoryOutput &getVelOutput() { return _engineOutput.getVelOutput(); }
    [[nodiscard]] output::TrajectoryOutput &getForceOutput() { return _engineOutput.getForceOutput(); }
    [[nodiscard]] output::TrajectoryOutput &getChargeOutput() { return _engineOutput.getChargeOutput(); }
    [[nodiscard]] output::LogOutput        &getLogOutput() { return _engineOutput.getLogOutput(); }
    [[nodiscard]] output::StdoutOutput     &getStdoutOutput() { return _engineOutput.getStdoutOutput(); }
    [[nodiscard]] output::RstFileOutput    &getRstFileOutput() { return _engineOutput.getRstFileOutput(); }
    [[nodiscard]] output::InfoOutput       &getInfoOutput() { return _engineOutput.getInfoOutput(); }

    [[nodiscard]] forceField::ForceField *getForceFieldPtr() { return &_forceField; }
};

#endif   // _ENGINE_HPP_