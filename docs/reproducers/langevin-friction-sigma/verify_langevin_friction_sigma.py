#!/usr/bin/env python3
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src/thermostat/langevinThermostat.cpp"
TEST_SOURCE = ROOT / "tests/src/thermostat/testThermostat.cpp"


def require(condition, message):
    if not condition:
        raise SystemExit(f"finding not reproduced: {message}")


def extract_function_body(source_text, function_name):
    start = source_text.find(function_name)
    require(start != -1, f"could not locate {function_name}")
    open_brace = source_text.find("{", start)
    require(open_brace != -1, f"could not locate opening brace for {function_name}")

    depth = 0
    for index in range(open_brace, len(source_text)):
        if source_text[index] == "{":
            depth += 1
        elif source_text[index] == "}":
            depth -= 1
            if depth == 0:
                return source_text[open_brace + 1 : index]

    raise SystemExit(f"finding not reproduced: could not parse {function_name}")


def sigma_like(friction, target_temperature, time_step):
    return math.sqrt(4.0 * friction * target_temperature / time_step)


def main():
    source_text = SOURCE.read_text()
    test_text = TEST_SOURCE.read_text()

    calculate_body = extract_function_body(source_text, "LangevinThermostat::calculateSigma")
    set_temperature_body = extract_function_body(source_text, "LangevinThermostat::setTargetTemperature")
    set_friction_body = extract_function_body(source_text, "LangevinThermostat::setFriction")

    require("4.0 * friction" in calculate_body, "sigma formula no longer depends on friction")
    require("calculateSigma(_friction, targetTemperature)" in set_temperature_body, "target-temperature setter no longer recomputes sigma")
    require("_friction = friction" in set_friction_body, "friction setter no longer updates _friction")
    require("calculateSigma" not in set_friction_body, "friction setter now recomputes sigma")
    require("langevin_setTargetTemperatureRecomputesSigma" in test_text, "temperature setter regression test missing")
    require("langevin.setFriction(0.5)" in test_text, "friction setter test changed")

    sigma_at_01 = sigma_like(0.1, 300.0, 0.1)
    sigma_at_05 = sigma_like(0.5, 300.0, 0.1)
    print(f"sigma-like value at friction 0.1: {sigma_at_01:.12f}")
    print(f"sigma-like value at friction 0.5: {sigma_at_05:.12f}")
    require(sigma_at_05 > sigma_at_01, "formula demonstration did not change with friction")

    print("finding reproduced: setFriction changes friction without recomputing friction-dependent sigma")


if __name__ == "__main__":
    main()
