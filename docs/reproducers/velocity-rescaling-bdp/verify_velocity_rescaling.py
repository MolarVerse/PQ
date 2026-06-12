#!/usr/bin/env python3
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src/thermostat/velocityRescalingThermostat.cpp"
DOCS = ROOT / "docs/sphinx/src/userGuide/inputFile.rst"


def require(condition, message):
    if not condition:
        raise SystemExit(f"finding not reproduced: {message}")


def extract_apply_body(source_text):
    match = re.search(
        r"void VelocityRescalingThermostat::applyThermostat\([^)]*\)\s*\{(?P<body>.*)\n\}",
        source_text,
        re.DOTALL,
    )
    require(match is not None, "could not locate applyThermostat body")
    return match.group("body")


def current_zero_temperature_lambda():
    target_temperature = 300.0
    current_temperature = 0.0
    time_step = 0.1
    tau = 100.0
    dof = 3.0

    temp_ratio = math.inf if current_temperature == 0.0 else target_temperature / current_temperature
    lambda_value = 1.0 + time_step / tau * (temp_ratio - 1.0)
    rescaling_factor = 2.0 * math.sqrt(time_step * temp_ratio / (dof * tau))
    return temp_ratio, lambda_value, rescaling_factor


def main():
    source_text = SOURCE.read_text()
    docs_text = DOCS.read_text()
    body = extract_apply_body(source_text)

    require("tempRatio = _targetTemperature / _temperature" in body, "temperature ratio division changed")
    require("while (lambda < 0.0)" in body, "negative-lambda rejection loop missing")
    require("std::gamma_distribution" not in source_text, "gamma/chi-square distribution is present")
    require("chi" not in body.lower(), "chi-square term appears to be present")
    require("Bussi-Donadio-Parrinello" in docs_text, "docs no longer advertise BDP thermostat")
    require("Enforces a canonical kinetic energy distribution" in docs_text, "docs no longer claim canonical sampling")

    temp_ratio, lambda_value, rescaling_factor = current_zero_temperature_lambda()
    print(f"zero-temperature tempRatio: {temp_ratio}")
    print(f"zero-temperature lambda: {lambda_value}")
    print(f"zero-temperature stochastic prefactor: {rescaling_factor}")

    require(not math.isfinite(temp_ratio), "zero-temperature ratio stayed finite")
    require(not math.isfinite(lambda_value), "zero-temperature lambda stayed finite")
    require(not math.isfinite(rescaling_factor), "zero-temperature stochastic prefactor stayed finite")

    print("finding reproduced: velocity_rescaling is the simplified documented formula and has a zero-temperature non-finite path")


if __name__ == "__main__":
    main()
