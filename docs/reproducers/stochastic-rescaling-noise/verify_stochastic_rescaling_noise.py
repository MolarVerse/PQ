#!/usr/bin/env python3
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "src/manostat/stochasticRescalingManostat.cpp"


FUNCTIONS = {
    "semi-isotropic": (
        "SemiIsotropicStochasticRescalingManostat::calculateMu",
        ("stochasticFactor_xy", "stochasticFactor_z"),
    ),
    "anisotropic": (
        "AnisotropicStochasticRescalingManostat::calculateMu",
        ("diagonalMatrix", "deltaP"),
    ),
    "full-anisotropic": (
        "FullAnisotropicStochasticRescalingManostat::calculateMu",
        ("expPade", "deltaP"),
    ),
}


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


def main():
    source_text = SOURCE.read_text()

    for label, (function_name, component_markers) in FUNCTIONS.items():
        body = extract_function_body(source_text, function_name)
        random_draws = body.count("getNormalDistribution")
        print(f"{label}: getNormalDistribution calls = {random_draws}")
        require(random_draws == 1, f"{label} no longer uses exactly one Gaussian draw")
        require("const auto random" in body, f"{label} no longer stores the Gaussian draw as random")
        for marker in component_markers:
            require(marker in body, f"{label} marker {marker!r} not found")

    semi_body = extract_function_body(source_text, FUNCTIONS["semi-isotropic"][0])
    require(
        re.search(r"stochasticFactor_xy\s*=.*\* random", semi_body, re.DOTALL),
        "semi-isotropic xy factor no longer uses the shared random draw",
    )
    require(
        re.search(r"stochasticFactor_z\s*=.*\* random", semi_body, re.DOTALL),
        "semi-isotropic z factor no longer uses the shared random draw",
    )

    print("finding reproduced: anisotropic stochastic cell-rescaling paths reuse one Gaussian draw across multiple components")


if __name__ == "__main__":
    main()
