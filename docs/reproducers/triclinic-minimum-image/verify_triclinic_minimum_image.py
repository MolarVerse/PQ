#!/usr/bin/env python3
import itertools
import math


def mat_vec_mul(matrix, vector):
    return [sum(matrix[row][col] * vector[col] for col in range(3)) for row in range(3)]


def determinant(matrix):
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def inverse(matrix):
    det = determinant(matrix)
    return [
        [
            (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1]) / det,
            (matrix[0][2] * matrix[2][1] - matrix[0][1] * matrix[2][2]) / det,
            (matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1]) / det,
        ],
        [
            (matrix[1][2] * matrix[2][0] - matrix[1][0] * matrix[2][2]) / det,
            (matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0]) / det,
            (matrix[0][2] * matrix[1][0] - matrix[0][0] * matrix[1][2]) / det,
        ],
        [
            (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0]) / det,
            (matrix[0][1] * matrix[2][0] - matrix[0][0] * matrix[2][1]) / det,
            (matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]) / det,
        ],
    ]


def cpp_round(value):
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def norm(vector):
    return math.sqrt(sum(component * component for component in vector))


def make_box_matrix(dimensions, angles_degrees):
    alpha, beta, gamma = [math.radians(angle) for angle in angles_degrees]
    cos_alpha, cos_beta, cos_gamma = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sin_gamma = math.sin(gamma)

    transform = [[0.0 for _ in range(3)] for _ in range(3)]
    transform[0][0] = 1.0
    transform[0][1] = cos_gamma
    transform[0][2] = cos_beta
    transform[1][1] = sin_gamma
    transform[1][2] = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    transform[2][2] = math.sqrt(
        1.0
        - cos_alpha * cos_alpha
        - cos_beta * cos_beta
        - cos_gamma * cos_gamma
        + 2.0 * cos_alpha * cos_beta * cos_gamma
    ) / sin_gamma

    return [[transform[row][col] * dimensions[col] for col in range(3)] for row in range(3)]


def current_calc_shift_vector(box_matrix, displacement):
    fractional = mat_vec_mul(inverse(box_matrix), displacement)
    rounded = [cpp_round(component) for component in fractional]
    return mat_vec_mul(box_matrix, rounded)


def brute_force_nearest_image(box_matrix, displacement, radius=3):
    best = displacement
    best_norm = norm(best)
    for image in itertools.product(range(-radius, radius + 1), repeat=3):
        candidate_shift = mat_vec_mul(box_matrix, image)
        candidate = [displacement[i] - candidate_shift[i] for i in range(3)]
        candidate_norm = norm(candidate)
        if candidate_norm < best_norm:
            best = candidate
            best_norm = candidate_norm
    return best, best_norm


def main():
    box_matrix = make_box_matrix((1.0, 2.0, 3.0), (30.0, 60.0, 45.0))
    fractional_displacement = (-0.99, -0.99, -0.51)
    displacement = mat_vec_mul(box_matrix, fractional_displacement)

    current_shift = current_calc_shift_vector(box_matrix, displacement)
    current_image = [displacement[i] - current_shift[i] for i in range(3)]
    current_norm = norm(current_image)

    brute_image, brute_norm = brute_force_nearest_image(box_matrix, displacement)

    print(f"current image norm: {current_norm:.12f}")
    print(f"brute-force image norm: {brute_norm:.12f}")
    print(f"current image: {current_image}")
    print(f"brute-force image: {brute_image}")

    if current_norm <= brute_norm + 1.0e-12:
        raise SystemExit("finding not reproduced: current image is not longer")

    print("finding reproduced: current triclinic minimum-image path picks a longer image")


if __name__ == "__main__":
    main()
