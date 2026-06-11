from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ManifoldMesh:
    vertices: np.ndarray
    faces: np.ndarray
    face_areas: np.ndarray
    quadrature_points: np.ndarray
    quadrature_weights: np.ndarray
    embedded_quadrature_points: np.ndarray


def circle_to_embedding(theta: np.ndarray) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    return np.column_stack([np.cos(theta), np.sin(theta)])


def embedding_to_circle(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.mod(np.arctan2(x[:, 1], x[:, 0]), 2.0 * np.pi).reshape(-1, 1)


def sphere_to_embedding(theta_phi: np.ndarray) -> np.ndarray:
    theta_phi = np.asarray(theta_phi, dtype=np.float64)
    theta = theta_phi[:, 0]
    phi = theta_phi[:, 1]
    sin_theta = np.sin(theta)
    return np.column_stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)]
    )


def embedding_to_sphere(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    unit = x / np.maximum(norm, 1e-12)
    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    phi = np.mod(np.arctan2(unit[:, 1], unit[:, 0]), 2.0 * np.pi)
    return np.column_stack([theta, phi])


def intrinsic_to_embedding(points: np.ndarray, manifold_type: str) -> np.ndarray:
    if manifold_type == "circle":
        return circle_to_embedding(points.reshape(-1))
    if manifold_type == "sphere":
        return sphere_to_embedding(points)
    raise ValueError(f"Unknown manifold type: {manifold_type}")


def embedding_to_intrinsic(points: np.ndarray, manifold_type: str) -> np.ndarray:
    if manifold_type == "circle":
        return embedding_to_circle(points)
    if manifold_type == "sphere":
        return embedding_to_sphere(points)
    raise ValueError(f"Unknown manifold type: {manifold_type}")


def sample_uniform_circle(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return theta.reshape(-1, 1), circle_to_embedding(theta)


def sample_uniform_sphere(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    u = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    theta = np.arccos(np.clip(u, -1.0, 1.0))
    coords = np.column_stack([theta, phi])
    return coords, sphere_to_embedding(coords)


def circle_geodesic(theta_a: np.ndarray, theta_b: np.ndarray) -> np.ndarray:
    a = np.asarray(theta_a).reshape(-1, 1)
    b = np.asarray(theta_b).reshape(1, -1)
    diff = np.abs(a - b)
    return np.minimum(diff, 2.0 * np.pi - diff)


def sphere_geodesic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    y = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
    dots = np.clip(x @ y.T, -1.0, 1.0)
    return np.arccos(dots)


def make_circle_mesh(n_segments: int = 256) -> ManifoldMesh:
    theta = (np.arange(n_segments) + 0.5) * (2.0 * np.pi / n_segments)
    vertices = circle_to_embedding(np.linspace(0, 2.0 * np.pi, n_segments, endpoint=False))
    weights = np.full(n_segments, 2.0 * np.pi / n_segments, dtype=np.float64)
    return ManifoldMesh(
        vertices=vertices,
        faces=np.empty((0, 2), dtype=np.int64),
        face_areas=weights.copy(),
        quadrature_points=theta.reshape(-1, 1),
        quadrature_weights=weights,
        embedded_quadrature_points=circle_to_embedding(theta),
    )


def make_sphere_mesh(n_theta: int = 32, n_phi: int = 64) -> ManifoldMesh:
    # Equal-area midpoint quadrature: u=cos(theta) is uniform on [-1, 1].
    u_edges = np.linspace(-1.0, 1.0, n_theta + 1)
    u_mid = 0.5 * (u_edges[:-1] + u_edges[1:])
    theta_mid = np.arccos(np.clip(u_mid, -1.0, 1.0))
    phi_mid = (np.arange(n_phi) + 0.5) * (2.0 * np.pi / n_phi)
    theta_grid, phi_grid = np.meshgrid(theta_mid, phi_mid, indexing="ij")
    coords = np.column_stack([theta_grid.ravel(), phi_grid.ravel()])
    weights = np.full(coords.shape[0], 4.0 * np.pi / coords.shape[0], dtype=np.float64)

    theta_vertices = np.linspace(0.0, np.pi, n_theta + 1)
    phi_vertices = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    tv, pv = np.meshgrid(theta_vertices, phi_vertices, indexing="ij")
    vertices = sphere_to_embedding(np.column_stack([tv.ravel(), pv.ravel()]))
    faces = []
    for i in range(n_theta):
        for j in range(n_phi):
            a = i * n_phi + j
            b = i * n_phi + ((j + 1) % n_phi)
            c = (i + 1) * n_phi + j
            d = (i + 1) * n_phi + ((j + 1) % n_phi)
            faces.append((a, c, b))
            faces.append((b, c, d))
    faces_arr = np.asarray(faces, dtype=np.int64)
    return ManifoldMesh(
        vertices=vertices,
        faces=faces_arr,
        face_areas=np.full(faces_arr.shape[0], 4.0 * np.pi / faces_arr.shape[0]),
        quadrature_points=coords,
        quadrature_weights=weights,
        embedded_quadrature_points=sphere_to_embedding(coords),
    )
