import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class FEMSolver:
    def __init__(self, mesh_dict, f_func, bc_func):
        self.points = mesh_dict["points"]
        self.elements = mesh_dict["triangles"]
        self._f = f_func
        self._bc_func = bc_func
        self.u = np.zeros(len(self.points))
        self._K = None
        self._F = None

        self.boundary_edges, self.boundary_nodes = self._find_boundary()

    def _find_boundary(self):
        edges = {}
        for tri in self.elements:
            for i in range(3):
                edge = tuple(sorted([tri[i], tri[(i+1)%3]]))
                edges[edge] = edges.get(edge, 0) + 1
        bound_edges = [e for e, count in edges.items() if count == 1]
        bound_nodes = list(set([n for e in bound_edges for n in e]))
        return bound_edges, bound_nodes

    def _assemble(self):
        N = len(self.points)
        self._K = np.zeros((N, N))
        self._F = np.zeros(N)
        for e in self.elements:
            coords = self.points[e]
            Ke, Fe = self._local_matrices(coords)
            for i in range(3):
                I = e[i]
                self._F[I] += Fe[i]
                for j in range(3):
                    J = e[j]
                    self._K[I, J] += Ke[i, j]

    def _local_matrices(self, coords):
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        A = np.array([[1, x1, y1],
                      [1, x2, y2],
                      [1, x3, y3]])
        area = 0.5 * np.linalg.det(A)
        b = np.array([y2 - y3, y3 - y1, y1 - y2])
        c = np.array([x3 - x2, x1 - x3, x2 - x1])
        B = np.vstack([b, c]) / (2 * area)
        Ke = area * (B.T @ B)
        xc, yc = np.mean(coords, axis=0)
        fval = self._f(xc, yc)
        Fe = np.full(3, fval * area / 3.0)
        return Ke, Fe

    def _apply_neumann_edge(self, edge, func_val):
        p1, p2 = self.points[edge[0]], self.points[edge[1]]
        length = np.linalg.norm(p2 - p1)
        self._F[edge[0]] += func_val * length / 2
        self._F[edge[1]] += func_val * length / 2

    def _apply_dirichlet_node(self, node, value):
        self._F -= value * self._K[:, node]  
        self._K[node, :] = 0.0
        self._K[:, node] = 0.0
        self._K[node, node] = 1.0
        self._F[node] = value

    def _apply_boundary_conditions(self):
        for edge in self.boundary_edges:
            midpoint = np.mean(self.points[list(edge)], axis=0)
            bc_result = self._bc_func(midpoint[0], midpoint[1])
            if bc_result is not None and bc_result[0] == 'neumann':
                self._apply_neumann_edge(edge, bc_result[1])

        for node in self.boundary_nodes:
            x, y = self.points[node]
            bc_result = self._bc_func(x, y)
            if bc_result is not None and bc_result[0] == 'dirichlet':
                self._apply_dirichlet_node(node, bc_result[1])

    def solve(self):
        self._assemble()
        self._apply_boundary_conditions()
        self.u = np.linalg.solve(self._K, self._F)
        return self.u