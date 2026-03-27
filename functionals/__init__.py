from .operators import gradient, laplacian, normal_derivative
from .integrals import domain_integral, boundary_integral
from .errors import energy_error, relative_l2_error, relative_energy_error

__all__ = [
    "gradient",
    "laplacian",
    "normal_derivative",
    "domain_integral",
    "boundary_integral",
    "energy_error",
    "relative_l2_error",
    "relative_energy_error",
]
