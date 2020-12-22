import numpy as np


def get_phase_new(in_phase: np.ndarray, quad_phase: np.ndarray) -> np.ndarray:
    if len(in_phase.shape) > 1 or len(quad_phase.shape) > 1:
        raise IndexError("get_phase error: the dimension of in_phase or quadrature_phase is not one, but %s, %s" % (
            in_phase.shape, quad_phase.shape))
    delta_in_phase = np.diff(in_phase)
    delta_quad_phase = np.diff(quad_phase)
    pow_in_phase = np.power(2, in_phase)[1:]
    pow_quad_phase = np.power(2, quad_phase)[1:]
    delta_phase = (delta_quad_phase * in_phase[1:] - delta_in_phase * quad_phase[1:]) / (pow_quad_phase + pow_in_phase)
    phase = np.cumsum(delta_phase)
    return phase


def get_phase(in_phase: np.ndarray, quadrature_phase: np.ndarray) -> np.ndarray:
    if len(in_phase.shape) > 1 or len(quadrature_phase.shape) > 1:
        raise IndexError("get_phase error: the dimension of in_phase or quadrature_phase is not one, but %s, %s" % (
            in_phase.shape, quadrature_phase.shape))
    # c_phase = list(map(lambda iq: np.complex(iq[0], iq[1]), zip(in_phase, quadrature_phase)))
    c_phase = in_phase + 1j * quadrature_phase
    return np.angle(c_phase)


def get_amplitude(in_phase: np.ndarray, quadrature_phase: np.ndarray) -> np.ndarray:
    if len(in_phase.shape) > 1 or len(quadrature_phase.shape) > 1:
        raise IndexError("get_phase error: the dimension of in_phase or quadrature_phase is not one, but %s, %s" % (
            in_phase.shape, quadrature_phase.shape))
    # c_phase = list(map(lambda iq: np.complex(iq[0], iq[1]), zip(in_phase, quadrature_phase)))
    c_phase = in_phase + 1j * quadrature_phase
    return np.abs(c_phase)