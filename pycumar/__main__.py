import os
import ctypes

import numpy as np


class _g:
    c_double_p = ctypes.POINTER(ctypes.c_double)
    c_bool_p = ctypes.POINTER(ctypes.c_bool)
    c_int_p = ctypes.POINTER(ctypes.c_int32)
    dll = ctypes.cdll.LoadLibrary(os.path.join(os.path.dirname(__path__), "cumar_f64.dll"))
    m, n = 0, 0


def init(target, base):
    """
    Loads models to GPU memory.

    ``target`` - May be meassured 3D model. Must be or be convertible to ``np.ndarray((m,3,3), np.float64)``.

    ``base`` - May be 3DCAD model. Must be or be convertible to ``np.ndarray((n,3,3), np.float64)``.
    """
    if target is not None:
        target: np.ndarray = np.array(target, np.float64, copy=False)
        assert target.size % 9 == 0
        m = target.size // 9
        _g.m = m
        target = target.ctypes.data_as(_g.c_double_p)
    else: target, m = 0, 0
    if base is not None:
        base: np.ndarray = np.array(base, np.float64, copy=False)
        assert base.size % 9 == 0
        n = base.size // 9
        _g.n = n
        base = base.ctypes.data_as(_g.c_double_p)
    else: base, n = 0, 0
    _g.dll.cumar_init(target, m, base, n)


def calc(*, rotation=None, translation=None, dist=None, dist2=None, closest=None, right_above=None):
    """
    Calculates the margin between models.

    ``rotation`` - Rotates ``target`` before translating. Must be or be convertible to ``np.ndarray(3, np.float64)``.

    ``translation`` - Translates ``target`` before calculating. Must be or be convertible to ``np.ndarray(3, np.float64)``.

    ``dist`` - Stores the distance from each facet of ``target`` to the closest facet of ``base``. Must have the valid ``ctypes`` attribute.

    ``dist2`` - Stores the square distance. Must have the valid ``ctypes`` attribute.

    ``closest`` - Stores the index of the closest facet of ``base``. Must have the valid ``ctypes`` attribute.

    ``right_above`` - Stores the boolean of whether each facet of ``target`` is right above the nearest facet of ``base``. Must have the valid ``ctypes`` attribute.
    """
    if rotation is not None:
        rotation: np.ndarray = np.array(rotation, np.float64, copy=False)
        assert rotation.size == 3
        rotation = rotation.ctypes.data_as(_g.c_double_p)
    else: rotation = 0
    if translation is not None:
        translation: np.ndarray = np.array(translation, np.float64, copy=False)
        assert translation.size == 3
        translation = translation.ctypes.data_as(_g.c_double_p)
    else: translation = 0
    if dist is not None:
        assert dist.size == _g.m
        dist = dist.ctypes.data_as(_g.c_double_p)
    else: dist = 0
    if dist2 is not None:
        assert dist2.size == _g.m
        dist2 = dist2.ctypes.data_as(_g.c_double_p)
    else: dist2 = 0
    if closest is not None:
        assert closest.size == _g.m
        closest = closest.ctypes.data_as(_g.c_int_p)
    else: closest = 0
    if right_above is not None:
        assert right_above.size == _g.m
        right_above = right_above.ctypes.data_as(_g.c_bool_p)
    else: right_above = 0
    _g.dll.cumar_calc(rotation, translation, dist, dist2, closest, right_above)
