import numpy as np
import cv2

from . import convergence


def replay(history: np.ndarray,
           convergence_window_width: int,
           display_fps: int = 5,
           break_on_convergence: bool = True) -> convergence.ConvergenceProperties:

    """
    :param history: history of a single execution
    :param convergence_window_width: moving window used when trying to determine convergence
    :param display_fps: as in frames per second. Also affects replay speed.
    :param break_on_convergence: Whether to cut replay if convergence is determenined
    :return: properties of convergence
    """

    result: convergence.ConvergenceProperties = None
    num_cells = history.sum(axis=(1, 2))
    for i, state in enumerate(history, start=1):
        slice_start = max(i-convergence_window_width+2, 0)
        cnv_properties = convergence.check(num_cells[slice_start:i])
        cnv_properties.set_indicator_state(history[slice_start:i])
        cnv_properties.set_step(i)
        if cnv_properties.type != convergence.NONE:
            result = cnv_properties
        if display_fps:
            canvas = cv2.resize(state*255, (0, 0), fx=100, fy=100, interpolation=cv2.INTER_NEAREST)
            if result is not None:
                canvas = cv2.putText(state, result.type, (5, 5), cv2.FONT_HERSHEY_SIMPLEX, 1., (0, 0, 255))
            cv2.imshow("States", canvas)
            cv2.waitKey(1000 // display_fps)
        if break_on_convergence and result is not None:
            break
    return result


def classify_convergence(history: np.ndarray,
                         convergence_window_width: int) -> convergence.ConvergenceProperties:

    num_cells = history[-convergence_window_width:].sum(axis=(1, 2))
    cnv_properties = convergence.check(num_cells)
    cnv_properties.set_indicator_state(history[-convergence_window_width:])
    return cnv_properties
