from contextlib import contextmanager
import logging
import atexit
import threading
import importlib.util
import multiprocessing as mp
import time
import json
import numpy as np
from os.path import dirname, exists, isdir, join
from . import example_pb2, log_utils
from rich import progress
from rich.logging import RichHandler
from rich.highlighter import NullHighlighter
from rich.progress import Progress, TextColumn, BarColumn
from modules import errors
import os
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Tuple, Union, Iterator
)

# --- Global vars -------------------------------------------------------------
CPLEX_AVAILABLE = (importlib.util.find_spec('cplex') is not None)
def check_cplex_availability() -> bool:
    """Vérifie si CPLEX est disponible dans l'environnement.

    Returns:
        bool: True si CPLEX est installé, False sinon
    """
    return CPLEX_AVAILABLE

try:
    import pyomo.environ as pyo
    from pyomo.opt import SolverFactory
    opt = SolverFactory('bonmin', validate=False)
    if not opt.available():
        raise errors.SolverNotFoundError
except Exception:
    BONMIN_AVAILABLE = False
else:
    BONMIN_AVAILABLE = True
SUPPORTED_FORMATS = ['svs', 'tif', 'ndpi', 'vms', 'vmu', 'scn', 'mrxs',
                     'tiff', 'svslide', 'bif', 'jpg', 'jpeg', 'png',
                     'ome.tif', 'ome.tiff']

# --- Commonly used types -----------------------------------------------------

# Outcome labels
Labels = Union[Dict[str, str], Dict[str, int], Dict[str, List[float]]]

# --- Configure logging--------------------------------------------------------

log = logging.getLogger('slideflow')
log.setLevel(logging.DEBUG)

torch_available = importlib.util.find_spec('torch')

# Does a path exist?
# This is false for dangling symbolic links on systems that support them.
def exists(path):
    """Test whether a path exists.  Returns False for broken symbolic links"""
    try:
        os.stat(path)
    except (OSError, ValueError):
        return False
    return True


def setLoggingLevel(level):
    """Set the logging level.

    Uses standard python logging levels:

    - 50: CRITICAL
    - 40: ERROR
    - 30: WARNING
    - 20: INFO
    - 10: DEBUG
    - 0:  NOTSET

    Args:
        level (int): Logging level numeric value.

    """
    log.handlers[0].setLevel(level)


def getLoggingLevel():
    """Return the current logging level."""
    return log.handlers[0].level


@contextmanager
def logging_level(level: int):
    _initial = getLoggingLevel()
    setLoggingLevel(level)
    try:
        yield
    finally:
        setLoggingLevel(_initial)


def addLoggingFileHandler(path):
    fh = logging.FileHandler(path)
    fh.setFormatter(log_utils.FileFormatter())
    handler = log_utils.MultiProcessingHandler(
        "mp-file-handler-{0}".format(len(log.handlers)),
        sub_handler=fh
    )
    log.addHandler(handler)
    atexit.register(handler.close)


# Add tqdm-friendly stream handler
#ch = log_utils.TqdmLoggingHandler()
ch = RichHandler(
    markup=True,
    log_time_format="[%X]",
    show_path=False,
    highlighter=NullHighlighter(),
    rich_tracebacks=True
)
ch.setFormatter(log_utils.LogFormatter())
if 'SF_LOGGING_LEVEL' in os.environ:
    try:
        intLevel = int(os.environ['SF_LOGGING_LEVEL'])
        ch.setLevel(intLevel)
    except ValueError:
        pass
else:
    ch.setLevel(logging.INFO)
log.addHandler(ch)

# Add multiprocessing-friendly file handler
try:
    addLoggingFileHandler("slideflow.log")
except Exception as e:
    # If we can't write to the log file, just ignore it
    pass

# Workaround for duplicate logging with TF 2.9
log.propagate = False


class TileExtractionSpeedColumn(progress.ProgressColumn):
    """Renders human readable transfer speed."""

    def render(self, task: "progress.Task") -> progress.Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return progress.Text("?", style="progress.data.speed")
        data_speed = f'{int(speed)} img'
        return progress.Text(f"{data_speed}/s", style="progress.data.speed")


class LabeledMofNCompleteColumn(progress.MofNCompleteColumn):
    """Renders a completion column with labels."""

    def __init__(self, unit: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unit = unit

    def render(self, task: "progress.Task") -> progress.Text:
        """Show completion status with labels."""
        if task.total is None:
            return progress.Text("?", style="progress.spinner")
        return progress.Text(
            f"{task.completed}/{task.total} {self.unit}",
            style="progress.spinner"
        )


class ImgBatchSpeedColumn(progress.ProgressColumn):
    """Renders human readable transfer speed."""

    def __init__(self, batch_size=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size

    def render(self, task: "progress.Task") -> progress.Text:
        """Show data transfer speed."""
        speed = task.finished_speed or task.speed
        if speed is None:
            return progress.Text("?", style="progress.data.speed")
        data_speed = f'{int(speed * self.batch_size)} img'
        return progress.Text(f"{data_speed}/s", style="progress.data.speed")


class TileExtractionProgress(Progress):
    def get_renderables(self):
        for task in self.tasks:
            if task.fields.get("progress_type") == 'speed':
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    TileExtractionSpeedColumn()
                )
            if task.fields.get("progress_type") == 'slide_progress':
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    progress.TaskProgressColumn(),
                    progress.MofNCompleteColumn(),
                    "●",
                    progress.TimeRemainingColumn(),
                )
            yield self.make_tasks_table([task])


class FeatureExtractionProgress(Progress):
    def get_renderables(self):
        for task in self.tasks:
            if task.fields.get("progress_type") == 'speed':
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    TileExtractionSpeedColumn(),
                    LabeledMofNCompleteColumn('tiles'),
                    "●",
                    progress.TimeRemainingColumn(),
                )
            if task.fields.get("progress_type") == 'slide_progress':
                self.columns = (
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    progress.TaskProgressColumn(),
                    LabeledMofNCompleteColumn('slides')
                )
            yield self.make_tasks_table([task])


def set_ignore_sigint():
    """Ignore keyboard interrupts."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class MultiprocessProgressTracker:
    """Wrapper for a rich.progress tracker that can be shared across processes."""

    def __init__(self, tasks):
        ctx = mp.get_context('spawn')
        self.mp_values = {
            task.id: ctx.Value('i', task.completed)
            for task in tasks
        }

    def advance(self, id, amount):
        with self.mp_values[id].get_lock():
            self.mp_values[id].value += amount

    def __getitem__(self, id):
        return self.mp_values[id].value

class MultiprocessProgress:
    """Wrapper for a rich.progress bar that can be shared across processes."""

    def __init__(self, pb):
        self.pb = pb
        self.tracker = MultiprocessProgressTracker(self.pb.tasks)
        self.should_stop = False

    def _update_progress(self):
        while not self.should_stop:
            for task in self.pb.tasks:
                self.pb.update(task.id, completed=self.tracker[task.id])
            time.sleep(0.1)

    def __enter__(self):
        self._thread = threading.Thread(target=self._update_progress)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self.should_stop = True
        self._thread.join()

def path_to_name(path: str) -> str:
    '''Returns name of a file, without extension,
    from a given full path string.'''
    _file = os.path.basename(path)
    dot_split = _file.split('.')
    if len(dot_split) == 1:
        return _file
    elif len(dot_split) > 2 and '.'.join(dot_split[-2:]) in SUPPORTED_FORMATS:
        return '.'.join(dot_split[:-2])
    else:
        return '.'.join(dot_split[:-1])
    
def create_triangles(vertices, hole_vertices=None, hole_points=None):
    """
    Tessellate a complex polygon, possibly with holes.

    :param vertices: A list of vertices [(x1, y1), (x2, y2), ...] defining the polygon boundary.
    :param holes: An optional list of points [(hx1, hy1), (hx2, hy2), ...] inside each hole in the polygon.
    :return: A numpy array of vertices for the tessellated triangles.
    """
    import triangle as tr

    # Prepare the segment information for the exterior boundary
    segments = np.array([[i, (i + 1) % len(vertices)] for i in range(len(vertices))])

    # Prepare the polygon for Triangle
    polygon = {'vertices': np.array(vertices), 'segments': segments}

    # If there are holes and hole boundaries, add them to the polygon definition
    if hole_points is not None and hole_vertices is not None and len(hole_vertices):
        polygon['holes'] = np.array(hole_points).astype(np.float32)

        # Start adding hole segments after the exterior segments
        start_idx = len(vertices)
        for hole in hole_vertices:
            hole_segments = [[start_idx + i, start_idx + (i + 1) % len(hole)] for i in range(len(hole))]
            segments = np.vstack([segments, hole_segments])
            start_idx += len(hole)

        # Update the vertices and segments in the polygon
        all_vertices = np.vstack([vertices] + hole_vertices)
        polygon['vertices'] = all_vertices
        polygon['segments'] = segments

    # Tessellate the polygon
    tess = tr.triangulate(polygon, 'pF')

    # Extract tessellated triangle vertices
    if 'triangles' not in tess:
        return None

    tessellated_vertices = np.array([tess['vertices'][t] for t in tess['triangles']]).reshape(-1, 2)

    # Convert to float32
    tessellated_vertices = tessellated_vertices.astype('float32')

    return tessellated_vertices

def load_json(filename: str) -> Any:
    '''Reads JSON data from file.'''
    with open(filename, 'r') as data_file:
        return json.load(data_file)

def get_model_config(model_path: str) -> Dict:
    """Loads model configuration JSON file."""

    if model_path.endswith('params.json'):
        config = load_json(model_path)
    elif exists(join(model_path, 'params.json')):
        config = load_json(join(model_path, 'params.json'))
    elif exists(model_path) and exists(join(dirname(model_path), 'params.json')):
        if not (sf.util.torch_available
                and sf.util.path_to_ext(model_path) == 'zip'):
            log.warning(
                "Hyperparameters not in model directory; loading from parent"
                " directory. Please move params.json into model folder."
            )
        config = load_json(join(dirname(model_path), 'params.json'))
    else:
        raise errors.ModelParamsNotFoundError
    # Compatibility for pre-1.1
    if 'norm_mean' in config:
        config['norm_fit'] = {
            'target_means': config['norm_mean'],
            'target_stds': config['norm_std'],
        }
    if 'outcome_label_headers' in config:
        log.debug("Replacing outcome_label_headers in params.json -> outcomes")
        config['outcomes'] = config.pop('outcome_label_headers')
    # Compatibility for pre-3.0
    if 'model_type' in config and config['model_type'] == 'categorical':
        config['model_type'] = 'classification'
    if 'model_type' in config and config['model_type'] == 'linear':
        config['model_type'] = 'regression'
    return config

