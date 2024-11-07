# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

__version__ = '11.0.5'

from .postprocessing.gsi import gsi
from .tracker_zoo import create_tracker, get_tracker_config
from .trackers.botsort.botsort import BotSort
from .trackers.bytetrack.bytetrack import ByteTrack
from .trackers.deepocsort.deepocsort import DeepOcSort
from .trackers.hybridsort.hybridsort import HybridSort
from .trackers.ocsort.ocsort import OcSort
from .trackers.strongsort.strongsort import StrongSort
from .trackers.imprassoc.imprassoctrack import ImprAssocTrack


TRACKERS = ['bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'hybridsort', 'imprassoc']

__all__ = ("__version__",
           "StrongSort", "OcSort", "ByteTrack", "BotSort", "DeepOcSort", "HybridSort", "ImprAssocTrack"
           "create_tracker", "get_tracker_config", "gsi")
