# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from importlib import import_module
import sys
from pathlib import Path
import yaml
from .utils import BOXMOT, TRACKER_CONFIGS

def get_tracker_config(tracker_type):
    """Returns the path to the tracker configuration file."""
    return TRACKER_CONFIGS / f'{tracker_type}.yaml'


def create_tracker(tracker_type, tracker_config=None, reid_weights=None, device=None, half=None, per_class=None,
                   evolve_param_dict=None, is_fast_reid: bool = True,
                   fast_reid_config: str = "boxmot/appearance/fast_reid/configs/MOT20/sbs_S50.yml",):
    """
    Creates and returns an instance of the specified tracker type.
    
    Parameters:
    - tracker_type: The type of the tracker (e.g., 'strongsort', 'ocsort').
    - tracker_config: Path to the tracker configuration file.
    - reid_weights: Weights for ReID (re-identification).
    - device: Device to run the tracker on (e.g., 'cpu', 'cuda').
    - half: Boolean indicating whether to use half-precision.
    - per_class: Boolean for class-specific tracking (optional).
    - evolve_param_dict: A dictionary of parameters for evolving the tracker.
    
    Returns:
    - An instance of the selected tracker.
    """
    
    # Load configuration from file or use provided dictionary
    if evolve_param_dict is None:
        with open(tracker_config, "r") as f:
            yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            tracker_args = {param: details['default'] for param, details in yaml_config.items()}
    else:
        tracker_args = evolve_param_dict

    # Arguments specific to ReID models
    reid_args = {
        'reid_weights': reid_weights,
        'device': device,
        'half': half,
        'is_fast_reid': is_fast_reid,
        'fast_reid_config': fast_reid_config

    }

    # Map tracker types to their corresponding classes
    tracker_mapping = {
        'strongsort': '.trackers.strongsort.strongsort.StrongSort',
        'ocsort': '.trackers.ocsort.ocsort.OcSort',
        'bytetrack': '.trackers.bytetrack.bytetrack.ByteTrack',
        'botsort': '.trackers.botsort.botsort.BotSort',
        'deepocsort': '.trackers.deepocsort.deepocsort.DeepOcSort',
        'hybridsort': '.trackers.hybridsort.hybridsort.HybridSort',
        'imprassoc': '.trackers.imprassoc.imprassoctrack.ImprAssocTrack'
    }

    # Check if the tracker type exists in the mapping
    if tracker_type not in tracker_mapping:
        print('Error: No such tracker found.')
        exit()

    # Dynamically import and instantiate the correct tracker class
    module_path, class_name = tracker_mapping[tracker_type].rsplit('.', 1)

    try:
        tracker_module = import_module(module_path, package='boxmot')
        tracker_class = getattr(tracker_module, class_name)
    except ImportError:
        project_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(project_root))

        tracker_mapping = {
            'strongsort': 'trackers.boxmot.trackers.strongsort.strongsort.StrongSort',
            'ocsort': 'trackers.boxmot.trackers.ocsort.ocsort.OcSort',
            'bytetrack': 'trackers.boxmot.trackers.bytetrack.bytetrack.ByteTrack',
            'botsort': 'trackers.boxmot.trackers.botsort.botsort.BotSort',
            'deepocsort': 'trackers.boxmot.trackers.deepocsort.deepocsort.DeepOcSort',
            'hybridsort': 'trackers.boxmot.trackers.hybridsort.hybridsort.HybridSort',
            'imprassoc': 'trackers.boxmot.trackers.imprassoc.imprassoctrack.ImprAssocTrack'
        }

        module_path, class_name = tracker_mapping[tracker_type].rsplit('.', 1)
        tracker_module = import_module(module_path)
        tracker_class = getattr(tracker_module, class_name)

    # For specific trackers, update tracker arguments with ReID parameters
    if tracker_type in ['strongsort', 'botsort', 'deepocsort', 'hybridsort', 'imprassoc']:
        tracker_args['per_class'] = per_class
        tracker_args.update(reid_args)
        if tracker_type == 'strongsort':
            tracker_args.pop('per_class')  # per class not supported by
    else:
        tracker_args['per_class'] = per_class

    # Return the instantiated tracker class with arguments
    return tracker_class(**tracker_args)