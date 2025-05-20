# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

from __future__ import absolute_import

import numpy as np

from ....motion.cmc import get_cmc_method
from ....motion.kalman_filters.xyah_kf import KalmanFilterXYAH
from . import iou_matching, linear_assignment
from basetracker import GeneralTracker, TrackState
from utility import Converter
from ....utils.matching import chi2inv95


class Tracker:
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """

    GATING_THRESHOLD = np.sqrt(chi2inv95[4])

    def __init__(
        self,
        metric,
        max_iou_dist=0.9,
        max_age=30,
        n_init=3,
        _lambda=0,
        ema_alpha=0.9,
        mc_lambda=0.995,
    ):
        self.metric = metric
        self.max_iou_dist = max_iou_dist
        self.max_age = max_age
        self.n_init = n_init
        self._lambda = _lambda
        self.ema_alpha = ema_alpha
        self.mc_lambda = mc_lambda

        self.tracks = []
        self._next_id = 1
        self.cmc = get_cmc_method('ecc')()

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.mean, track.covariance = track.kalman_filter.predict(track.mean, track.covariance)
            track.age += 1
            track.time_since_update += 1

    def increment_ages(self):
        for track in self.tracks:
            track.increment_age()
            track.mark_missed()

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.update_tracks(detections[detection_idx], track_idx)
        for track_idx in unmatched_tracks:
            self.mark_missed(track_idx)
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.id for _ in track.features]
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feat for i in detection_indices])
            targets = np.array([tracks[i].id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                cost_matrix,
                tracks,
                dets,
                track_indices,
                detection_indices,
                self.mc_lambda,
            )

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]

        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_dist,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        xyxy = Converter.tlwh2xyxy(detection.tlwh)
        det = np.concatenate([xyxy, [detection.conf, detection.cls, detection.det_ind]])
        track = GeneralTracker(det, feat=None, feat_history=50, max_obs=50, type=None)
        track.id = self._next_id
        track.xyah = detection.to_xyah()
        track.conf = detection.conf
        track.cls = detection.cls
        track.det_ind = detection.det_ind
        track.hits = 1
        track.age = 1
        track.time_since_update = 0
        track.alpha = self.ema_alpha

        # start with confirmed in Ci as test expect equal amount of outputs as inputs
        track.state = TrackState.Tentative
        track.features = []
        if detection.feat is not None:
            detection.feat /= np.linalg.norm(detection.feat)
            track.features.append(detection.feat)

        track._n_init = self.n_init
        track._max_age = self.max_age

        track.kalman_filter = KalmanFilterXYAH()
        track.mean, track.covariance = track.kalman_filter.initiate(detection.to_xyah())
        self.tracks.append(
            track
        )
        self._next_id += 1

    def update_tracks(self, detection, id: int):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        Parameters
        ----------
        detection : Detection
            The associated detection.
        """
        self.tracks[id].xyah = detection.to_xyah()
        self.tracks[id].conf = detection.conf
        self.tracks[id].cls = detection.cls
        self.tracks[id].det_ind = detection.det_ind
        self.tracks[id].mean, self.tracks[id].covariance = self.tracks[id].kalman_filter.update(
            self.tracks[id].mean, self.tracks[id].covariance, self.tracks[id].xyah, self.tracks[id].conf
        )

        feature = detection.feat / np.linalg.norm(detection.feat)

        smooth_feat = (
                self.tracks[id].alpha * self.tracks[id].features[-1] + (1 - self.tracks[id].alpha) * feature
        )
        smooth_feat /= np.linalg.norm(smooth_feat)
        self.tracks[id].features = [smooth_feat]

        self.tracks[id].hits += 1
        self.tracks[id].time_since_update = 0
        if self.tracks[id].state == TrackState.Tentative and self.tracks[id].hits >= self.tracks[id]._n_init:
            self.tracks[id].state = TrackState.Tracked

    def mark_missed(self, id: int):
        """Mark this track as missed (no association at the current time step)."""
        if self.tracks[id].state == TrackState.Tentative:
            self.tracks[id].state = TrackState.Removed
        elif self.tracks[id].time_since_update > self.tracks[id]._max_age:
            self.tracks[id].state = TrackState.Removed
