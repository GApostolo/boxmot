# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import torch
import numpy as np
from pathlib import Path

from ...motion.kalman_filters.xywh_kf import KalmanFilterXYWH
from ...appearance.reid_auto_backend import ReidAutoBackend
from ...motion.cmc.sof import SOF
from basetracker import GeneralTracker, TrackState
from ...utils.matching import (embedding_distance, fuse_score,
                                   iou_distance, linear_assignment)
from ..basetracker import BaseTracker
from .botsort_utils import joint_stracks, sub_stracks, remove_duplicate_stracks
from ...motion.cmc import get_cmc_method
from ...appearance.fast_reid.fast_reid_interfece import FastReIDInterface


class BotSort(BaseTracker):
    """
    BoTSORT Tracker: A tracking algorithm that combines appearance and motion-based tracking.

    Args:
        reid_weights (str): Path to the model weights for ReID.
        device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
        half (bool): Use half-precision (fp16) for faster inference.
        per_class (bool, optional): Whether to perform per-class tracking.
        track_high_thresh (float, optional): Detection confidence threshold for first association.
        track_low_thresh (float, optional): Detection confidence threshold for ignoring detections.
        new_track_thresh (float, optional): Threshold for creating a new track.
        track_buffer (int, optional): Frames to keep a track alive after last detection.
        match_thresh (float, optional): Matching threshold for data association.
        proximity_thresh (float, optional): IoU threshold for first-round association.
        appearance_thresh (float, optional): Appearance embedding distance threshold for ReID.
        cmc_method (str, optional): Method for correcting camera motion, e.g., "sof" (simple optical flow).
        frame_rate (int, optional): Video frame rate, used to scale the track buffer.
        fuse_first_associate (bool, optional): Fuse appearance and motion in the first association step.
        with_reid (bool, optional): Use ReID features for association.
    """

    def __init__(
        self,
        reid_weights: Path,
        device: torch.device,
        half: bool,
        per_class: bool = False,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        proximity_thresh: float = 0.5,
        appearance_thresh: float = 0.25,
        cmc_method: str = "ecc",
        frame_rate=30,
        fuse_first_associate: bool = False,
        with_reid: bool = True,
        is_fast_reid: bool = False,
        fast_reid_config: str = "fast_reid/configs/MOT20/sbs_S50.yml"
    ):
        super().__init__(per_class=per_class)
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self._counter = 0

        self.per_class = per_class
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYWH()

        # ReID module
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        self.with_reid = with_reid
        if self.with_reid:
            if is_fast_reid:
                self.model = FastReIDInterface(fast_reid_config, str(reid_weights), device.type)
            else:
                self.model = ReidAutoBackend(weights=reid_weights, device=device, half=half).model

        self.cmc = get_cmc_method(cmc_method)()
        self.fuse_first_associate = fuse_first_associate

    @BaseTracker.on_first_frame_setup
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray, embs: np.ndarray = None) -> np.ndarray:
        self.check_inputs(dets, img)
        self.frame_count += 1

        activated_stracks, refind_stracks, lost_stracks, removed_stracks = [], [], [], []

        # Preprocess detections
        dets, dets_first, embs_first, dets_second = self._split_detections(dets, embs)

        # Extract appearance features
        if self.with_reid and embs is None:
            features_high = self.model.get_features(dets_first[:, 0:4], img)
        else:
            features_high = embs_first if embs_first is not None else []

        # Create detections
        detections = self._create_detections(dets_first, features_high)

        # Separate unconfirmed and active tracks
        unconfirmed, active_tracks = self._separate_tracks()
        
        strack_pool = joint_stracks(active_tracks, self.lost_stracks)

        # First association
        matches_first, u_track_first, u_detection_first = self._first_association(dets, dets_first, active_tracks, unconfirmed, img, detections, activated_stracks, refind_stracks, strack_pool)

        # Second association
        matches_second, u_track_second, u_detection_second = self._second_association(dets_second, activated_stracks, lost_stracks, refind_stracks, u_track_first, strack_pool)

        # Handle unconfirmed tracks
        matches_unc, u_track_unc, u_detection_unc = self._handle_unconfirmed_tracks(u_detection_first, detections, activated_stracks, removed_stracks, unconfirmed)

        # Initialize new tracks
        self._initialize_new_tracks(u_detection_unc, activated_stracks, [detections[i] for i in u_detection_first])

        # Update lost and removed tracks
        self._update_track_states(lost_stracks, removed_stracks)

        # Merge and prepare output
        return self._prepare_output(activated_stracks, refind_stracks, lost_stracks, removed_stracks)

    def _split_detections(self, dets, embs):
        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        confs = dets[:, 4]
        second_mask = np.logical_and(confs > self.track_low_thresh, confs < self.track_high_thresh)
        dets_second = dets[second_mask]
        first_mask = confs > self.track_high_thresh
        dets_first = dets[first_mask]
        embs_first = embs[first_mask] if embs is not None else None
        return dets, dets_first, embs_first, dets_second

    def _create_detections(self, dets_first, features_high):
        if len(dets_first) > 0:
            if self.with_reid:
                detections = [GeneralTracker(det, f, max_obs=self.max_obs, type="bot") for (det, f) in zip(dets_first, features_high)]
            else:
                detections = [GeneralTracker(det, max_obs=self.max_obs, type="bot") for det in dets_first]
        else:
            detections = []
        return detections

    def _separate_tracks(self):
        unconfirmed, active_tracks = [], []
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracks.append(track)
        return unconfirmed, active_tracks

    def _first_association(self, dets, dets_first, active_tracks, unconfirmed, img, detections, activated_stracks, refind_stracks, strack_pool):
        
        self.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.cmc.apply(img, dets)
        self.multi_gmc(strack_pool, warp)
        self.multi_gmc(unconfirmed, warp)

        # Associate with high confidence detection boxes
        ious_dists = iou_distance(strack_pool, detections)
        ious_dists_mask = ious_dists > self.proximity_thresh
        if self.fuse_first_associate:
            ious_dists = fuse_score(ious_dists, detections)

        if self.with_reid:
            emb_dists = embedding_distance(strack_pool, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
                
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                self.update_tracks(track, detections[idet], self.frame_count)
                activated_stracks.append(track)
            else:
                self.re_activate(track, det, self.frame_count, new_id=False)
                refind_stracks.append(track)
                
        return matches, u_track, u_detection

    def _second_association(self, dets_second, activated_stracks, lost_stracks, refind_stracks, u_track_first, strack_pool):
        if len(dets_second) > 0:
            detections_second = [GeneralTracker(det, max_obs=self.max_obs, type="bot") for det in dets_second]
        else:
            detections_second = []

        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track_first
            if strack_pool[i].state == TrackState.Tracked
        ]

        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection = linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                self.update_tracks(track, det, self.frame_count)
                activated_stracks.append(track)
            else:
                self.re_activate(track, det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)
                
        return matches, u_track, u_detection


    def _handle_unconfirmed_tracks(self, u_detection, detections, activated_stracks, removed_stracks, unconfirmed):
        """
        Handle unconfirmed tracks (tracks with only one detection frame).

        Args:
            u_detection: Unconfirmed detection indices.
            detections: Current list of detections.
            activated_stracks: List of newly activated tracks.
            removed_stracks: List of tracks to remove.
        """
        # Only use detections that are unconfirmed (filtered by u_detection)
        detections = [detections[i] for i in u_detection]
        
        # Calculate IoU distance between unconfirmed tracks and detections
        ious_dists = iou_distance(unconfirmed, detections)
        
        # Apply IoU mask to filter out distances that exceed proximity threshold
        ious_dists_mask = ious_dists > self.proximity_thresh
        ious_dists = fuse_score(ious_dists, detections)
        
        # Fuse scores for IoU-based and embedding-based matching (if applicable)
        if self.with_reid:
            emb_dists = embedding_distance(unconfirmed, detections) / 2.0
            emb_dists[emb_dists > self.appearance_thresh] = 1.0
            emb_dists[ious_dists_mask] = 1.0  # Apply the IoU mask to embedding distances
            dists = np.minimum(ious_dists, emb_dists)
        else:
            dists = ious_dists

        # Perform data association using linear assignment on the combined distances
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        
        # Update matched unconfirmed tracks
        for itracked, idet in matches:
            self.update_tracks(unconfirmed[itracked], detections[idet], self.frame_count)
            activated_stracks.append(unconfirmed[itracked])

        # Mark unmatched unconfirmed tracks as removed
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)
            
        return matches, u_unconfirmed, u_detection

    def _initialize_new_tracks(self, u_detections, activated_stracks, detections):
        for inew in u_detections:
            track = detections[inew]
            if track.conf < self.new_track_thresh:
                continue

            self.activate(track, self.kalman_filter, self.frame_count)
            activated_stracks.append(track)

    def _update_tracks(self, matches, strack_pool, detections, activated_stracks, refind_stracks, mark_removed=False):
        # Update or reactivate matched tracks
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                self.update_tracks(track, det, self.frame_count)
                activated_stracks.append(track)
            else:
                self.re_activate(track, det, self.frame_count, new_id=False)
                refind_stracks.append(track)
        
        # Mark only unmatched tracks as removed, if mark_removed flag is True
        if mark_removed:
            unmatched_tracks = [strack_pool[i] for i in range(len(strack_pool)) if i not in [m[0] for m in matches]]
            for track in unmatched_tracks:
                track.mark_removed()

    def _update_track_states(self, lost_stracks, removed_stracks):
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

    def _prepare_output(self, activated_stracks, refind_stracks, lost_stracks, removed_stracks):
        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_stracks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )

        outputs = [
            [*t.xyxy, t.id, t.conf, t.cls, t.det_ind]
            for t in self.active_tracks if t.is_activated
        ]

        return np.asarray(outputs)

    def next_id(self) -> int:
        """
        Generates the next unique track ID.

        Returns:
            int: A unique track ID.
        """
        self._counter += 1
        return self._counter

    @staticmethod
    def update_features(track, feat):
        """Normalize and update feature vectors."""
        feat /= np.linalg.norm(feat)
        track.curr_feat = feat
        if track.smooth_feat is None:
            track.smooth_feat = feat
        else:
            track.smooth_feat = track.alpha * track.smooth_feat + (1 - track.alpha) * feat
        track.smooth_feat /= np.linalg.norm(track.smooth_feat)
        track.features.append(feat)

    @staticmethod
    def update_cls(track, cls, conf):
        """Update class history based on detection confidence."""
        max_freq = 0
        found = False
        for c in track.cls_hist:
            if cls == c[0]:
                c[1] += conf
                found = True
            if c[1] > max_freq:
                max_freq = c[1]
                track.cls = c[0]
        if not found:
            track.cls_hist.append([cls, conf])
            track.cls = cls

    @staticmethod
    def multi_predict(stracks):
        """Perform batch prediction for multiple tracks."""
        shared_kalman = KalmanFilterXYWH()
        if not stracks:
            return
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        multi_covariance = np.asarray([st.covariance for st in stracks])
        for i, st in enumerate(stracks):
            if st.state != TrackState.Tracked:
                multi_mean[i][6:8] = 0  # Reset velocities
        multi_mean, multi_covariance = shared_kalman.multi_predict(multi_mean, multi_covariance)
        for st, mean, cov in zip(stracks, multi_mean, multi_covariance):
            st.mean, st.covariance = mean, cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """Apply geometric motion compensation to multiple tracks."""
        if not stracks:
            return
        R = H[:2, :2]
        R8x8 = np.kron(np.eye(4), R)
        t = H[:2, 2]

        for st in stracks:
            mean = R8x8.dot(st.mean)
            mean[:2] += t
            st.mean = mean
            st.covariance = R8x8.dot(st.covariance).dot(R8x8.T)

    def activate(self, track, kalman_filter, frame_id):
        """Activate a new track."""
        track.kalman_filter = kalman_filter
        track.id = self.next_id()
        track.mean, track.covariance = track.kalman_filter.initiate(track.xywh)
        track.tracklet_len = 0
        track.state = TrackState.Tracked
        if frame_id == 1:
            track.is_activated = True
        track.frame_id = frame_id
        track.start_frame = frame_id

    def re_activate(self, track, new_track, frame_id, new_id=False):
        """Re-activate a track with a new detection."""
        track.mean, track.covariance = track.kalman_filter.update(track.mean, track.covariance, new_track.xywh)
        if new_track.curr_feat is not None:
            self.update_features(track, new_track.curr_feat)
        track.tracklet_len = 0
        track.state = TrackState.Tracked
        track.is_activated = True
        track.frame_id = frame_id
        if new_id:
            track.id = self.next_id()
        track.conf = new_track.conf
        track.cls = new_track.cls
        track.det_ind = new_track.det_ind
        self.update_cls(track, new_track.cls, new_track.conf)

    def update_tracks(self, track, new_track, frame_id):
        """Update the current track with a matched detection."""
        track.frame_id = frame_id
        track.tracklet_len += 1
        track.history_observations.append(track.xyxy)

        track.mean, track.covariance = track.kalman_filter.update(track.mean, track.covariance, new_track.xywh)
        if new_track.curr_feat is not None:
            self.update_features(track, new_track.curr_feat)

        track.state = TrackState.Tracked
        track.is_activated = True
        track.conf = new_track.conf
        track.cls = new_track.cls
        track.det_ind = new_track.det_ind
        self.update_cls(track, new_track.cls, new_track.conf)
