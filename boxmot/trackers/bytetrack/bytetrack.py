# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
from collections import deque

from ...motion.kalman_filters.xyah_kf import KalmanFilterXYAH
from basetracker import TrackState, GeneralTracker
from ...utils.matching import fuse_score, iou_distance, linear_assignment
from ..basetracker import BaseTracker


class ByteTrack(BaseTracker):
    """
    BYTETracker: A tracking algorithm based on ByteTrack, which utilizes motion-based tracking.

    Args:
        track_thresh (float, optional): Threshold for detection confidence. Detections above this threshold are considered for tracking in the first association round.
        match_thresh (float, optional): Threshold for the matching step in data association. Controls the maximum distance allowed between tracklets and detections for a match.
        track_buffer (int, optional): Number of frames to keep a track alive after it was last detected. A longer buffer allows for more robust tracking but may increase identity switches.
        frame_rate (int, optional): Frame rate of the video being processed. Used to scale the track buffer size.
        per_class (bool, optional): Whether to perform per-class tracking. If True, tracks are maintained separately for each object class.
    """
    def __init__(
        self,
        track_thresh: float = 0.6,
        match_thresh: float = 0.7,
        track_buffer: int = 100,
        frame_rate: int = 30,
        per_class: bool = False,
    ):
        super().__init__(per_class=per_class)
        self.active_tracks = []  # type: list[GeneralTracker]
        self.lost_stracks = []  # type: list[GeneralTracker]
        self.removed_stracks = []  # type: list[GeneralTracker]
        self._counter = 0

        self.frame_id = 0
        self.track_buffer = track_buffer

        self.per_class = per_class
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.det_thresh = track_thresh
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilterXYAH()

    @BaseTracker.on_first_frame_setup
    @BaseTracker.per_class_decorator
    def update(self, dets: np.ndarray, img: np.ndarray = None, embs: np.ndarray = None) -> np.ndarray:
        
        self.check_inputs(dets, img)

        dets = np.hstack([dets, np.arange(len(dets)).reshape(-1, 1)])
        self.frame_count += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        confs = dets[:, 4]

        remain_inds = confs > self.track_thresh

        inds_low = confs > 0.1
        inds_high = confs < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)

        dets_second = dets[inds_second]
        dets = dets[remain_inds]

        if len(dets) > 0:
            """Detections"""
            detections = [
                GeneralTracker(det, max_obs=self.max_obs, type="byte") for det in dets
            ]
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.active_tracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        """ Step 2: First association, with high conf detection boxes"""
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        self.multi_predict(strack_pool)
        dists = iou_distance(strack_pool, detections)
        # if not self.args.mot20:
        dists = fuse_score(dists, detections)
        matches, u_track, u_detection = linear_assignment(
            dists, thresh=self.match_thresh
        )

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                self.update_track(track, detections[idet], self.frame_count)
                activated_starcks.append(track)
            else:
                self.re_activate(track, det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        """ Step 3: Second association, with low conf detection boxes"""
        # association the untrack to the low conf detections
        if len(dets_second) > 0:
            """Detections"""
            detections_second = [GeneralTracker(det_second, max_obs=self.max_obs, type="byte") for det_second in dets_second]
        else:
            detections_second = []
        r_tracked_stracks = [
            strack_pool[i]
            for i in u_track
            if strack_pool[i].state == TrackState.Tracked
        ]
        dists = iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                self.update_track(track, det, self.frame_count)
                activated_starcks.append(track)
            else:
                self.re_activate(track, det, self.frame_count, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        """Deal with unconfirmed tracks, usually tracks with only one beginning frame"""
        detections = [detections[i] for i in u_detection]
        dists = iou_distance(unconfirmed, detections)
        # if not self.args.mot20:
        dists = fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            self.update_track(unconfirmed[itracked], detections[idet], self.frame_count)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.conf < self.det_thresh:
                continue
            self.activate(track, self.kalman_filter, self.frame_count)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_count - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.active_tracks = [
            t for t in self.active_tracks if t.state == TrackState.Tracked
        ]
        self.active_tracks = joint_stracks(self.active_tracks, activated_starcks)
        self.active_tracks = joint_stracks(self.active_tracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.active_tracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.active_tracks, self.lost_stracks = remove_duplicate_stracks(
            self.active_tracks, self.lost_stracks
        )
        # get confs of lost tracks
        output_stracks = [track for track in self.active_tracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output = []
            output.extend(t.xyxy)
            output.append(t.id)
            output.append(t.conf)
            output.append(t.cls)
            output.append(t.det_ind)
            outputs.append(output)
        outputs = np.asarray(outputs)
        return outputs


    def next_id(self):
        self._counter += 1
        return self._counter

    @staticmethod
    def update_track(track, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        track.frame_id = frame_id
        track.tracklet_len += 1
        track.history_observations.append(track.xyxy)

        track.mean, track.covariance = track.kalman_filter.update(
            track.mean, track.covariance, new_track.xyah
        )
        track.state = TrackState.Tracked
        track.is_activated = True

        track.conf = new_track.conf
        track.cls = new_track.cls
        track.det_ind = new_track.det_ind

    @staticmethod
    def multi_predict(stracks):
        shared_kalman = KalmanFilterXYAH()
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = shared_kalman.multi_predict(
                multi_mean, multi_covariance
            )
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov


    def activate(self, track, kalman_filter, frame_id):
        """Start a new tracklet"""
        track.kalman_filter = kalman_filter
        track.id = self.next_id()
        track.mean, track.covariance = track.kalman_filter.initiate(track.xyah)

        track.tracklet_len = 0
        track.state = TrackState.Tracked
        if frame_id == 1:
            track.is_activated = True
        # self.is_activated = True
        track.frame_id = frame_id
        track.start_frame = frame_id


    def re_activate(self, track, new_track, frame_id, new_id=False):
        track.mean, track.covariance = track.kalman_filter.update(
            track.mean, track.covariance, new_track.xyah
        )
        track.tracklet_len = 0
        track.state = TrackState.Tracked
        track.is_activated = True
        track.frame_id = frame_id
        if new_id:
            track.id = self.next_id()
        track.conf = new_track.conf
        track.cls = new_track.cls
        track.det_ind = new_track.det_ind

# id, class_id, conf


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.id] = t
    for t in tlistb:
        tid = t.id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if i not in dupa]
    resb = [t for i, t in enumerate(stracksb) if i not in dupb]
    return resa, resb
