import numpy as np
from spatialmath import SE3
import cv2
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from camera import PinholeCamera

@dataclass
class FrontendLandmark():
    global_id: int = -1
    measurements: dict[list[int]] = field(default_factory=dict) #[u, v, depth]
    observed_frames: list[int] = field(default_factory=list)

    def is_outlier_gt(self, frontend, threshold=0.1):
        point_xyz_gt = [self.back_project(frontend, i) for i in self.observed_frames]
        point_xyz_gt = np.array(point_xyz_gt)
        error = np.abs((np.mean(point_xyz_gt, axis=0)-point_xyz_gt)).sum()/len(self.observed_frames)
        return error>threshold
    
    def back_project(self, frontend, frame_id):
        pose = frontend.frames[frame_id].odom_pose
        camera = PinholeCamera(frontend.params)
        measurement = self.measurements[frame_id]
        points_xyz = camera.back_project(*measurement, pose).flatten()
        return points_xyz

@dataclass
class FrontendKeyframe():
    frame_id: int = -1
    odom_pose: SE3 = SE3()
    color: np.ndarray = np.array([])
    gray: np.ndarray = np.array([])
    depth: np.ndarray = np.array([])
    points: list[np.ndarray] = field(default_factory=list)
    keypoints: list[cv2.KeyPoint] = field(default_factory=list)
    descriptors: list[np.ndarray] = field(default_factory=list)
    matches: list[cv2.DMatch] = field(default_factory=list)
    global_id: list[int] = field(default_factory=list)
    landmarks: list[FrontendLandmark] = field(default_factory=list)

class Frontend():
    def __init__(self, params):
        # Parameters
        self.params = params

        # All states
        self.frames = []
        self.landmarks = {}
        self.last_frame = FrontendKeyframe()
        self.curr_frame = FrontendKeyframe()
        self.frame_id = -1

    def keyframe_selection(self, odom_pose) -> bool:
        '''
        @param odom_pose: current odometry pose
        @return: True if a keyframe is selected
        '''
        is_keyframe = False
        if self.frame_id == -1:
            # First frame
            is_keyframe = True
        else:
            # Determine if the motion is large enough
            rel_pose = self.curr_frame.odom_pose.inv() @ odom_pose
            w_trans = self.params['frontend']['keyframe']['trans_weight']
            w_rot = self.params['frontend']['keyframe']['rot_weight']
            weight = np.array([w_trans]*3+[w_rot]*3)
            diff = np.linalg.norm(rel_pose.log(True) * weight)
            threshold = self.params['frontend']['keyframe']['threshold']
            is_keyframe = diff > threshold
        return is_keyframe

    def add_keyframe(self, odom_pose, color, depth) -> None:
        '''
        @param odom_pose: current pose
        @param color: current color image
        @param depth: current depth image
        '''
        self.frame_id += 1
        self.last_frame = self.curr_frame
        self.curr_frame = FrontendKeyframe()
        self.curr_frame.frame_id = self.frame_id
        self.curr_frame.odom_pose = odom_pose
        #print(self.params['frontend']['feature']['sample']['method'])
        self.curr_frame.color = color
        self.curr_frame.gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        self.curr_frame.depth = depth
        self.frames.append(self.curr_frame)

    def extract_features(self, n=None, detector_name=None, sample_method=None, append_mode=False) -> None:
        # select detector
        if n is None:
            n = self.params['frontend']['feature']['number']
        if detector_name is None:
            detector_name = self.params['frontend']['feature']['detector']
        detector_name = detector_name.lower()
        if detector_name == 'sift':
            detector = cv2.SIFT_create(n)
        elif detector_name == 'orb':
            detector = cv2.ORB_create(n)
        elif detector_name == 'akaze':
            detector = cv2.AKAZE_create()
        
        print(self.params['frontend']['feature']['sample']['method'])
        # todo: add sample methods
        
        if sample_method is None:
            sample_method = self.params['frontend']['feature']['sample']['method']
        if sample_method == 'none':
            keypoints, descriptors = detector.detectAndCompute(
                self.curr_frame.color, None)
        else:
            raise NotImplementedError
        if not append_mode:
            self.curr_frame.keypoints = keypoints
            self.curr_frame.descriptors = descriptors
            self.curr_frame.points = [x.pt for x in self.curr_frame.keypoints]
        else:
            self.curr_frame.keypoints += keypoints
            self.curr_frame.points += [x.pt for x in keypoints]

    def match_features(self, matcher_name=None, sample_method=None) -> None:
        if self.frame_id == 0:
            return
        # select matcher
        if matcher_name is None:
            matcher_name = self.params['frontend']['match']['matcher']
        if matcher_name == 'bruteforce':
            cross_check = self.params['frontend']['match']['cross_check']
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=cross_check)
            self.curr_frame.matches = matcher.match(self.last_frame.descriptors, self.curr_frame.descriptors)
        elif matcher_name == 'opticalflow':
            curr_points_all, status, err = cv2.calcOpticalFlowPyrLK(
                self.last_frame.gray, self.curr_frame.gray,
                np.array(self.last_frame.points, dtype=np.float32), None, winSize=(21, 21), maxLevel=3)
            self.curr_frame.points = []
            self.curr_frame.matches = []
            for i, (point, is_inlier) in enumerate(zip(curr_points_all, status)):
                if is_inlier and self._in_frame(point):
                    query_idx = i
                    train_idx = len(self.curr_frame.points)
                    self.curr_frame.points.append(point)
                    self.curr_frame.matches.append(cv2.DMatch(query_idx, train_idx, 0))
        else:
            raise NotImplementedError
        self._reducePoints()
    
    def _in_frame(self, point):
        h, w = self.curr_frame.color.shape[:2]
        return 0 <= point[0] < w and 0 <= point[1] < h

    def eliminate_outliers(self, ransacReprojThreshold=3):
        if len(self.curr_frame.matches)<8:
            return
        last_points = np.array([self.last_frame.points[match.queryIdx] for match in self.curr_frame.matches])
        curr_points = np.array([self.curr_frame.points[match.trainIdx] for match in self.curr_frame.matches])
        retval, mask = cv2.findFundamentalMat(last_points, curr_points, cv2.FM_RANSAC, ransacReprojThreshold, 0.99, None)
        
        matches = []
        for i, is_inlier in enumerate(mask):
            if is_inlier:
                matches.append(self.curr_frame.matches[i])
        self.curr_frame.matches = matches
        self._reducePoints()

    def _reducePoints(self):
        '''
        Reduce points to the matched ones and update idx in matches
        '''
        points = []
        for i, match in enumerate(self.curr_frame.matches):
            points.append(self.curr_frame.points[match.trainIdx])
            match.trainIdx = i
        self.curr_frame.points = points

    def plot_features(self, ax=None):
        canvas = np.array(self.curr_frame.color)
        for point in self.curr_frame.points:
            cv2.circle(canvas, (int(point[0]), int(
                point[1])), 4, (255, 0, 0), 1)
        if ax is None:
            ax = plt.gca()
        ax.imshow(canvas[:, :, ::-1])
        return canvas[:, :, ::-1]

    def plot_matches(self, with_global_id=False, ax=None):
        if self.frame_id == 0:
            return self.plot_features(ax=ax)
        else:
            canvas = np.concatenate([self.last_frame.color, self.curr_frame.color], axis=1)
            for match in self.curr_frame.matches:
                pt1 = [int(x) for x in self.last_frame.points[match.queryIdx]]
                pt2 = [int(x) for x in self.curr_frame.points[match.trainIdx]]
                pt2[0] += self.last_frame.color.shape[1]
                cv2.circle(canvas, pt1, 4, (255, 0, 0), 1)
                cv2.circle(canvas, pt2, 4, (255, 0, 0), 1)
                cv2.line(canvas, (int(pt1[0]), int(pt1[1])), (int(
                    pt2[0]), int(pt2[1])), (255, 0, 0), 1)
                if with_global_id:
                    pt1[0] += 10
                    pt1[1] += 10
                    pt1[0] += 10
                    pt2[1] += 10
                    cv2.putText(canvas, str(self.last_frame.global_id[match.queryIdx]), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255))
                    cv2.putText(canvas, str(self.curr_frame.global_id[match.trainIdx]), pt2, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255))
            if ax is None:
                ax = plt.gca()
            ax.imshow(canvas[:, :, ::-1])
            return canvas[:, :, ::-1]
    
    def assign_global_id(self):
        if self.frame_id == 0:
            self.curr_frame.global_id = [i for i in range(len(self.curr_frame.points))]
        else:
            self.curr_frame.global_id = [-1 for i in range(len(self.curr_frame.points))]
            for match in self.curr_frame.matches:
                self.curr_frame.global_id[match.trainIdx] = self.last_frame.global_id[match.queryIdx]
            index = np.max(self.curr_frame.global_id) + 1
            for i, global_id in enumerate(self.curr_frame.global_id):
                if global_id == -1:
                    self.curr_frame.global_id[i] = index
                    index += 1
        for global_id, point in zip(self.curr_frame.global_id, self.curr_frame.points):
            if global_id in self.landmarks:
                landmark = self.landmarks[global_id]
            else:
                landmark = FrontendLandmark()
                landmark.global_id = global_id
                self.landmarks[global_id] = landmark
            depth = self.curr_frame.depth[int(point[1]), int(point[0])]
            measurement = [*point, depth]
            landmark.observed_frames.append(self.frame_id)
            landmark.measurements[self.frame_id] = measurement
            self.curr_frame.landmarks.append(landmark)


if __name__ == '__main__':
    frontend = Frontend('./params/tartanair.yaml')
    new_pose = SE3()
    frontend.keyframe_selection(new_pose)
    new_pose = SE3.Rx(0.5)
    frontend.keyframe_selection(new_pose)
