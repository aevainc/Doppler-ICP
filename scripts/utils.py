# MIT License
#
# Copyright (c) 2022 Aeva, Inc
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Utilities library."""

import copy
import glob
import json
import os

import numpy as np
import open3d as o3d
import transformations as tf


def translation_quaternion_to_transform(translation,
                                        quaternion,
                                        inverse=False,
                                        quat_xyzw=False):
    """Converts translation and WXYZ quaternion to a transformation matrix.

    Args:
        translation: (3,) ndarray representing the translation vector.
        quaternion: (4,) ndarray representing the quaternion.
        inverse: If True, returns the inverse transformation.
        quat_xyzw: If True, this indicates that quaternion is in XYZW format.

    Returns:
        (4, 4) ndarray representing the transformation matrix.
    """
    if quat_xyzw: quaternion = np.roll(quaternion, 1)
    R = tf.quaternion_matrix(quaternion)  # [w, x, y, z]
    T = tf.translation_matrix(translation)  # [x, y, z]
    transform = tf.concatenate_matrices(T, R)
    return np.linalg.inv(transform) if inverse else transform


def transform_to_translation_quaternion(transform,
                                        inverse=False,
                                        quat_xyzw=False):
    """Converts a transformation matrix to translation and WXYZ quaternion.

    Args:
        transform: (4, 4) ndarray representing the transformation matrix.
        inverse: If True, inverts the input transformation first.
        quat_xyzw: If True, return a quaternion in XYZW format.

    Returns:
        (3,) ndarray representing the translation vector and
        (4,) ndarray representing the quaternion.
    """
    transform = np.linalg.inv(transform) if inverse else transform
    translation = transform[:3, -1]  # [x, y, z]
    quaternion = tf.quaternion_from_matrix(transform)  # [w, x, y, z]
    if quat_xyzw: quaternion = np.roll(quaternion, -1)
    return translation, quaternion


def rmse(prediction, reference=None):
    """Computes the root mean square (error).

    Args:
        prediction: array-like predicted values.
        reference: array-like reference values.

    Returns:
        The scalar RMSE between the prediction and reference arrays.
        If reference values are not specified, computes the RMS of prediction.
    """
    prediction = np.asarray(prediction)
    if reference is None: return np.sqrt(np.mean(prediction ** 2))
    return np.sqrt(np.mean((prediction - np.asarray(reference)) ** 2))


def relative_pose(T_W_to_A, T_W_to_B):
    """Computes the relative pose, given two absolute poses.

    Args:
        T_W_to_A: (4, 4) ndarray representing the first pose.
        T_W_to_B: (4, 4) ndarray representing the second pose.

    Returns:
        (4, 4) ndarray representing the relative pose T_A_to_B.
    """
    return tf.inverse_matrix(T_W_to_A) @ T_W_to_B


def relative_pose_error(transform, ref_transform, degrees=True):
    """Computes the relative pose errors (RPE) given two poses.

    Based on D. Prokhorov et al., Measuring robustness of Visual SLAM.

    Args:
        transform: (4, 4) ndarray representing the first relative pose.
        ref_transform: (4, 4) ndarray representing the second relative pose.
        degrees: If True, computes the rotation RPE in degrees.

    Returns:
        The translation RPE (in m) which is the norm of the translation
        and rotation RPE which is the angle (in deg) about the rotation axis
        corresponding to the the relative rotation matrix.
    """
    error_transform = relative_pose(ref_transform, transform)
    trans_rpe = np.linalg.norm(error_transform[:3, -1])
    rot_rpe = np.arccos(min((np.trace(error_transform[:3, :3]) - 1) / 2.0, 1))
    rot_rpe = np.degrees(rot_rpe) if degrees else rot_rpe
    return trans_rpe, rot_rpe


def path_length(poses):
    """Computes the path length, give a list of relative poses.

    Args:
        poses: (N, 4, 4) ndarray containing N relative poses.

    Returns:
        The path length accumulated from all the relative poses.
    """
    return np.linalg.norm(poses[:, :3, -1], axis=1).sum()


def load_point_cloud_filenames(directory):
    """Returns a list of point cloud (.bin) absolute paths in the directory.

    Args:
        directory: Path containing the point cloud files.

    Returns:
        Sorted list of point cloud (.bin) absolute paths.
    """
    assert os.path.isdir(directory)
    return sorted([os.path.join(directory, file)
                   for file in glob.glob(os.path.join(directory, '*.bin'))])


def load_point_cloud(filename, ndarray=False):
    """Loads a point cloud binary file and converts to o3d.geometry.PointCloud.

    Args:
        filename: Absolute path to the point cloud binary file.
        ndarray: If true, returns the ndarray, else an Open3D point cloud.

    Returns:
        o3d.geometry.PointCloud with geometry and Doppler velocities or
        (N, 4) ndarray containing the geometry and Doppler velocties.
    """
    data = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    if ndarray: return data.astype('float64')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3].astype('float64'))
    pcd.dopplers = o3d.utility.DoubleVector(data[:, -1].astype('float64'))
    return pcd


def load_calibration(directory):
    """Load the calibration parameters, given the sequence directory.

    Args:
        directory: Path containing the calibration.json file.

    Returns:
        (4, 4) ndarray extrinsic vehicle-to-sensor calibration (T_V_to_S).
    """
    with open(os.path.join(directory, 'calibration.json')) as file:
        data = json.load(file)
    t, q = data['T_V_to_S']['translation'], data['T_V_to_S']['quaternion']
    return translation_quaternion_to_transform(
        translation=np.array([t['x'], t['y'], t['z']]),
        quaternion=np.array([q['w'], q['x'], q['y'], q['z']]))


def load_tum_poses(filename):
    """Parses a TUM file to extract the trajectory poses and timestamps.

    Args:
        filename: Path to a TUM trajectory file.

    Returns:
        (N, 4, 4) ndarray of absolute poses from the trajectory and
        (N,) ndarray of corresponding timestamps (sec).
    """
    # Load the TUM file.
    data = np.loadtxt(filename, delimiter=' ')
    print('Loaded %d poses from %s (%.2f secs)' % (
        len(data), os.path.basename(filename), data[-1][0] - data[0][0]))

    # Parse timestamps and poses.
    timestamps = data[:, 0]
    poses = [translation_quaternion_to_transform(tq[:3], tq[3:], quat_xyzw=True)
             for tq in data[:, 1:]]
    return np.array(poses), timestamps


def export_tum_poses(filename, poses, timestamps, accumulate=True):
    """Exports relative/absolute poses with timestamps to TUM file.

    Args:
        filename: Path to save the TUM trajectory file.
        poses: (N, 4, 4) ndarray of absolute or relative poses to write.
        timestamps: (N,) ndarray of corresponding timestamps (sec).
        accumulate: If True, relative poses are accumulated to absolute poses.
    """
    assert len(poses) == len(timestamps)
    assert os.path.isdir(os.path.dirname(filename)), 'Invalid output filename'

    accumulated_pose = np.eye(4)
    with open(filename, 'w') as f:
        for pose, timestamp in zip(poses, timestamps):
            if accumulate:
                accumulated_pose = accumulated_pose @ pose
                t, q = transform_to_translation_quaternion(accumulated_pose)
            else:
                t, q = transform_to_translation_quaternion(pose)

            # TUM poses format: [timestamp, x, y, z, qx, qy, qz, qw].
            f.write('%f %f %f %f %f %f %f %f\n' % (
                timestamp, t[0], t[1], t[2], q[1], q[2], q[3], q[0]))

    print('Exported %d poses to file://%s' % (len(poses), filename))


def linear_interpolate(point1, point2, alphas):
    """Linearly interpolates point1 --> point2, given a vector of alphas.

    Args:
        point1: (M,) ndarray of start point.
        point2: (M,) ndarray of end point.
        alphas: (N,) ndarray of alpha values used for blending.

    Returns:
        (N, M) ndarray of blended values (1 - alphas)*point1 + alphas*point2.
    """
    points1 = np.repeat(np.expand_dims(point1, 0), len(alphas), axis=0)
    points2 = np.repeat(np.expand_dims(point2, 0), len(alphas), axis=0)
    return (1 - alphas[:, None]) * points1 + alphas[:, None] * points2


def generate_velocity_colors(velocity_channel,
                             min_velocity=-15,
                             max_velocity=15,
                             static_color=[1.0, 1.0, 1.0]):
    """Generates point cloud colors based on Doppler velocities.

    Args:
        velocity_channel: (N,) ndarray of Doppler velocities.
        min_velocity: The negative velocity value colored blue RGB(0, 0, 1).
        max_velocity: The positive velocity value colored red RGB(1, 0, 0).
        static_color: The color of the points at zero velocity.

    Returns:
        o3d.utility.Vector3dVector containing the colors for the point cloud.
    """
    # Color all points with static_color.
    colors = np.tile(static_color, (len(velocity_channel), 1))

    # Color all positive velocity points static_color -> red till max velocity.
    pos_idx = velocity_channel > 0
    pos_mag = np.clip(velocity_channel[pos_idx] / max_velocity, 0, 1)
    colors[pos_idx] = linear_interpolate(static_color, [1.0, 0.0, 0.0], pos_mag)

    # Color all negative velocity points static_color -> blue till min velocity.
    neg_idx = velocity_channel < 0
    neg_mag = np.clip(velocity_channel[neg_idx] / min_velocity, 0, 1)
    colors[neg_idx] = linear_interpolate(static_color, [0.0, 0.0, 1.0], neg_mag)

    return o3d.utility.Vector3dVector(colors.astype('float64'))


def display_registration(source, target, transform, frame_index=-1):
    """Visualizes the two point clouds in the same reference frame using Open3D.

    Args:
        source: o3d.geometry.PointCloud source point cloud.
        target: o3d.geometry.PointCloud target point cloud.
        transform: (4, 4) ndarray representing the registration pose (T_T_to_S).
        frame_index: Frame index/number for the window title.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([0, 0.651, 0.929])
    target_temp.paint_uniform_color([1, 0.706, 0])
    source_temp.transform(transform)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window('Registration Result (frame: %d)' % frame_index)
    visualizer.get_render_option().background_color = [0, 0, 0]
    visualizer.get_render_option().point_size = 1
    visualizer.add_geometry(source_temp)
    visualizer.add_geometry(target_temp)
    visualizer.run()
