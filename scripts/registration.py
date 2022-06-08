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

"""Library containing the Open3D API calls to different registration methods."""

import copy

import numpy as np
import open3d as o3d
import transformations as tf

assert hasattr(o3d.pipelines.registration, 'registration_doppler_icp'), \
    'Open3D must be built from source using https://github.com/aevainc/Open3D'


def point_to_plane_icp(source, target, params, init_transform=np.eye(4)):
    """Registers the two point clouds using the point-to-plane ICP method.

    Args:
        source: o3d.geometry.PointCloud source point cloud.
        target: o3d.geometry.PointCloud target point cloud.
        params: Dictionary containing all the algorithm parameters.
        init_transform: (4, 4) ndarray representing the seed pose (T_T_to_S).

    Returns:
        o3d.pipelines.registration.RegistrationResult object.
    """
    # Downsample and transform the point cloud from sensor to vehicle frame.
    source_in_V = source.uniform_down_sample(
        params['downsample_factor']).transform(params['T_V_to_S'])
    target_in_V = target.uniform_down_sample(
        params['downsample_factor']).transform(params['T_V_to_S'])

    # Compute normal vectors for the target point cloud.
    target_in_V.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=params['normals_radius'],
                                             max_nn=params['normals_max_nn']))

    return o3d.pipelines.registration.registration_icp(
        source_in_V, target_in_V, params['max_corr_distance'], init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(
            o3d.pipelines.registration.TukeyLoss(k=params['geometric_k'])),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=params['convergence_thresh'],
            relative_rmse=params['convergence_thresh'],
            max_iteration=params['max_iters']))


def doppler_icp(source, target, params, init_transform=np.eye(4)):
    """Registers the two point clouds using the Doppler ICP method.

    Args:
        source: o3d.geometry.PointCloud source point cloud (with Doppler).
        target: o3d.geometry.PointCloud target point cloud.
        params: Dictionary containing all the algorithm parameters.
        init_transform: (4, 4) ndarray representing the seed pose (T_T_to_S).

    Returns:
        o3d.pipelines.registration.RegistrationResult object.
    """
    # Uniformly downsample the point cloud.
    source_in_S_down = source.uniform_down_sample(params['downsample_factor'])
    target_in_S_down = target.uniform_down_sample(params['downsample_factor'])

    # Transform the point cloud from sensor to vehicle frame.
    source_in_V = copy.deepcopy(source_in_S_down).transform(params['T_V_to_S'])
    target_in_V = copy.deepcopy(target_in_S_down).transform(params['T_V_to_S'])

    # Compute direction vectors for the source point cloud (in sensor frame).
    # NOTE: These direction vectors must correspond to the original range
    # and Doppler measurement's reference frame. This includes ignoring any
    # sort of motion or rolling shutter correction applied on the points.
    source_directions_in_S = o3d.utility.Vector3dVector(
        tf.unit_vector(np.array(source_in_S_down.points), axis=1))

    # Compute normal vectors for the target point cloud.
    target_in_V.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=params['normals_radius'],
                                             max_nn=params['normals_max_nn']))

    return o3d.pipelines.registration.registration_doppler_icp(
        source_in_V, target_in_V, source_directions_in_S,
        params['max_corr_distance'], init_transform,
        o3d.pipelines.registration.TransformationEstimationForDopplerICP(
            lambda_doppler=params['lambda_doppler'],
            reject_dynamic_outliers=params['reject_outliers'],
            doppler_outlier_threshold=params['outlier_thresh'],
            outlier_rejection_min_iteration=params['rejection_min_iters'],
            geometric_robust_loss_min_iteration=params['geometric_min_iters'],
            doppler_robust_loss_min_iteration=params['doppler_min_iters'],
            geometric_kernel=o3d.pipelines.registration.TukeyLoss(
                k=params['geometric_k']),
            doppler_kernel=o3d.pipelines.registration.TukeyLoss(
                k=params['doppler_k'])),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=params['convergence_thresh'],
            relative_rmse=params['convergence_thresh'],
            max_iteration=params['max_iters']),
        period=params['period'],
        T_V_to_S=params['T_V_to_S'])
