#!/usr/bin/env python
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

"""Script to run registration algorithms on a sequence.

The dataset structure is as follows:

    REPOSITORY_ROOT/dataset/
    ├── sequence_01/
    │   ├── point_clouds/
    │   │   ├── 00001.bin  # N * (3 + 1) float32 bytes containing XYZ points
    │   │   ├── 00002.bin  # and Doppler velocities.
    │   │   └── ...
    │   ├── calibration.json
    │   └── ref_poses.txt  # N reference poses with timestamps in TUM format.
    ├── sequence_02/
    │   └── ...
    └── ...

Example usage:

    # Run Doppler ICP the sample sequence.
    $ python run.py -o /tmp/sample_output

    # Run point-to-plane ICP on a sequence in another directory (frame 100-150).
    $ python run.py --sequence /tmp/carla-town05 -o /tmp/sample_output \
        -s 100 -e 150 -m point_to_plane
"""

import argparse
import os
import sys
import traceback
from os.path import basename, dirname, isdir, join, realpath

import numpy as np
from tqdm import tqdm

import registration
import utils


def run(args):
    # Convert args to dict.
    params = vars(args)

    if isdir(args.sequence):
        # Directory mode.
        seq_dir = args.sequence
        args.sequence = basename(args.sequence)
    else:
        # Sequence name mode.
        script_dir = dirname(realpath(sys.argv[0]))
        seq_dir = join(join(dirname(script_dir), 'dataset'), args.sequence)

    assert isdir(seq_dir), 'Sequence: %s not found' % args.sequence

    # Extract the paths to all the point cloud scans.
    pcd_files = utils.load_point_cloud_filenames(join(seq_dir, 'point_clouds'))
    print('Loaded sequence: %s with %d scans' % (args.sequence, len(pcd_files)))

    # Load the reference poses (for evaluation) and calibration parameters.
    ref_poses, timestamps = utils.load_tum_poses(join(seq_dir, 'ref_poses.txt'))
    params['T_V_to_S'] = utils.load_calibration(seq_dir)

    # Picks the specified registration algorithm (Point-to-Plane vs DICP).
    icp_method = registration.doppler_icp if args.method == 'doppler' else \
                 registration.point_to_plane_icp

    results = {
        'poses': [np.eye(4)],
        'ref_poses': [np.eye(4)],
        'timestamps': [timestamps[args.start]],
        'convergence': [],
        'iterations': [],
        'translation_rpe': [],
        'rotation_rpe': [],
    }

    success = True
    if args.end < 0: args.end = len(pcd_files) - 1
    for i in tqdm(range(args.start, args.end), initial=1, desc='Processing'):
        # Load the point cloud into Open3D format (without any pre-processing).
        source = utils.load_point_cloud(pcd_files[i])
        target = utils.load_point_cloud(pcd_files[i + 1])

        # Compute the relative pose (for reference/ground-truth).
        ref_transform = utils.relative_pose(ref_poses[i + 1], ref_poses[i])
        params['period'] = timestamps[i + 1] - timestamps[i]

        # If the seed flag is set, the previous estimated pose in the sequence
        # is used to seed ICP under a constant-velocity motion assumption.
        init_transform = results['poses'][-1] if args.seed else np.eye(4)

        try:
            result = icp_method(source, target, params, init_transform)
        except Exception:
            print('FATAL: Failed to run ICP, terminating sequence')
            print(traceback.format_exc())
            success = False
            break

        # Computes the relative poses errors wrt to the reference pose.
        # Based on: D. Prokhorov et al., Measuring robustness of Visual SLAM.
        trans_rpe, rot_rpe = utils.relative_pose_error(result.transformation,
                                                       ref_transform,
                                                       degrees=True)

        # Store inverse poses which represent odometry.
        results['poses'].append(np.linalg.inv(result.transformation))
        results['ref_poses'].append(np.linalg.inv(ref_transform))
        results['timestamps'].append(timestamps[i + 1])
        results['convergence'].append(result.converged)
        results['iterations'].append(result.num_iterations)
        results['translation_rpe'].append(trans_rpe)
        results['rotation_rpe'].append(rot_rpe)

        if args.gui:
            utils.display_registration(source, target, result.transformation, i)

    sys.stdout.flush()

    # Compute the total path length using the relative poses.
    icp_path = utils.path_length(np.array(results['poses']))
    ref_path = utils.path_length(np.array(results['ref_poses']))

    print('-' * 40)
    print(('Sequence: %s' % args.sequence).center(40))
    print('-' * 40)
    print('Translation RPE (m): %.4f' % utils.rmse(results['translation_rpe']))
    print('Rotation RPE (deg): %.4f' % np.mean(results['rotation_rpe']))
    print('Estimated path length (m): %.2f' % icp_path)
    print('Reference path length (m): %.2f' % ref_path)
    print('Path length error (m): %.2f' % np.abs(icp_path - ref_path))
    print('Average iterations: %.1f' % np.mean(results['iterations']))
    print('Convergence rate: %.1f%%' % (np.mean(results['convergence']) * 100))
    print('-' * 40)

    # Save all the registration data to visualize or evaluate later using evo.
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez_compressed(join(args.output_dir, 'registration_data.npz'),
                        args=args, results=results)

    # Save the trajectory from ICP in TUM format.
    utils.export_tum_poses(join(args.output_dir, 'icp_poses.txt'),
                           results['poses'], results['timestamps'])

    # Emit exit code on failure.
    exit(not success)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=str, default='sample',
                        help='Sequence name from the dataset or the absolute'
                             ' path to the sequence directory')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='Directory to save the registration results at')
    parser.add_argument('--start', '-s', type=int, default=0,
                        help='Start frame index (inclusive)')
    parser.add_argument('--end', '-e', type=int, default=-1,
                        help='End frame index (inclusive)')
    parser.add_argument('--gui', action='store_true',
                        help='Shows the Open3D GUI after each registration')

    parser.add_argument('--method', '-m', default='doppler',
                        help='Registration method to use',
                        choices=['doppler', 'point_to_plane'])
    parser.add_argument('--seed', action='store_true',
                        help='Seed ICP using the previous pose estimate')
    parser.add_argument('--convergence_thresh', type=float, default=1e-5,
                        help='Convergence threshold for the registration'
                             ' algorithm. Higher the value, faster the'
                             ' convergence and lower the pose accuracy.')
    parser.add_argument('--max_iters', type=int, default=100,
                        help='Max iterations for the registration algorithm')
    parser.add_argument('--max_corr_distance', type=float, default=0.3,
                        help='Maximum correspondence points-pair distance (m)')
    parser.add_argument('--downsample_factor', type=int, default=2,
                        help='Factor to uniformly downsample the points by')
    parser.add_argument('--normals_radius', type=float, default=10.0,
                        help='Search radius (m) used in normal estimation')
    parser.add_argument('--normals_max_nn', type=int, default=30,
                        help='Max neighbors used in normal estimation search')

    parser.add_argument('--lambda_doppler', type=float, default=0.01,
                        help='Factor that weighs the Doppler residual term in'
                             ' the overall DICP objective. Setting a value of'
                             ' 0 is equivalent to point-to-plane ICP.')
    parser.add_argument('--reject_outliers', action='store_true',
                        help='Enable dynamic point outlier rejection')
    parser.add_argument('--outlier_thresh', type=float, default=2.0,
                        help='Error threshold (m/s) to reject dynamic outliers')
    parser.add_argument('--rejection_min_iters', type=int, default=2,
                        help='Number of iterations of ICP after which dynamic'
                             ' point outlier rejection is enabled')
    parser.add_argument('--geometric_min_iters', type=int, default=0,
                        help='Number of iterations of ICP after which robust'
                             ' loss for the geometric term is enabled')
    parser.add_argument('--doppler_min_iters', type=int, default=2,
                        help='Number of iterations of ICP after which robust'
                             ' loss for the Doppler term is enabled')
    parser.add_argument('--geometric_k', type=float, default=0.5,
                        help='Scale factor for the geometric robust loss')
    parser.add_argument('--doppler_k', type=float, default=0.2,
                        help='Scale factor for the Doppler robust loss')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('PID: [%d]' % os.getpid())
    run(args)
