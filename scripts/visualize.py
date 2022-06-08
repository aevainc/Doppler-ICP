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

"""Utility to visualize the point cloud sequences using Open3D.

Example usage:

    # Visualize the sample sequence.
    $ python visualize.py --sequence sample

    # Visualize a sequence in another directory from frames 100-150.
    $ python visualize.py --sequence /tmp/carla-town05 -s 100 -e 150
"""

import argparse
import sys
import time
from os.path import basename, dirname, isdir, join, realpath

import numpy as np
import open3d as o3d
from tqdm import tqdm

import utils


def visualize(args):
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

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(args.sequence)
    pcd = o3d.geometry.PointCloud()

    if args.end < 0: args.end = len(pcd_files)
    for i in tqdm(range(args.start, args.end), desc='Frame', leave=False):
        # Load the point cloud and color using the Doppler velocity channel.
        points = utils.load_point_cloud(pcd_files[i], ndarray=True)
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        pcd.colors = utils.generate_velocity_colors(points[:, -1])

        if i - args.start == 0:
            # Setup the renderer in the first frame.
            visualizer.add_geometry(pcd)
            visualizer.get_render_option().background_color = [0, 0, 0]
            visualizer.get_render_option().point_size = 2

            # Sets the camera viewpoint (front).
            view_ctl = visualizer.get_view_control()
            camera = view_ctl.convert_to_pinhole_camera_parameters()
            camera.extrinsic = np.array([[0, -1, 0, 0], [-0.374, 0, -0.927, 10],
                                         [0.927, 0, -0.374, 22], [0, 0, 0, 1]])
            view_ctl.convert_from_pinhole_camera_parameters(camera)
        else:
            visualizer.update_geometry(pcd)
            visualizer.poll_events()
            visualizer.update_renderer()

        time.sleep(args.delay)

    visualizer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', type=str, default='sample',
                        help='Sequence name from the dataset or the absolute'
                             ' path to the sequence directory')
    parser.add_argument('--start', '-s', type=int, default=0,
                        help='Start frame index (inclusive)')
    parser.add_argument('--end', '-e', type=int, default=-1,
                        help='End frame index (exclusive)')
    parser.add_argument('--delay', '-d', type=float, default=0.02,
                        help='Frame delay for the visualizer')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    visualize(args)
