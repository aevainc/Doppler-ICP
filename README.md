# DICP: Doppler Iterative Closest Point Algorithm

[Link to oral presentation at RSS 2022](https://youtu.be/HZqbmIaknNc?t=3114) | [Link to paper](http://www.roboticsproceedings.org/rss18/p015.pdf)

Doppler ICP is a novel algorithm for point cloud registration for range sensors capable of measuring per-return instantaneous radial velocity. Existing variants of ICP that solely rely on geometry or other features generally fail to estimate the motion of the sensor correctly in scenarios that have non-distinctive features and/or repetitive geometric structures such as hallways, tunnels, highways, and bridges.

We jointly optimize a new Doppler velocity objective function that exploits the compatibility of each point's Doppler measurement and the sensor's current motion estimate, and the geometric objective function which sufficiently constrains the point cloud alignment problem even in feature-denied environments.  

<img src="/images/Intro-RWT-02.jpg" alt="3D Reconstruction of Robin Williams Tunnel" width="600"/>
Comparison of tunnel reconstructions using point-to-plane ICP (left) and Doppler ICP (right) with measurements collected by an FMCW LiDAR. Point-to-plane ICP fails in this degenerate case due to the lack of distinctive features in the scene whereas the Doppler ICP algorithm is able to reconstruct the scene with very low error.

## Setup and Dependencies
Clone this repository along with the submodules.
```bash
git clone --recurse-submodules https://github.com/aevainc/Doppler-ICP.git
```
Install the Python dependencies.
```bash
pip install -r requirements.txt
```
The Doppler ICP algorithm is implemented in [our fork](https://github.com/aevainc/Open3D) of Open3D library. You need to build the Open3D wheel file for Python from source and install it in your Python environment. See the [Open3D docs](http://www.open3d.org/docs/release/compilation.html) for more details.
```bash
cd Open3D
mkdir build
cd build
cmake ..
make -j$(nproc) install-pip-package
```

## Dataset
Download [link](https://drive.google.com/file/d/11_-QnAEkIgUFYkeusQsIHa5_TiGNYqti/view?usp=sharing) for `CARLA-Town04-Straight-Walls` and `CARLA-Town05-Curved-Walls` sequences.

Extract and copy the downloaded sequences to the `dataset/` directory in the repository root. A `sample` sequence has been provided in the dataset. The file structure should look like the following:
```
REPOSITORY_ROOT/dataset/
├── sample/
│   ├── point_clouds/
│   │   ├── 00001.bin  # NUM_POINTS * (3 + 1) float32 bytes containing XYZ points and Doppler velocities.
│   │   ├── 00002.bin
│   │   └── ...
│   ├── calibration.json
│   └── ref_poses.txt  # NUM_SCANS reference poses with timestamps in TUM format.
├── carla-town04-straight-walls/
│   └── ...
└── ...
```

## Usage
Run Doppler ICP on the `sample` sequence and display the results. Additional registration results for all pairs of scans and the ICP poses for the sequence (in TUM format) are saved in `OUTPUT_DIR`.
```console
foo@bar:~/Doppler-ICP/scripts$ python run.py --sequence sample -o OUTPUT_DIR
```

Run point-to-plane ICP on a sequence downloaded in another directory (for frames 100-150).
```console
foo@bar:~/Doppler-ICP/scripts$ python run.py --sequence /tmp/carla-town05-curved-walls \
    -o /tmp/town05-output -s 100 -e 150 -m point_to_plane
```

Visualize the `sample` point cloud sequence (colored by the Doppler velocity channel) using Open3D.
```console
foo@bar:~/Doppler-ICP/scripts$ python visualize.py --sequence sample
```

Visualize the trajectories generated from the registration algorithms and the reference trajectory from the dataset using [evo](https://github.com/MichaelGrupp/evo).
```console
foo@bar:~$ evo_traj tum SEQUENCE_DIR/ref_poses.txt OUTPUT_DIR/icp_poses.txt --plot --plot_mode xyz
```

## Citations
If you use Doppler ICP in your work, please cite the corresponding [paper](http://www.roboticsproceedings.org/rss18/p015.pdf).

```bibtex
@INPROCEEDINGS{Hexsel-RSS-22, 
    AUTHOR    = {Bruno Hexsel AND Heethesh Vhavle AND Yi Chen}, 
    TITLE     = {{DICP: Doppler Iterative Closest Point Algorithm}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2022}, 
    ADDRESS   = {New York City, NY, USA}, 
    MONTH     = {June}, 
    DOI       = {10.15607/RSS.2022.XVIII.015} 
}
```

## Poster
<img src="/images/dicp-poster.jpg" alt="Doppler ICP Poster"/>
