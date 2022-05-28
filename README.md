# DICP: Doppler Iterative Closest Point Algorithm
This is the code release for Doppler ICP (will be available soon). Doppler ICP is a novel algorithm for point cloud registration for range sensors capable of measuring per-return instantaneous radial velocity. Existing variants of ICP that solely rely on geometry or other features generally fail to estimate the motion of the sensor correctly in scenarios that have non-distinctive features and/or repetitive geometric structures such as hallways, tunnels, highways, and bridges. We jointly optimize a new Doppler velocity objective function that exploits the compatibility of each point's Doppler measurement and the sensor's current motion estimate, and the geometric objective function which sufficiently constrains the point cloud alignment problem even in feature-denied environments.  

<img src="/images/Intro-RWT-02.jpg" alt="3D Reconstruction of Robin Williams Tunnel" width="600"/>
Comparison of tunnel reconstructions using point-to-plane ICP (left) and Doppler ICP (right) with measurements collected by an FMCW LiDAR. Point-to-plane ICP fails in this degenerate case due to the lack of distinctive features in the scene whereas the Doppler ICP algorithm is able to reconstruct the scene with very low error.

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
