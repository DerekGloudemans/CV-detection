# CV-detection

This repository contains various utilities for object detection, object tracking, trajectory conversion from image space to world coordinates, and track plotting. A few frameworks for achieving object tracking are explored. A greedy algorithm as well as a Kalman Filter-based minimum cost bipartite matching (SORT) approach are both implemented. Objects are detected either by YOLO-V3 or by a ResNet-backbone bounding box regression network with likely object locations estimated from Kalman Filter propogation. Examples of tracks can be found [here](https://youtu.be/0WSP1GBL8m0 ) and [here](https://youtu.be/sMlnCxwmZ2w).





Like all good repositories for code development, I had no idea the behemoth of sundry files this repository would turn into when I first created it. As such, I'll soon be moving the relevant files to a new, streamlined repository and leaving this one here as legacy. 
