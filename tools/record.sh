#!/usr/bin/env bash
ts=$(date +%Y%m%d-%H%M%S)
rosrun image_view video_recorder \
        image:=/kinect_head/rgb/half/image_rect_color \
        _codec:=X264 \
        _filename:=video-${ts}.mp4 \
        _fps:=20
