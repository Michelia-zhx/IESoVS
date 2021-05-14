#!/bin/bash
video_path="/home/zzc/Downloads/"
videos=$(ls $video_path)
output_path="output/"
result_file="flow_0_3.txt"

for video_name in $videos
do 
    echo -e "\n" $video_name >> $result_file

    ffmpeg -i $video_path$video_name -hide_banner >> $result_file 2>&1

    out_name=$(echo $video_name | cut -d '.' -f 1)

    start=$(date +%s.%N)

    python ./demo.py --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml --video-input $video_path$video_name --output $output_path$out_name.mp4 --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl

    end=$(date +%s.%N)

    start_s=$(echo $start | cut -d '.' -f 1)
    start_ns=$(echo $start | cut -d '.' -f 2)
    end_s=$(echo $end | cut -d '.' -f 1)
    end_ns=$(echo $end | cut -d '.' -f 2)

    echo "运行时间： "$((( 10#$end_s - 10#$start_s ) * 1000 + ( 10#$end_ns / 1000000 - 10#$start_ns / 1000000 ) )) ms >> $result_file
done