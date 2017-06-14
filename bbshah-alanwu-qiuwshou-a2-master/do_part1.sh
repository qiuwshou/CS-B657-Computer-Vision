#!/bin/bash

query=$1
#ratio=$2
image_dir="a2-images/a2-images/part1_images/*"
file_name=""

for filename in $image_dir;
do
  #echo   $filename
  filename=" "$filename
  file_name=$file_name$filename
done;
#echo  ./a2 part1 $query $ratio$file_name
echo ./a2 part1 $query$file_name
