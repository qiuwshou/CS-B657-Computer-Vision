#!/bin/bash

image_dir1="a2-images/part2_images/seq1/*"
image_dir2="a2-images/part2_images/seq2/*"
file_name=""
file_name=""
for filename in $image_dir1;
do
  #echo   $filename
  filename=" "$filename
  file_name=$file_name$filename
done;
echo  ./a2 part2.2$file_name

for filename2 in $image_dir2;
do
  #echo   $filename
  filename2=" "$filename2
  file_name2=$file_name2$filename2
done;
echo  ./a2 part2.2$file_name2


