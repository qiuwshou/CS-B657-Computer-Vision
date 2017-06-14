from function import *


#if youwant to write out the result, the directories should have the similar structure as below
#for example /some-name/some-name/some-name/data-set
dir_no_obstacles = "/Users/qiuwshou/image_sequences/no_obstacles"
dir_with_obstacles= "/Users/qiuwshou/image_sequences/with_obstacles"



#parameter:
#dir of non-obstacle data
#dir of obstacle data
#lower bound of feature size (feature size should be larger than this number)
#scale ratio to initialze the template (initial template size = this number * feature size)
#score ratio to validate the performance of template matching (scaled template matching < this number* unscaled template matching)
#lower bound of the best scale (best scale should be lareger than this number)
#1 to write out the result of matching, 0 to not
detector1=Obstacle_detctor(dir_no_obstacles,dir_with_obstacles,30,1.2,0.8,1.2,0)
detector1.compare_all()
