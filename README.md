# install
```
catkin bt
pip3 install -e . -v
```

# how to run
Print 7x7 cm april tag using https://github.com/HiroIshida/print_apriltag

Running perception node
```
roslaunch frmax_ros perception.launch
```

train
```
python3 hubo_mugcup  # may resume with --resume option
```
refine 
```
python3 hubo_mugcup.py --refine
```

# apt requirement
festival
