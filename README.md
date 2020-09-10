# Multi-vehicles-see-through-effect-in-CARLA
## Introduction  
This repository realise a see-through effect in the autonomous driving simulator CARLA(https://github.com/carla-simulator/carla) using python.  
And to detect the front pedestrian or obstacle, a 3D bounding box detection method using Yolo v3(https://github.com/skhadem/3D-BoundingBox) is applied.  
The relative position of the two vehicles are estimated using GNSS data and refined with ICP(Iteractive closest point) method.  
The ideal diagram of this experiment is shown below:  
![Image](https://github.com/wangf98/Multi-vehicles-see-through-effect-in-CARLA/blob/master/pictures/Inked%E5%9B%BE%E7%89%871_LI.jpg)
  
However, limited by time, the part of get initial position data using EKF is not realized, the data is acquired directly using sensors with small error.  
The effect of the experiment is shown here:  
![Image](https://github.com/wangf98/Multi-vehicles-see-through-effect-in-CARLA/blob/master/pictures/result.png)  

## Requirements  
The requirements are listed in the file README.txt  
Pytorch, sklearn and some other basic python libraries are needed as well.  

## Usage   
Run CameraDetection.py to get the result  
