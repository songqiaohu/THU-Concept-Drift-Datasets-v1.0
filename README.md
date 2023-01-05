# Concept Drift Datasets v1.0
## Background
**Concept drift** describes unforeseeable changes in the underlying distribution of streaming data over time[1]. Concept drift problem exists in many real-world situations, such as sensor drift and the change of operating mode. Detecting concept drift timely and accurately is of great significance for judging system state and providing decision suggestions. In order to better test and evaluate the performance of concept drift detection algorithm, we have made some datasets with known drift types and drift time points, hoping to help the development of concept drift detection.
## Usage
- If you want to use the datasets in the project, you can download them directly and import them using the pandas library.  
- Example:


```
import pandas as pd

data = pd.read_csv('xxxxxx/nonlinear_gradual_chocolaterotation_noise_and_redunce.csv')

data = data.values 

X = data[:, 0 : 5] 

Y = data[:, 5] 
``` 

- If you want to regenerate the dataset and import it directly, you can download *DataStreamGenerator.py* and put it under the file where your code is located, and then import the class.  
- Example:

```
from DataStreamGenerator import DataStreamGenerator

C = DataStreamGenerator(class_count=2, attribute_count=2, sample_count=100000, noise=True, redunce_variable=True)

X, Y = C.Nonlinear_Sudden_RollingTorus(plot=True, save=True)
``` 

- If you want to modify the source code, you can download it and do it in *DataStreamGenerator.py*.
## Dataset Introduction
We have made four categories of datasets, including *linear*, *rotating cake*, *rotating chocolate* and *rolling torus*. All of them contain four types of drifts: *Abrupt*, *Sudden*, *Gradual* and *Recurrent*. Users can choose whether to draw in real time, save pictures, and make sample change videos. See the picture below for a more detailed introduction. Note that the dataset name in the following figure is also the name of the intra class function.  
<div align=center><img src="https://github.com/songqiaohu/pictureandgif/blob/main/QQ%E5%9B%BE%E7%89%8720230105200626.png"/></div>  

### Linear
In the dataset *Linear*, the decision boundary is a straight line. We simulate the change of the decision boundary through the rotation of the straight line. The user can freely select the rotation axis within the range of [-10, 10]×[-10, 10].
- Example for the gradual drift:
<div align=center><img src="[https://github.com/songqiaohu/pictureandgif/blob/main/QQ%E5%9B%BE%E7%89%8720230105200626.png](https://github.com/songqiaohu/pictureandgif/blob/main/figure_linear_gradual_rotation_noise_and_redunce.gif?raw=true)"/></div>  

![image](https://github.com/songqiaohu/pictureandgif/blob/main/nonlinear_gradual_chocolaterotation_noise_and_redunce.gif?raw=true)
![image](https://github.com/songqiaohu/pictureandgif/blob/main/figure_nonlinear_gradual_rollingtorus_noise_and_redunce.gif?raw=true)
![image](https://github.com/songqiaohu/pictureandgif/blob/main/figure_nonlinear_gradual_cakerotation_noise_and_redunce.gif?raw=true)


[1]Lu J, Liu A, Dong F, et al. Learning under concept drift: A review[J]. IEEE Transactions on Knowledge and Data Engineering, 2018, 31(12): 2346-2363.
