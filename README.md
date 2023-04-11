# Concept Drift Datasets v1.0
## Background
**Concept drift** describes unforeseeable changes in the underlying distribution of streaming data over time[1]. Concept drift problem exists in many real-world situations, such as sensor drift and the change of operating mode[2][3]. Detecting concept drift timely and accurately is of great significance for judging system state and providing decision suggestions[4]. In order to better test and evaluate the performance of concept drift detection algorithm, we have made some datasets with known drift types and drift time points, hoping to help the development of concept drift detection.
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

- Or you can download *DatasetsInput.py*, and then import the class, as shown in *DatasetsInput_main.py*.(**Recommended**)
- Example:
```
from DatasetsInput import Datasets

Data = Datasets()

X, Y = Data.CNNS_Nonlinear_Gradual_ChocolateRotation()
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
We have made four categories of datasets, including *linear*, *rotating cake*, *rotating chocolate* and *rolling torus*. All of them contain four types of drifts: *Abrupt*, *Sudden*, *Gradual* and *Recurrent*. Users can choose whether to draw distribution of samples in real time, save pictures, and make sample change videos. Users can choose whether to add noise or redundant variables as well. See the picture below for a more detailed introduction. Note that the dataset name in the following figure is also the name of the intra class function.  
<div align=center><img src="https://github.com/songqiaohu/pictureandgif/blob/main/QQ%E5%9B%BE%E7%89%8720230105200626.png"/></div>  

### Linear
In the dataset *Linear*, the decision boundary is a straight line. We simulate the change of the decision boundary through the rotation of the straight line. Users can freely select the rotation axis within the range of [-10, 10]×[-10, 10].
- Data distribution display:
<div align=center><img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_linear_gradual_rotation_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px"/>

<img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_linear_sudden_rotation_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px"/>
</div>  
<p align="center">(a)Gradual&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;(b)Sudden</p>
<div align=center><img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_linear_recurrent_rotation_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px" alt="Gradual"/>
 
<img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_linear_abrupt_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px"/>
</div>  
<p align="center">(c)Recurrent&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;(d)Abrupt</p>


### CakeRotation
In the dataset *CakeRotation*, samples with odd angle area belong to one class, while samples with even angle area belong to another class. We simulate concept drift by rotating the disk, and the range of the angle area will change during the rotation. **If you need data sets of multiple categories, you can achieve it by using modulus instead of odd and even numbers on this basis[5].**
- Data distribution display:
<div align=center><img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_nonlinear_gradual_cakerotation_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px"/>

<img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_nonlinear_sudden_cakerotation_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px"/>
</div>  
<p align="center">(a)Gradual&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;(b)Sudden</p>
<div align=center><img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_nonlinear_recurrent_cakerotation_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px" alt="Gradual"/>
 
<img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_nonlinear_abrupt_cakerotation_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px"/>
</div>  
<p align="center">(c)Recurrent&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;(d)Abrupt</p>

### ChocolateRotation
In the dataset *ChocolateRotation*, samples with odd *x+y* area belong to one class, while samples with even angle area belong to another class. We simulate concept drift by rotating the chocolate plate, and use the rotation matrix to calculate the coordinates of the samples in the new coordinate system and reclassify them. **If you need data sets of multiple categories, you can achieve it by using modulus instead of odd and even numbers on this basis.**
- Data distribution display:
<div align=center><img src="https://github.com/songqiaohu/pictureandgif/blob/main/nonlinear_gradual_chocolaterotation_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px"/>

<img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_nonlinear_sudden_chocolaterotation_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px"/>
</div>  
<p align="center">(a)Gradual&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;(b)Sudden</p>
<div align=center><img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_nonlinear_recurrent_chocolaterotation_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px" alt="Gradual"/>
 
<img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_nonlinear_abrupt_chocolaterotation_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px"/>
</div>  
<p align="center">(c)Recurrent&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;(d)Abrupt</p>

### RollingTorus
In the dataset *RollingTorus*, we set two torus of the same size close together, and the samples in different torus belong to different classes. We let the third torus roll over at a constant speed, and the samples overlapping the first two tori will become the opposite category. **If you need a dataset with unbalanced number of category samples, you can adjust the initial torus radius to achieve[6].**
- Data distribution display:
<div align=center><img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_nonlinear_gradual_rollingtorus_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px"/>

<img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_nonlinear_sudden_rollingtorus_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px"/>
</div>  
<p align="center">(a)Gradual&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;(b)Sudden</p>
<div align=center><img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_nonlinear_recurrent_rollingtorus_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px" alt="Gradual"/>
 
<img src="https://github.com/songqiaohu/pictureandgif/blob/main/figure_nonlinear_abrupt_rollingtorus_noise_and_redunce.gif?raw=true" width="320px" hegiht="240px"/>
</div>  
<p align="center">(c)Recurrent&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;&#8195;(d)Abrupt</p>

## Change Log
### v1.0(2023/1/2)

## About us
- Welcome to discuss and put forward valuable suggestions. Email: liuzy21@mails.tsinghua.edu.cn, 1120193091@bit.edu.cn.

[1] Lu J, Liu A, Dong F, et al. Learning under concept drift: A review[J]. IEEE Transactions on Knowledge and Data Engineering, 2018, 31(12): 2346-2363.  
[2] Liu Z, Zhang Y, Ding Z, et al. “An Online Active Broad Learning Approach for Real-Time Safety Assessment of Dynamic Systems in Nonstationary Environments,” IEEE Transactions on Neural Networks and Learning Systems, 2022.  
[3] Hoens T R, Polikar R, Chawla N V. “Learning from streaming data
with concept drift and imbalance: an overview,” Progress in Artificial
Intelligence, 2012, 1(1): 89-101.  
[4] Liu A, Song Y, Zhang G, et al. “Regional concept drift detection
and density synchronized drift adaptation,” IJCAI International Joint
Conference on Artificial Intelligence. 2017.  
[5] Korycki Ł, Krawczyk B. “Concept drift detection from multi-class
imbalanced data streams,” 2021 IEEE 37th International Conference on
Data Engineering (ICDE). IEEE, 2021: 1068-1079.  
[6] Wang S, Minku L L, Yao X. “A systematic study of online class
imbalance learning with concept drift,” IEEE transactions on neural
networks and learning systems, 2018, 29(10): 4802-4821.  
[7]López Lobo, Jesús. "Synthetic datasets for concept drift detection purposes." Harv. Dataverse (2020). https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/5OWRGB.
