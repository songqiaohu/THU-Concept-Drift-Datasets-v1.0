from DatasetsInput import Datasets
from skmultiflow.data import DataStream

Data = Datasets()
X, Y = Data.Harvard_mixed_0101_abrupto()
'''
you can choose
CNNS_Linear_Abrupt()
CNNS_Linear_Gradual_Rotation()
CNNS_Linear_Sudden_Rotation()
CNNS_Linear_Recurrent_Rotation()
CNNS_Nonlinear_Abrupt_CakeRotation()
CNNS_Nonlinear_Gradual_CakeRotation()
CNNS_Nonlinear_Sudden_CakeRotation()
CNNS_Nonlinear_Recurrent_CakeRotation()
CNNS_Nonlinear_Abrupt_ChocolateRotation()
CNNS_Nonlinear_Gradual_ChocolateRotation()
CNNS_Nonlinear_Sudden_ChocolateRotation()
CNNS_Nonlinear_Recurrent_ChocolateRotation()
CNNS_Nonlinear_Abrupt_RollingTorus()
CNNS_Nonlinear_Gradual_RollingTorus()
CNNS_Nonlinear_Sudden_RollingTorus()
CNNS_Nonlinear_Recurrent_RollingTorus()
Harvard_mixed_0101_abrupto()
Harvard_mixed_0101_gradual()
Harvard_mixed_1010_abrupto()
Harvard_mixed_1010_gradual()
Harvard_rt_2563789698568873_abrupto()
Harvard_rt_2563789698568873_gradual()
Harvard_rt_8873985678962563_abrupto()
Harvard_rt_8873985678962563_gradual()
Harvard_sea_0123_abrupto_noise_0dot2()
Harvard_sea_0123_gradual_noise_0dot2()
Harvard_sea_3210_gradual_noise_0dot2()
Harvard_sea_3210_abrupto_noise_0dot2()
Harvard_sine_0123_abrupto()
Harvard_sine_0123_gradual()
Harvard_sine_3210_abrupto()
Harvard_sine_3210_gradual()
Harvard_stagger_0120_abrupto()
Harvard_stagger_0120_gradual()
Harvard_stagger_2102_abrupto()
Harvard_stagger_2102_gradual()
'''
stream = DataStream(X, Y)
print(stream.next_sample(100))