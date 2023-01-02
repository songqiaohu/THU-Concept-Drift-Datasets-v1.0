from DataStreamGenerator import DataStreamGenerator
from skmultiflow.data import DataStream


C = DataStreamGenerator(class_count=2, attribute_count=2, sample_count=100000, noise=True, redunce_variable=True)
X, Y = C.Nonlinear_Sudden_RollingTorus(plot=True, save=True)
strem = DataStream(X, Y)
print(strem.next_sample(100))