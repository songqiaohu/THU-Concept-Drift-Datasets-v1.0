import pandas as pd
class Datasets(object):
    def __init__(self):
        return

    def Read_Csv(self, url):
        print('please wait, csv is loading...')
        data = pd.read_csv(url)
        data = data.values
        l, w = data.shape
        X = data[:, 0 : w - 1]
        Y = data[:, w - 1]
        return X, Y
    def CNNS_Linear_Abrupt(self):
        return Datasets.Read_Csv(self, 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/linear_abrupt_noise_and_redunce.csv')

    def CNNS_Linear_Gradual_Rotation(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/linear_gradual_rotation_noise_and_redunce.csv')

    def CNNS_Linear_Sudden_Rotation(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/linear_sudden_rotation_noise_and_redunce.csv')

    def CNNS_Linear_Recurrent_Rotation(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/linear_recurrent_rotation_noise_and_redunce.csv')

    def CNNS_Nonlinear_Abrupt_CakeRotation(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/nonlinear_abrupt_cakerotation_noise_and_redunce.csv')

    def CNNS_Nonlinear_Gradual_CakeRotation(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/nonlinear_gradual_cakerotation_noise_and_redunce.csv')

    def CNNS_Nonlinear_Sudden_CakeRotation(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/nonlinear_sudden_cakerotation_noise_and_redunce.csv')

    def CNNS_Nonlinear_Recurrent_CakeRotation(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/nonlinear_recurrent_cakerotation_noise_and_redunce.csv')

    def CNNS_Nonlinear_Abrupt_ChocolateRotation(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/nonlinear_abrupt_chocolaterotation_noise_and_redunce.csv')

    def CNNS_Nonlinear_Gradual_ChocolateRotation(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/nonlinear_gradual_chocolaterotation_noise_and_redunce.csv')

    def CNNS_Nonlinear_Sudden_ChocolateRotation(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/nonlinear_sudden_chocolaterotation_noise_and_redunce.csv')

    def CNNS_Nonlinear_Recurrent_ChocolateRotation(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/nonlinear_recurrent_chocolaterotation_noise_and_redunce.csv')

    def CNNS_Nonlinear_Abrupt_RollingTorus(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/nonlinear_abrupt_rollingtorus_noise_and_redunce.csv')

    def CNNS_Nonlinear_Gradual_RollingTorus(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/nonlinear_gradual_rollingtorus_noise_and_redunce.csv')

    def CNNS_Nonlinear_Sudden_RollingTorus(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/nonlinear_sudden_rollingtorus_noise_and_redunce.csv')

    def CNNS_Nonlinear_Recurrent_RollingTorus(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/nonlinear_recurrent_rollingtorus_noise_and_redunce.csv')

    def Harvard_mixed_0101_abrupto(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/mixed_0101_abrupto.csv')

    def Harvard_mixed_0101_gradual(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/mixed_0101_gradual.csv')

    def Harvard_mixed_1010_abrupto(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/mixed_1010_abrupto.csv')

    def Harvard_mixed_1010_gradual(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/mixed_1010_gradual.csv')

    def Harvard_rt_2563789698568873_abrupto(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/rt_2563789698568873_abrupto.csv')

    def Harvard_rt_2563789698568873_gradual(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/rt_2563789698568873_gradual.csv')

    def Harvard_rt_8873985678962563_abrupto(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/rt_8873985678962563_abrupto.csv')

    def Harvard_rt_8873985678962563_gradual(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/rt_8873985678962563_gradual.csv')

    def Harvard_sea_0123_abrupto_noise_0dot2(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/sea_0123_abrupto_noise_0.2.csv')

    def Harvard_sea_0123_gradual_noise_0dot2(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/sea_0123_gradual_noise_0.2.csv')

    def Harvard_sea_3210_gradual_noise_0dot2(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/sea_3210_gradual_noise_0.2.csv')

    def Harvard_sea_3210_abrupto_noise_0dot2(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/sea_3210_abrupto_noise_0.2.csv')

    def Harvard_sine_0123_abrupto(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/sine_0123_abrupto.csv')

    def Harvard_sine_0123_gradual(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/sine_0123_gradual.csv')

    def Harvard_sine_3210_abrupto(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/sine_3210_abrupto.csv')

    def Harvard_sine_3210_gradual(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/sine_3210_gradual.csv')

    def Harvard_stagger_0120_abrupto(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/stagger_0120_abrupto.csv')

    def Harvard_stagger_0120_gradual(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/stagger_0120_gradual.csv')

    def Harvard_stagger_2102_abrupto(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/stagger_2102_abrupto.csv')

    def Harvard_stagger_2102_gradual(self):
        return Datasets.Read_Csv(self,
                                 'https://raw.githubusercontent.com/songqiaohu/THU-Concept-Drift-Datasets-v1.0/main/Harvard_Concept_Drift_Datasets/stagger_2102_gradual.csv')

