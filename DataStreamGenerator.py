#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import csv
import sympy as sp
import matplotlib.animation as animation
import os
import cv2
import random
from matplotlib.pyplot import MultipleLocator

class DataStreamGenerator(object):

    def __init__(self,
                 class_count : int = 2,
                 attribute_count : int = 2,
                 sample_count : int = 100000,
                 noise : bool = False,
                 redunce_variable : bool = False):
        """
        Set the parameters of the data stream.

        Args:
            class_count (int): The count of classes
            attribute_count (int): The count of attributes of the data
        """
        self.class_count = class_count
        self.attribute_count = attribute_count
        self.sample_count = sample_count
        self.condition_count = 0
        self.noise = noise
        self.redunce_variable = redunce_variable

    def Linear_Conditions(self, decision_variables : list, type : str = "abrupt"):
        if "abrupt" in type:
            if self.condition_count == 1 :
                return decision_variables[1] - decision_variables[3] - (decision_variables[0] - decision_variables[2]) >= 0
            else:
                return decision_variables[1] - decision_variables[3] - (decision_variables[0] - decision_variables[2]) <= 0
        if "gradual" in type or "sudden" in type or "recurrent" in type:
            if np.pi /4 - self.condition_count / self.sample_count * np.pi >= - np.pi /2:
                if (np.pi /4 - self.condition_count / self.sample_count * np.pi == - np.pi /2) or \
                    (np.pi / 4 - self.condition_count / self.sample_count * np.pi == np.pi / 2):
                    return decision_variables[0] - decision_variables[2]>= 0
                else:
                    return decision_variables[1] - decision_variables[3] - \
                           (decision_variables[0] -decision_variables[2])\
                            * np.tan(np.pi /4 - self.condition_count / self.sample_count * np.pi) >= 0
            else:
                if (np.pi / 4 - self.condition_count / self.sample_count * np.pi == - np.pi / 2) or \
                        (np.pi / 4 - self.condition_count / self.sample_count * np.pi == np.pi / 2):
                    return decision_variables[0] - decision_variables[2] <= 0
                else:
                    return decision_variables[1] - decision_variables[3]- \
                           (decision_variables[0] - decision_variables[2]) \
                           * np.tan(np.pi / 4 - self.condition_count / self.sample_count * np.pi) <= 0

    def Plot_Decision_Boundary(self, equation : str, type : str, variables : list = []):
        """  e.g. equation = "2*x-y+1=0" ,the right of the equation must be 0"""
        if "abrupt" in type and "nonlinear" not in type:
            equation = equation.replace("=0", "")
            x = np.linspace(-10, 10, 20)
            y_symbol = sp.symbols('y')
            y = np.array([sp.solve([equation.replace('x', str(i))], [y_symbol])[y_symbol] for i in x])
            index = np.where((y >= -10) & (y <= 10))
            x = x[index]
            y = y[index]
            plt.plot(x, y, 'r', label='decision boundary')

        if ("gradual" in type or "sudden" in type or "recurrent" in type) and "nonlinear" not in type:
            if len(variables) == 2 and variables[0] == -1:
                print("plot_decisionboundary")
                y = np.linspace(-10, 10, 20)
                x = np.array([variables[1] for i in y])
                plt.plot(x, y, 'r', label='decision boundary')
            else:
                equation = equation.replace("=0", "")
                x = np.linspace(-10, 10, 20)
                y_symbol = sp.symbols('y')
                y = np.array([sp.solve([equation.replace('x', str(i))], [y_symbol])[y_symbol] for i in x])
                index = np.where((y >= -10) & (y <= 10))
                x = x[index]
                y = y[index]
                plt.plot(x, y, 'r', label='decision boundary')
                plt.scatter(variables[0], variables[1], c='r', label='spin axis', s=25)

        if ("gradual" in type or "sudden" in type or "abrupt" in type or "recurrent" in type) and "nonlinear" in type and "cake" in type:

            line_k = [i - variables[0] for i in range(30, 210, 30)]
            r = np.linspace(-10, 10, 10)
            first = True
            for i in line_k:
                x = r * np.cos(i * np.pi / 180)
                y = r * np.sin(i * np.pi / 180)
                if first:
                    plt.plot(x, y, 'r', label="decision boundary", linewidth=0.8)
                    first = False
                else:
                    plt.plot(x, y, 'r', linewidth=0.8)
            theta = np.linspace(0, 2 * np.pi, 50)
            x = 10 * np.cos(theta)
            y = 10 * np.sin(theta)
            plt.plot(x, y, 'c')
            plt.axis('equal')

        if ("gradual" in type or "sudden" in type or "abrupt" in type or "recurrent" in type) and "nonlinear" in type and "chocolate" in type:
            rotate_matrix = np.array(
                [[np.cos(variables[0]), -np.sin(variables[0])],
                 [np.sin(variables[0]), np.cos(variables[0])]]).T
            x_line = [-10, -5, 0, 5, 10]
            y_line = [-10, -5, 0, 5, 10]
            first = True
            x = np.linspace(-12, 12, 10).reshape(1, 10)
            for i in x_line:
                y = np.array([i for j in x[0]]).reshape(1, 10)
                xy = np.row_stack((x, y))
                xy_rotation = rotate_matrix.dot(xy)
                if first:
                    plt.plot(xy_rotation[0, :], xy_rotation[1, :], 'k', label="decision boundary", alpha=0.6)
                    first = False
                else:
                    plt.plot(xy_rotation[0, :], xy_rotation[1, :], 'k', alpha=0.6)
            y = np.linspace(-12, 12, 10).reshape(1, 10)
            for i in y_line:
                x = np.array([i for j in y[0]]).reshape(1, 10)
                xy = np.row_stack((x, y))
                xy_rotation = rotate_matrix.dot(xy)
                plt.plot(xy_rotation[0, :], xy_rotation[1, :], 'k', alpha=0.6)

            # plt.axis('equal')
            plt.xlim(-15, 15)
            plt.ylim(-15, 15)

        if ("gradual" in type or "sudden" in type or "abrupt" in type or "recurrent" in type) and "nonlinear" in type and "rollingtorus" in type:
            distance = variables[0]
            r_torus = 10
            theta = np.linspace(0, 2* np.pi, 50)
            x_torus_orin = r_torus * np.cos(theta)
            y_torus_orin = r_torus * np.sin(theta)
            x_torus1 = x_torus_orin - 10
            y_torus1 = y_torus_orin + 10
            x_torus2 = x_torus_orin + 10
            y_torus2 = y_torus_orin + 10
            x_torus_rolling = x_torus_orin - 35 + distance
            y_torus_rolling = y_torus_orin + 10
            plt.xlim(-50, 50)
            # plt.ylim(0, 30)
            plt.axis('equal')
            plt.plot(x_torus1, y_torus1, 'b', alpha=0.35)
            plt.plot(x_torus2, y_torus2, 'g', alpha=0.35)
            plt.plot(x_torus_rolling, y_torus_rolling, 'r', label="decision boundary")
            plt.grid()


    def Plot(self, data, label, decision_boundary : str, type : str = "abrupt", variables : list = [], save_fig : bool = False):

        if save_fig:
            if not os.path.exists("./figure_{}".format(type)):
                os.mkdir("./figure_{}".format(type))
        size = 17 * np.ones(1000)

        for i in range(100):
            print(i)
            plt.clf()
            X = data[1000 * i : 1000 * (i + 1), 0]
            Y = data[1000 * i : 1000 * (i + 1), 1]
            label2 = label[1000 * i : 1000 * (i + 1), 0]

            area0 = np.ma.masked_where(label2 == 1, size)
            area1 = np.ma.masked_where(label2 == 0, size)
            plt.scatter(X, Y, s=area0, c="hotpink", edgecolors="g", label="class0", cmap='viridis', alpha=0.7)
            plt.scatter(X, Y, s=area1, c="#88c999", edgecolors="blue", label="class1", cmap='viridis', alpha=0.7)
            if "gradual" in type and "nonlinear" not in type:
                if np.pi /4 - i * 1000 / self.sample_count * np.pi == -np.pi / 2:
                    DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [-1, variables[0]])
                else:
                    equation = decision_boundary.replace("a", str(variables[1]))
                    equation = equation.replace("k", str(np.tan(np.pi /4 - i * 1000 / self.sample_count * np.pi)))
                    equation = equation.replace("b", str(variables[0]))
                    DataStreamGenerator.Plot_Decision_Boundary(self, equation, type, [variables[0], variables[1]])
            if "sudden" in type and "nonlinear" not in type:

                if np.pi /4 - ((i * 1000) // (0.2 * self.sample_count)) * 0.2 * self.sample_count\
                        / self.sample_count * np.pi == -np.pi / 2:
                    DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [0, variables[0]])
                else:
                    equation = decision_boundary.replace("a", str(variables[1]))
                    equation = equation.replace("k", str(np.tan(np.pi /4 - ((i * 1000) // (0.2 * self.sample_count)) * 0.2 * self.sample_count\
                        / self.sample_count * np.pi)))
                    equation = equation.replace("b", str(variables[0]))
                    DataStreamGenerator.Plot_Decision_Boundary(self, equation, type, [variables[0], variables[1]])
            if "recurrent" in type and "nonlinear" not in type:

                if np.pi / 4 - ((i * 1000) // (0.1 * self.sample_count)) % 2 * 0.2 * np.pi == -np.pi / 2:
                    DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [0, variables[0]])
                else:
                    equation = decision_boundary.replace("a", str(variables[1]))
                    equation = equation.replace("k", str(np.tan(np.pi / 4 - ((i * 1000) // (0.1 * self.sample_count)) % 2 \
                                                                * 0.2 * np.pi)))
                    equation = equation.replace("b", str(variables[0]))
                    DataStreamGenerator.Plot_Decision_Boundary(self, equation, type, [variables[0], variables[1]])
            if "abrupt" in type and "nonlinear" not in type:
                equation = decision_boundary.replace("a", str(variables[1]))
                equation = equation.replace("k", str(1))
                equation = equation.replace("b", str(variables[0]))
                DataStreamGenerator.Plot_Decision_Boundary(self, equation, type)

            if "gradual" in type and "nonlinear" in type and "cake" in type:
                DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [i * 1000 / self.sample_count * 30])
            if "sudden" in type and "nonlinear" in type and "cake" in type:
                DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [((i * 1000) // (0.2 * self.sample_count)) * 6])
            if "abrupt" in type and "nonlinear" in type and "cake" in type:
                DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [0])
            if "recurrent" in type and "nonlinear" in type and "cake" in type:

                if ((i * 1000) // (0.1 * self.sample_count)) % 2 == 0:
                    DataStreamGenerator.Plot_Decision_Boundary(self, "", type,
                                                               [((i * 1000) % (0.1 * self.sample_count)) / \
                                                                (0.1 * self.sample_count) * 30])
                else:
                    DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [
                        (0.1 * self.sample_count - ((i * 1000) % (0.1 * self.sample_count))) / (
                                    0.1 * self.sample_count) * 30])

            if "gradual" in type and "nonlinear" in type and "chocolate" in type:
                DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [i * 1000 / self.sample_count * np.pi / 2])
            if "sudden" in type and "nonlinear" in type and "chocolate" in type:
                DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [((i * 1000) // (0.2 * self.sample_count)) * 0.2 * np.pi / 2])
            if "abrupt" in type and "nonlinear" in type and "chocolate" in type:
                DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [0])
            if "recurrent" in type and "nonlinear" in type and "chocolate" in type:

                if ((i * 1000) // (0.1 * self.sample_count)) % 2 == 0:
                    DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [((i * 1000) % (0.1 * self.sample_count)) / \
                                                                                (0.1 * self.sample_count)* 0.1 * 2 * np.pi])
                else:
                    DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [
                        (0.1 * self.sample_count - ((i * 1000) % (0.1 * self.sample_count))) / (0.1 * self.sample_count) * 0.1 * 2 * np.pi])

            if "gradual" in type and "nonlinear" in type and "rollingtorus" in type:
                DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [i * 1000 / self.sample_count * 70])
            if "sudden" in type and "nonlinear" in type and "rollingtorus" in type:
                DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [((i * 1000) // (0.2 * self.sample_count)) * 0.2 * 70])
            if "abrupt" in type and "nonlinear" in type and "rollingtorus" in type:
                DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [0])
            if "recurrent" in type and "nonlinear" in type and "rollingtorus" in type:
                if ((i * 1000) // (0.1 * self.sample_count)) % 2 == 0:
                    DataStreamGenerator.Plot_Decision_Boundary(self, "", type,
                                                               [((i * 1000) % (0.1 * self.sample_count)) / \
                                                                (0.1 * self.sample_count) * 14])
                else:
                    DataStreamGenerator.Plot_Decision_Boundary(self, "", type, [
                        (0.1 * self.sample_count - ((i * 1000) % (0.1 * self.sample_count))) / (
                                    0.1 * self.sample_count) * 14])


            plt.legend(loc=1)
            plt.title("t : {}-{}, {}".format(i * 1000, (i + 1) * 1000, type))
            plt.xlabel('$x$')
            plt.ylabel('$y$')
            if save_fig:
                plt.savefig('./figure_{}/{}.png'.format(type, i))
            plt.pause(0.1)
        plt.show()
        if save_fig:
            DataStreamGenerator.Vedio_Generator(self, "./figure_{}/".format(type))

    def Vedio_Generator(self, path : str):
        filelist = os.listdir(path)
        filelist2 = [str(i) + ".png" for i in range(len(filelist))]
        fps = 3  # 视频每秒3帧
        size = (640, 480)  # 需要转为视频的图片的尺寸，要和原图片大小保持一直
        video = cv2.VideoWriter(path + "Video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        for item in filelist2:
            if item.endswith('.png'):
                # 找到路径中所有后缀名为.png的文件，可以更换为.jpg或其它
                item = path + item
                img = cv2.imread(item)
                video.write(img)

        video.release()
        cv2.destroyAllWindows()

    def Save(self, data, label, csvname : str):
        with open(csvname, 'w', encoding='utf-8', newline='') as fp:
            writer = csv.writer(fp)
            if self.redunce_variable:
                writer.writerows(np.array([['x1', 'x2', 'x3', 'x4', 'x4', 'label']]))
            else:
                writer.writerows(np.array([['x1', 'x2', 'label']]))
            writer.writerows(np.column_stack((data, label)))

    def Noise_And_Redunce(self, data, label, type : str):
        if self.redunce_variable and self.noise:
            redunce_variable1 = np.random.uniform(-10, 10, self.sample_count)
            redunce_variable2 = np.random.uniform(-10, 10, self.sample_count)
            redunce_variable3 = np.random.uniform(-10, 10, self.sample_count)
            data = np.column_stack((data, redunce_variable1, redunce_variable2, redunce_variable3))
            l, w = data.shape
            for i in range(l):
                for j in range(w):
                    data[i, j] = data[i, j] * np.random.normal(1, 0.02)
            DataStreamGenerator.Save(self, data, label, type + ".csv")
        if self.noise and not self.redunce_variable:
            l, w = data.shape
            for i in range(l):
                for j in range(w):
                    data[i, j] = data[i, j] * np.random.normal(1, 0.02)
            DataStreamGenerator.Save(self, data, label, type + ".csv")
        if self.redunce_variable and not self.noise:
            redunce_variable1 = np.random.uniform(-10, 10, self.sample_count)
            redunce_variable2 = np.random.uniform(-10, 10, self.sample_count)
            redunce_variable3 = np.random.uniform(-10, 10, self.sample_count)
            data = np.column_stack((data, redunce_variable1, redunce_variable2, redunce_variable3))
            DataStreamGenerator.Save(self, data, label, type + ".csv")
        return data

    def Adjust_Type(self, type : str):
        if self.noise and self.redunce_variable:
            type = type + "_noise_and_redunce"
        elif self.noise and not self.redunce_variable:
            type = type + "_noise"
        else:
            type = type + "_redunce"
        return type
    def Init_Sample_Linear(self):
        X = np.random.uniform(-10, 10, self.sample_count)
        Y = np.random.uniform(-10, 10, self.sample_count)
        data = np.column_stack((X, Y))
        l, w = data.shape
        label = np.zeros((l, 1))
        return data, label, l

    def Init_Sample_Circle(self):
        rr = np.random.uniform(0, 100, self.sample_count)
        theta = np.random.uniform(0, 2 * np.pi, self.sample_count)
        X = np.sqrt(rr) * np.cos(theta)
        Y = np.sqrt(rr) * np.sin(theta)  # to be uniform
        data = np.column_stack((X, Y))
        l, w = data.shape
        label = np.zeros((l, 1))
        return data, label, l

    def Linear_Abrupt(self, plot : bool = False, save : bool = False):
        type = "linear_abrupt"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Linear(self)
        self.condition_count = 1
        x_pass = float(input("please input a x_point the line passes from [-10, 10]:\n"))
        y_pass = float(input("please input a y_point the line passes from [-10, 10]:\n"))
        for i in range(l):
            if DataStreamGenerator.Linear_Conditions(self, [data[i, 0], data[i, 1], x_pass, y_pass], type):
                label[i] = 0
            else:
                label[i] = 1
            if i % 5000 == 0:
                self.condition_count *= -1

        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "y-a-k*(x-b)=0", type, [x_pass, y_pass], save)
        return data, label

    def Linear_Gradual_Rotation(self, plot : bool = False, save : bool = False):
        type = "linear_gradual_rotation"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Linear(self)
        x_spinaxis = float(input("please input the x_spinaxis from [-10, 10]:\n"))
        y_spinaxis = float(input("please input the y_spinaxis from [-10, 10]:\n"))

        for i in range(l):
            if DataStreamGenerator.Linear_Conditions(self, [data[i, 0], data[i, 1], x_spinaxis, y_spinaxis], type):
                label[i] = 0
            else:
                label[i] = 1
            self.condition_count += 1

        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "y-a-k*(x-b)=0", type, [x_spinaxis, y_spinaxis], save)
        return data, label

    def Linear_Sudden_Rotation(self, plot : bool = False, save : bool = False):
        type = "linear_sudden_rotation"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Linear(self)
        x_spinaxis = float(input("please input the x_spinaxis from [-10, 10]:\n"))
        y_spinaxis = float(input("please input the y_spinaxis from [-10, 10]:\n"))

        for i in range(l):
            if DataStreamGenerator.Linear_Conditions(self, [data[i, 0], data[i, 1], x_spinaxis, y_spinaxis], type):
                label[i] = 0
            else:
                label[i] = 1
            if i % (0.2 * self.sample_count) == 0 and i > 0:
                self.condition_count += 0.2 * self.sample_count

        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "y-a-k*(x-b)=0", type, [x_spinaxis, y_spinaxis], save)
        return data, label

    def Linear_Recurrent_Rotation(self, plot : bool = False, save : bool = False):
        type = "linear_recurrent_rotation"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Linear(self)
        x_spinaxis = float(input("please input the x_spinaxis from [-10, 10]:\n"))
        y_spinaxis = float(input("please input the y_spinaxis from [-10, 10]:\n"))

        for i in range(l):
            if DataStreamGenerator.Linear_Conditions(self, [data[i, 0], data[i, 1], x_spinaxis, y_spinaxis], type):
                label[i] = 0
            else:
                label[i] = 1
            if (i // (0.1 * self.sample_count) == i / (0.1 * self.sample_count)) \
                    and (i // (0.1 * self.sample_count)) % 2 == 1 and i > 0:
                self.condition_count += 0.2 * self.sample_count
            elif (i // (0.1 * self.sample_count) == i / (0.1 * self.sample_count)) \
                    and (i // (0.1 * self.sample_count)) % 2 == 0 and i > 0:
                self.condition_count -= 0.2 * self.sample_count
        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "y-a-k*(x-b)=0", type, [x_spinaxis, y_spinaxis], save)
        return data, label

    def Nonlinear_Conditions(self, decision_variables : list, type : str):
        if "nonlinear_gradual_cakerotation" in type or "nonlinear_sudden_cakerotation" in type \
                or "nonlinear_recurrent_cakerotation" in type:
            # print('condition', self.condition_count)
            angle = np.arctan2(decision_variables[1], decision_variables[0]) * 180 / np.pi
            if angle < 0:
                angle = 360 + angle
            angle += self.condition_count / self.sample_count * 30

            if (angle // 30) % 2 ==0:
                return True
            else:
                return False
        if "nonlinear_abrupt_cakerotation" in type:
            # print('condition', self.condition_count)
            angle = np.arctan2(decision_variables[1], decision_variables[0]) * 180 / np.pi
            if angle < 0:
                angle = 360 + angle
            if self.condition_count == 1:
                if (angle // 30) % 2 == 0:
                    return True
                else:
                    return False
            else:
                if (angle // 30) % 2 == 0:
                    return False
                else:
                    return True
        if "nonlinear_gradual_chocolaterotation" in type or "nonlinear_sudden_chocolaterotation" in type \
                or "nonlinear_recurrent_chocolaterotation" in type:
            rotate_matrix = rotate_matrix = np.array(
                [[np.cos(decision_variables[2]), -np.sin(decision_variables[2])],
                 [np.sin(decision_variables[2]), np.cos(decision_variables[2])]])
            # print(decision_variables[2])
            # print("decision", decision_variables[0], decision_variables[1])
            xy_rotate = rotate_matrix.dot(np.array([[decision_variables[0]], [decision_variables[1]]]))
            # print("xy_rotate", xy_rotate[0][0], xy_rotate[1][0])
            if (xy_rotate[0][0] // 5 + xy_rotate[1][0] // 5) % 2 == 0:
                return True
            else:
                return False
        if "nonlinear_abrupt_chocolaterotation" in type:
            if self.condition_count == 1:
                if (decision_variables[0] // 5 + decision_variables[1] // 5) % 2 == 0:
                    return True
                else:
                    return False
            else:
                if (decision_variables[0] // 5 + decision_variables[1] // 5) % 2 == 0:
                    return False
                else:
                    return True
        if "nonlinear_gradual_rollingtorus" in type or "nonlinear_sudden_rollingtorus" in type \
                or "nonlinear_recurrent_rollingtorus" in type:

            x_rollingtorus = -35 + self.condition_count / self.sample_count * 70
            y_rollingtorus = 10
            if (decision_variables[0] - x_rollingtorus) ** 2 + (decision_variables[1] - y_rollingtorus) ** 2 <= 100:
                if (decision_variables[0] - 10) ** 2 + (decision_variables[1] - 10) ** 2 <= 100:
                    return False
                else:
                    return True
            else:
                if (decision_variables[0] - 10) ** 2 + (decision_variables[1] - 10) ** 2 <= 100:
                    return True
                else:
                    return False

        if "nonlinear_abrupt_rollingtorus" in type :

            x_rollingtorus = -35
            y_rollingtorus = 10
            if self.condition_count == 1:
                if (decision_variables[0] - x_rollingtorus) ** 2 + (decision_variables[1] - y_rollingtorus) ** 2 <= 100:
                    if (decision_variables[0] - 10) ** 2 + (decision_variables[1] - 10) ** 2 <= 100:
                        return False
                    else:
                        return True
                else:
                    if (decision_variables[0] - 10) ** 2 + (decision_variables[1] - 10) ** 2 <= 100:
                        return True
                    else:
                        return False
            else:
                if (decision_variables[0] - x_rollingtorus) ** 2 + (decision_variables[1] - y_rollingtorus) ** 2 <= 100:
                    if (decision_variables[0] - 10) ** 2 + (decision_variables[1] - 10) ** 2 <= 100:
                        return True
                    else:
                        return False
                else:

                    if (decision_variables[0] - 10) ** 2 + (decision_variables[1] - 10) ** 2 <= 100:
                        return False
                    else:
                        return True
    def Nonlinear_Gradual_CakeRotation(self, plot : bool = False, save : bool = False):
        type = "nonlinear_gradual_cakerotation"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Circle(self)

        for i in range(l):
            if DataStreamGenerator.Nonlinear_Conditions(self, [data[i, 0], data[i, 1]], type):
                label[i] = 0
            else:
                label[i] = 1
            self.condition_count += 1
        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "", type, [0], save)
        return data, label

    def Nonlinear_Sudden_CakeRotation(self, plot : bool = False, save : bool = False):
        type = "nonlinear_sudden_cakerotation"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Circle(self)

        for i in range(l):
            # print('i', i, 'condition', self.condition_count)
            if DataStreamGenerator.Nonlinear_Conditions(self, [data[i, 0], data[i, 1]], type):
                label[i] = 0
            else:
                label[i] = 1
            if i > 0 and i % (0.2 * self.sample_count) == 0:
                self.condition_count += 0.2 * self.sample_count
        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "", type, [0], save)
        return data, label

    def Nonlinear_Abrupt_CakeRotation(self, plot : bool = False, save : bool = False):
        type = "nonlinear_abrupt_cakerotation"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Circle(self)
        self.condition_count = 1
        for i in range(l):
            # print('i', i, 'condition', self.condition_count)
            if DataStreamGenerator.Nonlinear_Conditions(self, [data[i, 0], data[i, 1]], type):
                label[i] = 0
            else:
                label[i] = 1
            if i > 0 and i % (0.2 * self.sample_count) == 0:
                self.condition_count *= -1
        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "", type, [0], save)
        return data, label

    def Nonlinear_Recurrent_CakeRotation(self, plot : bool = False, save : bool = False):
        type = "nonlinear_recurrent_cakerotation"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Circle(self)
        for i in range(l):

            if DataStreamGenerator.Nonlinear_Conditions(self, [data[i, 0], data[i, 1]], type):
                label[i] = 0
            else:
                label[i] = 1
            if (i // (0.1 * self.sample_count)) % 2 == 0:
                self.condition_count += 10
            else:
                self.condition_count -= 10
        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "", type, [0], save)
        return data, label


    def Nonlinear_Gradual_RollingTorus(self, plot : bool = False, save : bool = False):
        type = "nonlinear_gradual_rollingtorus"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Circle(self)
        for i in range(l):
            prob = random.random()
            if prob <= 0.5:
                data[i, 0] += 10
                data[i, 1] += 10
            else:
                data[i, 0] -= 10
                data[i, 1] += 10
        for i in range(l):
            if DataStreamGenerator.Nonlinear_Conditions(self, [data[i, 0], data[i, 1]], type):
                label[i] = 0
            else:
                label[i] = 1
            self.condition_count += 1
        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "", type, [0], save)
        return data, label

    def Nonlinear_Sudden_RollingTorus(self, plot: bool = False, save: bool = False):
        type = "nonlinear_sudden_rollingtorus"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Circle(self)
        for i in range(l):
            prob = random.random()
            if prob <= 0.5:
                data[i, 0] += 10
                data[i, 1] += 10
            else:
                data[i, 0] -= 10
                data[i, 1] += 10
        for i in range(l):
            if DataStreamGenerator.Nonlinear_Conditions(self, [data[i, 0], data[i, 1]], type):
                label[i] = 0
            else:
                label[i] = 1
            if i > 0 and i % (0.2 * self.sample_count) == 0:
                self.condition_count += 0.2 * self.sample_count
        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "", type, [0], save)
        return data, label

    def Nonlinear_Abrupt_RollingTorus(self, plot: bool = False, save: bool = False):
        type = "nonlinear_abrupt_rollingtorus"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Circle(self)
        self.condition_count = 1
        for i in range(l):
            prob = random.random()
            if prob <= 0.5:
                data[i, 0] += 10
                data[i, 1] += 10
            else:
                data[i, 0] -= 10
                data[i, 1] += 10
        for i in range(l):
            if DataStreamGenerator.Nonlinear_Conditions(self, [data[i, 0], data[i, 1]], type):
                label[i] = 0
            else:
                label[i] = 1
            if i > 0 and i % (0.2 * self.sample_count) == 0:
                self.condition_count *= -1
        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "", type, [0], save)
        return data, label

    def Nonlinear_Recurrent_RollingTorus(self, plot: bool = False, save: bool = False):
        type = "nonlinear_recurrent_rollingtorus"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Circle(self)
        for i in range(l):
            prob = random.random()
            if prob <= 0.5:
                data[i, 0] += 10
                data[i, 1] += 10
            else:
                data[i, 0] -= 10
                data[i, 1] += 10
        for i in range(l):
            if DataStreamGenerator.Nonlinear_Conditions(self, [data[i, 0], data[i, 1]], type):
                label[i] = 0
            else:
                label[i] = 1
            if (i // (0.1 * self.sample_count)) % 2 == 0:
                self.condition_count += 2
            else:
                self.condition_count -= 2
        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "", type, [0], save)
        return data, label

    def Nonlinear_Gradual_ChocolateRotation(self, plot : bool = False, save : bool = False):
        type = "nonlinear_gradual_chocolaterotation"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Linear(self)

        for i in range(l):
            if DataStreamGenerator.Nonlinear_Conditions(self, [data[i, 0], data[i, 1],
                                                               self.condition_count / \
                                                               self.sample_count * np.pi / 2], type):
                label[i] = 0
            else:
                label[i] = 1
            self.condition_count += 1
        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "", type, [0], save)
        return data, label

    def Nonlinear_Sudden_ChocolateRotation(self, plot : bool = False, save : bool = False):
        type = "nonlinear_sudden_chocolaterotation"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Linear(self)

        for i in range(l):
            if DataStreamGenerator.Nonlinear_Conditions(self, [data[i, 0], data[i, 1],
                                                               self.condition_count / \
                                                               self.sample_count * np.pi / 2], type):
                label[i] = 0
            else:
                label[i] = 1
            if i > 0 and i % (0.2 * self.sample_count) == 0:
                self.condition_count += 0.2 * self.sample_count
        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "", type, [0], save)
        return data, label

    def Nonlinear_Abrupt_ChocolateRotation(self, plot: bool = False, save: bool = False):
        type = "nonlinear_abrupt_chocolaterotation"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Linear(self)
        self.condition_count = 1
        for i in range(l):
            if DataStreamGenerator.Nonlinear_Conditions(self, [data[i, 0], data[i, 1],
                                                               self.condition_count / \
                                                               self.sample_count * np.pi / 2], type):
                label[i] = 0
            else:
                label[i] = 1
            if i > 0 and i % (0.2 * self.sample_count) == 0:
                self.condition_count *= -1
        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "", type, [0], save)
        return data, label

    def Nonlinear_Recurrent_ChocolateRotation(self, plot: bool = False, save: bool = False):
        type = "nonlinear_recurrent_chocolaterotation"
        type = DataStreamGenerator.Adjust_Type(self, type)
        data, label, l = DataStreamGenerator.Init_Sample_Linear(self)
        for i in range(l):

            if DataStreamGenerator.Nonlinear_Conditions(self, [data[i, 0], data[i, 1],
                                                               self.condition_count / \
                                                               self.sample_count * 2 * np.pi], type):
                label[i] = 0
            else:
                label[i] = 1
            if (i // (0.1 * self.sample_count)) % 2 == 0:
                self.condition_count += 1
            else:
                self.condition_count -= 1

        self.condition_count = 0
        data = DataStreamGenerator.Noise_And_Redunce(self, data, label, type)
        if plot == True:
            DataStreamGenerator.Plot(self, data, label, "", type, [0], save)
        return data, label







