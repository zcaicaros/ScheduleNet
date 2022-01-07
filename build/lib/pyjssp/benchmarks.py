import numpy as np
from pyjssp.simulators import Simulator


def FT06(*args, **kwargs):

    machine_matrix = np.array([[2, 0, 1, 3, 5, 4],
                               [1, 2, 4, 5, 0, 3],
                               [2, 3, 5, 0, 1, 4],
                               [1, 0, 2, 3, 4, 5],
                               [2, 1, 4, 5, 0, 3],
                               [1, 3, 5, 0, 4, 2]])

    processing_time = np.array([[1., 3., 6., 7., 3., 6.],
                               [8., 5., 10., 10., 10., 4.],
                               [5., 4., 8., 9., 1., 7.],
                               [5., 5., 5., 3., 8., 9.],
                               [9., 3., 5., 4., 3., 1.],
                               [3., 3., 9., 10., 4., 1.]])

    num_job, num_machine = machine_matrix.shape

    sim = Simulator(num_machines=num_machine,
                    num_jobs=num_job,
                    name='FT06',
                    machine_matrix=machine_matrix,
                    processing_time_matrix=processing_time,
                    *args, **kwargs)

    return sim


def FT10(*args, **kwargs):

    machine_matrix = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                               [0, 2, 4, 9, 3, 1, 6, 5, 7, 8],
                               [1, 0, 3, 2, 8, 5, 7, 6, 9, 4],
                               [1, 2, 0, 4, 6, 8, 7, 3, 9, 5],
                               [2, 0, 1, 5, 3, 4, 8, 7, 9, 6],
                               [2, 1, 5, 3, 8, 9, 0, 6, 4, 7],
                               [1, 0, 3, 2, 6, 5, 9, 8, 7, 4],
                               [2, 0, 1, 5, 4, 6, 8, 9, 7, 3],
                               [0, 1, 3, 5, 2, 9, 6, 7, 4, 8],
                               [1, 0, 2, 6, 8, 9, 5, 3, 4, 7]])

    processing_time = np.array([[29., 78.,  9., 36., 49., 11., 62., 56., 44., 21.],
                                [43., 90., 75., 11., 69., 28., 46., 46., 72., 30.],
                                [91., 85., 39., 74., 90., 10., 12., 89., 45., 33.],
                                [81., 95., 71., 99.,  9., 52., 85., 98., 22., 43.],
                                [14.,  6., 22., 61., 26., 69., 21., 49., 72., 53.],
                                [84.,  2., 52., 95., 48., 72., 47., 65.,  6., 25.],
                                [46., 37., 61., 13., 32., 21., 32., 89., 30., 55.],
                                [31., 86., 46., 74., 32., 88., 19., 48., 36., 79.],
                                [76., 69., 76., 51., 85., 11., 40., 89., 26., 74.],
                                [85., 13., 61.,  7., 64., 76., 47., 52., 90., 45.]])

    num_job, num_machine = machine_matrix.shape

    sim = Simulator(num_machines=num_machine,
                    num_jobs=num_job,
                    name='FT10',
                    machine_matrix=machine_matrix,
                    processing_time_matrix=processing_time,
                    *args, **kwargs)

    return sim