# -*- encoding: utf-8 -*-
"""
 Created by Ênio Viana at 10/10/2021 at 16:20:46
 Project: processos-estocasticos [out, 2021]
"""
import matplotlib.pyplot as plt
import numpy as np


class Plot:

    @staticmethod
    def plot_histogram(vector: np.ndarray, sigma: [int, float], mu: [int, float], title: str):
        count, bins, ignored = plt.hist(vector, 30, density=True)
        plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
                 np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
                 linewidth=2, color='r')
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_scatter(vector_x: np.ndarray, vector_y: np.ndarray, title: str):
        plt.scatter(vector_x, vector_y, s=50, alpha=0.5)
        plt.title(title)
        plt.show()
