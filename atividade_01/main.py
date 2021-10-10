# -*- encoding: utf-8 -*-
"""
 Created by Ênio Viana at 10/10/2021 at 15:35:37
 Project: processos-estocasticos [out, 2021]
"""

import numpy as np
from scipy import linalg

# Questão 01A
from Plot import Plot

TOTAL_SAMPLES = 5000

matriz_covariancia_xy = np.array([[4, 0], [0, 9]], dtype=np.float64)
esperancaX = 1
esperancaY = 2
mi = np.transpose(np.array([[1, 2]], dtype=np.float64))
matriz_correlacao_xy = matriz_covariancia_xy + mi * np.transpose(mi)

# Questão 01B
mu, sigma = 1, 2
xk = np.random.normal(mu, sigma, size=(TOTAL_SAMPLES, 1))
Plot.plot_histogram(xk, mu, sigma, r"Distribuição Gaussiana, $\mu$ = " + str(mu) + r" $\sigma$=" + str(sigma))

mu_2, sigma_2 = 2, 2
yk = np.random.normal(mu_2, sigma_2, size=(TOTAL_SAMPLES, 1))
Plot.plot_histogram(yk, mu_2, sigma_2, r"Distribuição Gaussiana, $\mu$ = " + str(mu_2) + r" $\sigma$=" + str(sigma_2))

# Questão 01C
s = np.array([[xk], [yk]])
Plot.plot_scatter(s[0, 0], s[1, 0], "Gráfico de Dispersão")

# Questão 01D
rows, cols, itens, _ = s.shape
# print(s.shape)
# print(s[0][0][1])
# print(s[0][0][99])
# print(s[1][0][99])
soma = np.zeros((rows, rows))
vetor_media = np.zeros([cols, rows])
# print(vetor_media.shape)
vetor_media[0, 0] = np.sum(s[0, 0]) / TOTAL_SAMPLES
vetor_media[0, 1] = np.sum(s[1, 0]) / TOTAL_SAMPLES
# print(vetor_media)

# Questão 01E
s1 = np.random.normal(0, 1, size=(TOTAL_SAMPLES, 2))
Plot.plot_scatter(s1[:, 0], s1[:, 1], "Gráfico de Dispersão")

# Questão 02A
cwr = np.array([[4, 1.5], [1.5, 9]])
x2 = np.random.randn(2, TOTAL_SAMPLES)
matrix_transferencia = np.transpose(linalg.cholesky(np.transpose(cwr)))
s2 = np.transpose(np.dot(matrix_transferencia, x2))

# Questão 02B
Plot.plot_scatter(s2[:, 0], s2[:, 1], "Gráfico de Dispersão")

# Questão 02C

# Questão 03A
rho = cwr[0, 1] / (np.sqrt((cwr[0, 0])) * np.sqrt(cwr[1, 1]))
print(rho)

# Questão 03B
covs2 = np.cov(s2)
pho = covs2[0, 1] / (np.sqrt(covs2[0, 0]) * np.sqrt(covs2[1, 1]))
print(pho)
