# import matplotlib.pyplot as plt
# import numpy as np
# import math
# import scipy.stats as stats

# def calculate_confidence_interval(sample_mean, sample_std, sample_size, confidence_level):
#     # 根据置信水平计算z值
#     z_score = stats.norm.ppf((1 + confidence_level) / 2)
#     margin_of_error = z_score * (sample_std / math.sqrt(sample_size))
#     lower_bound = sample_mean - margin_of_error
#     upper_bound = sample_mean + margin_of_error
#     return (lower_bound, upper_bound)

# # 数据
# steps = ['20000', '5000', '1000']
# labels = ['random', 'rnd', 'ppo']
# values = np.array([
#     [0.50989, 0.46488, 0.29803],  # 对应 random
#     [0.50608, 0.48484, 0.36684],  # 对应 rnd
#     [0.47919, 0.49968, 0.35955]   # 对应 ppo
# ])
# std_devs = np.array([
#     [0.20889, 0.21956, 0.17074],  # 对应 random
#     [0.21225, 0.2113, 0.1854],    # 对应 rnd
#     [0.18866, 0.219, 0.1823]      # 对应 ppo
# ])

# # 样本数量和置信水平
# sample_size = 100000
# confidence_level = 0.95

# # 计算误差棒 (置信区间)
# errors = []
# for i in range(len(labels)):
#     error = []
#     for j in range(len(steps)):
#         lower_bound, upper_bound = calculate_confidence_interval(values[i][j], std_devs[i][j], sample_size, confidence_level)
#         margin_of_error = upper_bound - values[i][j]  # 只需计算上限偏差
#         error.append(margin_of_error)
#     errors.append(error)

# errors = np.array(errors)

# x = np.arange(len(steps)) 
# width = 0.2 

# fig, ax = plt.subplots()

# # 绘制柱状图
# for i, label in enumerate(labels):
#     ax.bar(x + i * width, values[i], width, yerr=errors[i], capsize=5, label=label)

# # 添加标签和标题
# ax.set_xlabel('Step')
# ax.set_ylabel('Score')
# ax.set_title('Mean reward with 95% confidence interval')
# ax.set_xticks(x + width)
# ax.set_xticklabels(steps)
# ax.legend()

# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.stats as stats

def calculate_confidence_interval(sample_mean, sample_std, sample_size, confidence_level):
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    margin_of_error = z_score * (sample_std / math.sqrt(sample_size))
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error
    return (lower_bound, upper_bound)

# 数据
steps = ['k = 0', 'k = 15 with initial goal', 'k = 16 without initial goal']
labels = ['random', 'rnd', 'ppo']
values = np.array([
    [0.19908, 0.26228, 0.50989],  # 对应 random
    [0.21874, 0.2675, 0.50608],   # 对应 rnd
    [0.26696, 0.35363, 0.47919]   # 对应 ppo
])
std_devs = np.array([
    [0.13112, 0.21527, 0.20889],  # 对应 random
    [0.15868, 0.22113, 0.21225],  # 对应 rnd
    [0.16611, 0.19165, 0.18866]   # 对应 ppo
])

# 样本数量和置信水平
sample_size = 100000
confidence_level = 0.95

# 计算误差棒 (置信区间)
errors = []
for i in range(len(labels)):
    error = []
    for j in range(len(steps)):
        lower_bound, upper_bound = calculate_confidence_interval(values[i][j], std_devs[i][j], sample_size, confidence_level)
        margin_of_error = upper_bound - values[i][j]  # 只需计算上限偏差
        error.append(margin_of_error)
    errors.append(error)

errors = np.array(errors)

x = np.arange(len(steps)) 
width = 0.2 

fig, ax = plt.subplots()

# 绘制柱状图
for i, label in enumerate(labels):
    ax.bar(x + i * width, values[i], width, yerr=errors[i], capsize=5, label=label)

# 添加标签和标题
ax.set_xlabel('Step')
ax.set_ylabel('Score')
ax.set_title('Mean reward with 95% confidence interval')
ax.set_xticks(x + width)
ax.set_xticklabels(steps)
ax.legend()

plt.show()
