# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 创建示例数据集
data = {
    'income': [45000, 32000, 65000, 72000, 55000, 38000, 48000, 80000],
    'credit_score': [680, 580, 720, 750, 700, 620, 690, 800],
    'debt_ratio': [0.3, 0.6, 0.2, 0.15, 0.4, 0.5, 0.35, 0.1],
    'approved': [1, 0, 1, 1, 1, 0, 1, 1]
}
df = pd.DataFrame(data)

# 准备特征和目标变量
X = df[['income', 'credit_score', 'debt_ratio']]
y = df['approved']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建并训练决策树
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# 可视化决策树
plt.figure(figsize=(20,10))
plot_tree(tree, 
          feature_names=['收入', '信用分数', '债务比率'],
          class_names=['拒绝', '通过'],
          filled=True,
          rounded=True,  # 添加圆角
          fontsize=10,   # 调整字体大小
          proportion=True,  # 显示样本比例
          precision=2)    #

plt.show()

# 进行预测
new_applicant = [[52000, 650, 0.35]]
prediction = tree.predict(new_applicant)
print("新申请人贷款审批预测:", "通过" if prediction[0] == 1 else "拒绝")