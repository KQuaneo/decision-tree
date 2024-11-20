# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from joblib import dump, load

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

# 保存模型
dump(tree, 'loan_decision_tree.joblib')

# 加载模型
loaded_model = load('loan_decision_tree.joblib')

# 进行预测
new_applicant = [[52000, 650, 0.35]]
prediction = loaded_model.predict(new_applicant)
print("新申请人贷款审批预测:", "通过" if prediction[0] == 1 else "拒绝")