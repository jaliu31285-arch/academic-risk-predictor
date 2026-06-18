# -*-coding:utf-8-*-
# 大创最终版：大学生学业风险与成绩预测模型
# 解决初代数据泄露、样本失衡问题，7000条均衡学生数据集
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
import time

# 解决matplotlib中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===================== 1. 加载数据集 =====================
file_path = 'college_students_academic_performance_dataset.csv'
try:
    df = pd.read_csv(file_path)
    print(f"数据加载成功! 样本总数:{df.shape[0]}, 原始特征数:{df.shape[1]}")
except FileNotFoundError:
    print("报错：未找到CSV数据集，请检查文件路径与文件名！")
    raise

# ===================== 2. 数据预处理 =====================
print("\n========== 数据预处理阶段 ==========")
# 缺失值填充
if df.isnull().sum().sum() > 0:
    print("检测到缺失值，数值列用中位数填充，类别列用众数填充")
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
else:
    print("数据集无缺失值，数据完整")

# 类别特征编码（性别）
label_cols = ['Gender']
encoder_dict = {}
for col in label_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoder_dict[col] = le
        print(f"完成类别编码：{col}")

# ===================== 3. 剔除泄露特征，划分特征与标签 =====================
# 定义预测目标
target_class = 'Academic_Risk_Class'    # 分类目标：学业风险等级
target_reg = 'Final_Academic_Score'     # 回归目标：最终学业分数

# 剔除ID、目标、衍生泄露字段（核心优化点）
exclude_cols = [
    'Student_ID',
    target_class,
    target_reg,
    'Academic_Risk_Level',
    'Intervention_Recommendation',
    'Performance_Score'
]

feature_cols = [col for col in df.columns if col not in exclude_cols]
X = df[feature_cols]
y_class = df[target_class]
y_reg = df[target_reg]

print(f"\n有效特征数量：{X.shape[1]}（剔除{len(exclude_cols)}个泄露字段）")
print("前10个特征：", feature_cols[:10], "……")
print("\n学业风险标签分布：")
print(y_class.value_counts().sort_index())

# 分层划分训练集、测试集 8:2
test_size = 0.2
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=test_size, random_state=42, stratify=y_class)
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=test_size, random_state=42)

# ===================== 4. 随机森林分类：学业风险预警 =====================
print("\n========== 训练学业风险分类模型 ==========")
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train_c, y_train_c)
y_pred_class = clf.predict(X_test_c)

acc = accuracy_score(y_test_c, y_pred_class)
print(f"分类模型整体准确率：{acc:.2%}")

print("\n分类评估报告：")
print(classification_report(y_test_c, y_pred_class, zero_division=0))

# 绘制混淆矩阵并保存
plt.figure(figsize=(7, 6))
cm = confusion_matrix(y_test_c, y_pred_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("模型预测风险等级")
plt.ylabel("真实风险等级")
plt.title("学业风险分类混淆矩阵")
plt.tight_layout()
plt.savefig("risk_confusion_matrix.png", dpi=150)
plt.show()

# ===================== 5. 随机森林回归：最终成绩预测 =====================
print("\n========== 训练最终学业分数回归模型 ==========")
print("模型训练中，请稍候……")
reg = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42, n_jobs=-1, verbose=1)

start_time = time.time()
reg.fit(X_train_r, y_train_r)
end_time = time.time()
print(f"训练完成，耗时{end_time - start_time:.2f}秒")

y_pred_reg = reg.predict(X_test_r)
rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_reg))
mae = np.mean(np.abs(y_test_r - y_pred_reg))
print(f"回归模型RMSE：{rmse:.4f}")
print(f"分数预测平均绝对误差MAE：{mae:.4f}")

# 绘制真实分数vs预测分数散点图
plt.figure(figsize=(8, 6))
plt.scatter(y_test_r, y_pred_reg, alpha=0.5, color='green')
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r-', lw=2)
plt.xlabel("真实最终学业分数")
plt.ylabel("模型预测分数")
plt.title(f"成绩预测拟合效果 (RMSE={rmse:.3f})")
plt.grid(True)
plt.savefig("score_predict_effect.png", dpi=150)
plt.show()

# ===================== 6. 特征重要性分析（论文核心结论） =====================
print("\n========== 影响学业的Top10关键特征 ==========")
feature_importance = clf.feature_importances_
# 取前10权重最高特征
top10_index = np.argsort(feature_importance)[::-1][:10]

print(f"{'排名':<5}{'特征名称':<30}{'重要性分数'}")
for rank, idx in enumerate(top10_index, start=1):
    print(f"{rank:<5}{feature_cols[idx]:<30}{feature_importance[idx]:.4f}")

# 绘制横向特征权重柱状图
plt.figure(figsize=(10, 6))
plt.barh(range(10), feature_importance[top10_index], align='center')
plt.yticks(range(10), [feature_cols[i] for i in top10_index])
plt.xlabel("特征重要性权重")
plt.title("预测学业风险Top10核心特征")
plt.tight_layout()
plt.savefig("top10_feature_importance.png", dpi=150)
plt.show()

# ===================== 7. 单样本预测演示（用于网页对接/现场演示） =====================
print("\n========== 单学生样本预测演示 ==========")
# 提取测试集第一条学生数据
single_sample = X_test_c.iloc[0].values.reshape(1, -1)
risk_result = clf.predict(single_sample)[0]
score_result = reg.predict(single_sample)[0]

print(f"预测学业风险等级：{risk_result}")
print(f"预测最终学业分数：{score_result:.2f}")