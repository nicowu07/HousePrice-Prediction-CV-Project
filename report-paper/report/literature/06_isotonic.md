# Isotonic regression 校准

- 来源：scikit-learn 官方 `IsotonicRegression` 文档。
- 类型：技术／API 文档，不是研究论文，也没有论文摘要。
- [官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html)
- 查阅日期：2026-10-24。
- 当前访问的文档版本：1.9.1，与组员仓库 `requirements.lock.txt` 中记录的 scikit-learn 版本一致。
- 建议引用键：`sklearn_isotonic`；作为网络资料引用时注明查阅日期。

## 技术说明摘要

Isotonic regression 拟合受单调性约束的一维映射。官方实现支持递增／递减约束，以及对输入超出拟合范围时的处理；`out_of_bounds='clip'` 将这种输入映射到最近端点对应的输出。它可以用于学习预测分数与真实价格之间的单调校正关系。

## 本报告可用的内容

- 在方法中说明如何把反变换后的回归预测进一步校准到价格尺度。
- 文档提供方法实现入口；校准器究竟用什么数据拟合，应根据我们实际代码说明。
- 当前单模型评估使用外层训练部分的内部 OOF 预测拟合 isotonic 映射，然后作用于外层验证预测；对应代码在 `src/exp_heads.py` 的 `evaluate_head`。
- 最终九模型融合还使用了线性再校准，不能把该融合的全部后处理统称为 isotonic。

## 写作时要区分

1. `out_of_bounds='clip'` 处理的是校准器输入超出拟合域；本项目把价格裁剪到 [195, 2000] 是另一个操作。
2. 校准图接近对角线不等于模型已达到条件均值或 Bayes 最优，也不意味着个别高价房没有大误差。
3. 校准带来的任务内变化要用本项目的受控结果支持，官方文档不保证它总能降低验证 MSE。

可采用的表达方向：我们用训练部分的内部 OOF 预测拟合单调校准映射，并将其应用于留出的验证预测。
