# 贡献指南（Contributing）

感谢关注与贡献 `edara-external-eval`！本仓库维护跨宿主通用的科研提案认知治理中间件（`edara_mvp_eval`）与 **Benchmark-50** 多裁判盲评套件。

## 1. 本地开发与离线验证

所有核心审计、路由、跨宿主适配与论文制表组件均提供零网络、零 API 密钥的确定性单元测试：

```bash
git clone https://github.com/xixi501/edara-external-eval.git
cd edara-external-eval
pip install -e ".[dev]"
pytest -q
```

## 2. 核心设计红线

1. **宿主无关与通用性**：严禁将特定研究题目、特定数据集名称或单次案例硬编码进提示词或审计器中。所有治理规则必须基于通用认识论原则（证据归因、未核验实体隔离、同源混杂对照臂、统计功效匹配）实现。
2. **原生结构零破坏**：任何修订动作必须 100% 保留上游宿主提案的原始字段键集合与嵌套层级。
3. **评测防泄漏与对称预算**：
   - 盲评阶段严禁向裁判暴露臂别标识、审计诊断或编辑历史；
   - `Arm E`（治理中间件）与 `Arm D`（非结构化检索自反强基线）必须严格共享相同的检索次数上限、编辑轮数上限、单次输出 Token 上限及评测证据池（`shared_full_pool`）；
   - 严禁修改已归档于 `results/` 的历史评测快照。
4. **安全与隐私**：严禁提交任何明文 API 密钥、私有服务器绝对路径、原始运行缓存或未脱敏日志。模型端点配置仅允许在 `configs/*.json` 中通过 `credential_env` 声明环境变量名。

## 3. 提交 Pull Request 流程

1. 确保 `pytest -q` 全量通过；
2. 若修改了评测汇总或论文制表逻辑，运行 `python pilots/generate_paper_artifacts.py --check` 验证产物可重现；
3. 在 PR 描述中说明变更动机、受影响模块及离线/在线验证证据。
