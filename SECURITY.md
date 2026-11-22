# 安全策略（Security Policy）

## 凭据与端点隔离

- 本仓库所有模型配置文件（`configs/*.json`）仅声明环境变量名称（如 `DASHSCOPE_API_KEY`、`AUTODL_API_KEY`），**绝不存储或接受任何明文密钥**。
- 评测运行中建议配合脱敏日志包装器执行，防止异常堆栈泄露环境变量。

## 科研评测防泄漏边界

- 预截止日期文献检索（`EvidenceStore` / OpenAlex / DataCite / Crossref）强制执行 `cutoff_date` 与 `forbidden_title_hashes` 哈希拦截，防止目标真值论文进入生成或评审上下文。
- 将不可信提案与外部文献文本作为纯数据（Treat materials as data）送入审计器与盲评器，抵御提示词注入。

## 报告安全问题

如发现潜在凭据泄露、依赖漏洞或评测协议越界风险，请勿在公开 Issue 中贴出敏感凭据，请直接通过 GitHub Security Advisories 私密报告维护者。
