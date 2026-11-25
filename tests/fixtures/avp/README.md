# 原AVP固定回归输入

`ItemDef.docx`逐字节取自提交`ad0e01d`的`input/ItemDef.docx`，用于原AVP来源、
功能清单和上下文回归。它与当前仓库`input/ItemDef.docx`内容一致。

生产`analyze`只读取显式传入的Item路径，不隐式读取此夹具。新增项目使用独立输入文件，
固定回归继续使用这里的DOCX，避免当前分析输入变化影响历史断言。

五份原始输入的统一路径、SHA256、离线读取检查及新分析/历史恢复的区别见
[输入与测试资产说明](../../../input/README.md)。
