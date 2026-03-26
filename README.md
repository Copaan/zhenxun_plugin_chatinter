# ChatInter

**ChatInter** 是一个基于 AI 意图识别的智能对话插件，为 [真寻Bot](https://github.com/zhenxun-org/zhenxun_bot) 提供强大的对话能力。

当用户消息未被其他插件匹配时，ChatInter 使用大语言模型分析用户意图，实现：
- **功能调用意图** → 自动重路由到对应插件
- **普通聊天意图** → 进行自然对话回复

> [!WARNING]
>
> 由于上下文包含了插件的帮助信息，导致消耗的 tokens 会随着插件的数量增加而增加

## ✨ 特性

- 🤖 **AI 意图识别** - 使用 LLM 精准分析用户真实意图
- 🔀 **智能重路由** - 识别插件调用命令时自动转发到对应插件
- 💬 **自然对话** - 支持多轮对话，保持上下文连贯性
- 🖼️ **多模态支持** - 支持图片识别和理解
- 🧠 **聊天记忆** - 持久化存储对话历史，支持语境构建
- 🔧 **Agent 框架** - 支持 Tool Calling 和 MCP 工具集成
- 📚 **知识库 RAG** - 支持知识检索增强生成
- 🛡️ **安全沙箱** - 提供安全的表达式求值和 Shell 命令执行

## 📦 插件结构

```
chatinter/
├── __init__.py              # 插件入口，定义响应器和元数据
├── config.py                # 配置项定义和获取
├── memory.py                # 聊天记忆管理（ChatMemory 类）
├── agent_runner.py          # Agent 运行器（Tool Calling 循环）
├── agent_tools.py           # Agent 工具集（安全执行、插件查询）
├── chat_handler.py          # 聊天响应处理
├── handler.py               # 主处理器（handle_fallback）
├── plugin_registry.py       # 插件信息注册表
├── tool_registry.py         # 工具注册表
├── skill_registry.py        # 技能注册表
├── route_engine.py          # 路由引擎
├── knowledge_rag.py         # 知识库 RAG 检索
├── retrieval.py             # 向量检索
├── sandbox.py               # 沙箱执行环境
├── runtime.py               # 运行时调度
├── lifecycle.py             # 生命周期钩子
├── trace.py                 # 追踪模块
├── data_source.py           # 导出模块（整合各子模块）
├── models/
│   ├── __init__.py          # 模型导出
│   ├── chat_history.py      # 数据库模型
│   └── pydantic_models.py   # Pydantic 结构化模型
└── utils/
    ├── __init__.py          # 工具函数导出
    ├── cache.py             # 缓存工具
    ├── multimodal.py        # 多模态处理（图片提取）
    └── unimsg_utils.py      # UniMessage 工具函数
```

## 🔧 配置项

在机器人配置文件中添加以下配置：

| 配置键 | 说明 | 默认值 | 类型 |
|--------|------|--------|------|
| `ENABLE_FALLBACK` | 是否启用 ChatInter 兜底对话能力 | `True` | bool |
| `INTENT_MODEL` | ChatInter 使用的模型名称 (格式: ProviderName/ModelName)，留空时复用 AI.DEFAULT_MODEL_NAME | `""` | str |
| `INTENT_TIMEOUT` | ChatInter 推理超时时间（秒），<=0 时复用 AI.CLIENT_SETTINGS.timeout | `20` | int |
| `CONFIDENCE_THRESHOLD` | 插件意图置信度阈值，低于该值时降级为普通聊天 | `0.72` | float |
| `CHAT_STYLE` | ChatInter 对话风格补充设定，留空使用默认风格 | `""` | str |
| `CUSTOM_PROMPT` | ChatInter 自定义系统提示词补充，会追加到系统提示词末尾 | `""` | str |
| `MCP_ENDPOINTS` | MCP 工具服务地址列表，使用英文逗号分隔 | `""` | str |
| `REASONING_EFFORT` | 强制推理强度，可选 MEDIUM 或 HIGH，留空表示不强制设置 | `"MEDIUM"` | str |

### 配置示例

```yaml
# configs.yml
chatinter:
  ENABLE_FALLBACK: true
  INTENT_MODEL: "Gemini/gemini-2.0-flash"
  INTENT_TIMEOUT: 20
  CONFIDENCE_THRESHOLD: 0.72
  CHAT_STYLE: "活泼可爱"
  CUSTOM_PROMPT: ""
  MCP_ENDPOINTS: "http://127.0.0.1:9001,http://127.0.0.1:9002"
  REASONING_EFFORT: "MEDIUM"
```

## 📖 使用方法

### 自动加载

插件会自动加载，无需手动操作。当消息满足以下条件时会被处理：

1. 消息 `@` 了机器人
2. 消息未被其他高优先级插件处理

### 超级用户命令

```
重置会话    # 重置当前会话历史（仅超级用户可用）
```

## 🏗️ 工作流程

```
用户消息 → 意图分析 → 判断意图类型
                     ├── 插件调用 → 重路由到对应插件 → 执行命令 → 保存记忆 → 返回结果
                     └── 普通聊天 → 构建上下文 → Agent 对话 → Tool Calling → 保存记忆 → 返回回复
```

### Agent 能力

ChatInter 1.1.0 引入了 Agent 框架，支持：

- **Tool Calling** - LLM 可调用工具获取实时信息
- **插件查询** - 动态检索可用插件和命令
- **安全执行** - 在沙箱中执行数学表达式和只读 Shell 命令
- **MCP 集成** - 支持外部 MCP 工具服务

### 多模态支持

- 支持识别消息中的图片
- 支持识别回复链中的图片
- 图片会作为多模态输入传递给 LLM

### 待补全机制

当用户发送的消息缺少必要信息时（如需要图片但未提供），ChatInter 会提示用户补充并等待后续消息。

## 🗄️ 数据库

ChatInter 使用 `ChatInterChatHistory` 模型持久化存储对话历史：

```python
class ChatInterChatHistory(Model):
    id: int = Field(pk=True, auto_increment=True)
    user_id: str
    group_id: str | None
    nickname: str
    user_message: str
    ai_response: str
    timestamp: datetime
    bot_id: str | None
    session_reset: bool = False
```

## 🎨 效果图

![example](docs_image/1.png)

## 🚀 更新日志

### v1.1.0

- 新增 Agent 框架，支持 Tool Calling
- 新增 MCP 工具服务集成
- 新增知识库 RAG 检索
- 新增安全沙箱执行环境
- 新增待补全机制（图片、@目标）
- 优化路由引擎，提升意图识别准确率
- 配置项重构，简化配置流程

### v1.0.0

- 初始版本
- 基础意图识别和重路由
- 多轮对话支持
- 多模态支持

## 📄 许可证

本项目采用 [AGPL-3.0](./LICENSE) 许可证。

## 🙏 致谢

- [绪山真寻 Bot](https://github.com/zhenxun-org/zhenxun_bot)
- [BYM AI 插件](https://github.com/zhenxun-org/zhenxun_bot_plugins/tree/main/plugins/bym_ai)
- Copaan - Agent 框架贡献
- 万能的 AI sama

## 📧 联系方式

如有问题或建议，请提交 Issue。