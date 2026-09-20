# 科研 MCP 部署指南(arxiv / semanticscholar / ssh)

> 2026-09-18 新增。来源:办公机(21153)同日实装并冒烟验证通过的配置。**主机A/B 各自照本文件自助安装**;装完本机的 ZCode 会话即可直接搜文献/查引用/远程管理其他机器。
> 三个服务器的作用:①`arxiv`——检索/下载/解析 arXiv 论文(预印本核验);②`semanticscholar`——论文与引用网络检索(撞车预警排查);③`ssh`——持久 SSH 连接池,办公机/工作机互查 GPU、tail 日志、重启 runner,不用再走"push 台账→人肉去主机"循环。

## 一、前置(按需)

| 依赖 | 用途 | 安装 | 备注 |
|---|---|---|---|
| uv/uvx | arxiv + semanticscholar | PowerShell:`irm https://astral.sh/uv/install.ps1 \| iex` 或 `winget install astral-sh.uv` | 装在 `%USERPROFILE%\.local\bin`,体积小 |
| Node.js LTS | 仅 ssh 需要 | `winget install OpenJS.NodeJS.LTS --silent` | ~90MB 装 C 盘 |
| ZCode 客户端 | 全部 | 已有 | 装完 MCP 后**必须重启客户端**才加载 |

**主机B 特别提醒(C 盘 97% 满)**:uv 系两个服务器很轻可直接装;uvx 缓存默认在 `%LOCALAPPDATA%\uv`,若担心 C 盘可设环境变量 `UV_CACHE_DIR=E:\uv-cache` 再用;Node 若 C 盘吃紧可暂缓,先装 arxiv+semanticscholar 两个。

## 二、写入配置

编辑(没有就新建)`C:\Users\<用户名>\.zcode\cli\config.json`,加入 `mcp.servers` 键:

**通用版(PATH 正常时直接可用):**

```json
{
  "mcp": {
    "servers": {
      "arxiv": {
        "command": "uvx",
        "args": ["arxiv-mcp-server"]
      },
      "semanticscholar": {
        "command": "uvx",
        "args": ["--with", "mcp<2", "--with", "semanticscholar", "semanticscholar-mcp-server"]
      },
      "ssh": {
        "command": "cmd",
        "args": ["/c", "npx", "-y", "ssh-mcp-sessions"]
      }
    }
  }
}
```

**绝对路径版(通用版连不上时用这个,办公机实测最稳):**

```json
{
  "mcp": {
    "servers": {
      "arxiv": {
        "command": "C:\\Users\\<用户名>\\.local\\bin\\uvx.exe",
        "args": ["arxiv-mcp-server"]
      },
      "semanticscholar": {
        "command": "C:\\Users\\<用户名>\\.local\\bin\\uvx.exe",
        "args": ["--with", "mcp<2", "--with", "semanticscholar", "semanticscholar-mcp-server"]
      },
      "ssh": {
        "command": "C:\\Program Files\\nodejs\\node.exe",
        "args": ["C:\\Users\\<用户名>\\AppData\\Roaming\\npm\\node_modules\\ssh-mcp-sessions\\build\\index.js"]
      }
    }
  }
}
```

路径按各机实际情况替换:uvx 用 `where uvx` 查;ssh 服务器需先 `npm install -g ssh-mcp-sessions`(全局装完后入口在 `%APPDATA%\npm\node_modules\ssh-mcp-sessions\build\index.js`)。主机A 用户名为 `admin`,主机B 为 `508`。

> 注:ZCode 用嵌套键 `mcp.servers`;若某机也用 Claude Desktop/Cursor 等客户端,对应配置是顶层 `mcpServers` 键,JSON 内容一致换个壳。

## 三、装前自测(可选但推荐,3 条命令验明服务器能启动)

```bash
uvx arxiv-mcp-server            # 下载依赖后静默等待输入 = 正常,Ctrl+C 退出
uvx --with "mcp<2" --with semanticscholar semanticscholar-mcp-server
                                # 看到 "Starting Semantic Scholar MCP server" = OK
node "%APPDATA%\npm\node_modules\ssh-mcp-sessions\build\index.js"
                                # 看到 "SSH MCP Server running on stdio" = OK
```

## 四、实装踩过的坑(勿重复)

1. **semanticscholar 的 PyPI 包不声明 `mcp` 依赖,且与 mcp 2.x 不兼容**(2.x 把 FastMCP 改名 MCPServer,直接装必报 `ModuleNotFoundError: mcp.server.fastmcp`)——`--with "mcp<2"` 是必需的,别省;
2. **npm 有同名 `arxiv-mcp-server` 包但是无关项目**,arxiv 服务器只能走 uvx/PyPI,别 npx;
3. **`.cmd` 垫片(npx/npm 全局命令)依赖 PATH**:客户端启动早于 Node 安装时会报 "'node' 不是内部或外部命令"——ssh 用 node.exe 绝对路径直调 `build/index.js` 可彻底绕开;
4. 装完配置**不重启客户端不生效**;验证:Settings → MCP 三项已连接;排障:`/diagnosing-mcp`。

## 五、ssh 服务器的使用前置

被管理的机器(主机A/B)需启用 Windows 自带 OpenSSH Server(PowerShell 管理员):

```powershell
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic
```

然后在装了 ssh MCP 的机器上,让会话用 `add-host` 工具登记目标机(host/user/password 或 keyPath),即可 `exec` 远程命令、查 `nvidia-smi`、tail 训练日志。主机B 注意:ToDesk/向日葵常驻不影响 ssh;GitHub 连通波动也不影响局域网 ssh。

## 六、semanticscholar 说明

无 API key 可用(匿名限速约 1 次/秒,常规调研够);后续量大可到 semanticscholar.org 免费申请 key。
