# 私有仓库与服务器迁移

## 本快照包含什么

本仓库是当前工作目录的源码快照，不包含原项目的 Git 历史。
保留源码、测试、配置、文档、许可证和版本化图片。
`third_party/LIBERO` 以普通文件保存，包含原目录的本地修改及其许可证；
导出时参考的上游提交为 `8f1084e3132a39270c3a13ebe37270a43ece2a01`，
但不代表这些本地文件与该提交完全一致。

不包含 `.venv-*`、缓存、训练输出、下载的模型、数据或本地 RoboTwin 安装。
这些内容必须在新服务器重新安装、下载或单独传输。
不要将访问令牌、私钥、环境文件或受限数据提交到仓库，即使仓库是私有的。

原目录的 Cosmos 子模块为空，且没有可恢复的原始提交号。
本快照不保留无效的子模块声明；需要 Cosmos 时按 README 克隆上游源码。
该下载不锁定版本，不能保证重现原环境。验证兼容性后，应记录实际使用的提交号。

## 首次上传

1. 在自己的 GitHub 账户下创建空的 **Private** 仓库，建议命名为
   `OpenWAM-private`。不要初始化 README、许可证或忽略文件。
2. 在终端配置 GitHub SSH 认证；不要将凭据写入仓库或发到聊天中。
3. 在本导出目录检查 `git status` 和提交清单。若尚未提交：

   ```bash
   git add .
   git diff --cached --stat
   git commit -m "Import current OpenWAM source snapshot"
   ```

4. 将以下占位地址替换为自己的私有仓库地址，然后执行：

   ```bash
   git remote add origin git@github.com:YOUR_ACCOUNT/OpenWAM-private.git
   git ls-remote origin
   git push -u origin main
   ```

   若远程已包含提交，先停止并检查，不使用强制推送。
   上传前和上传后均在 GitHub 页面确认仓库为 Private。

## 在其他服务器恢复

新服务器需要 Git、Python >= 3.10，以及适合目标 GPU 的 NVIDIA 驱动。
安装依赖时还可能需要编译器和 CUDA Toolkit，详见 README。
先为该服务器配置对此私有仓库的访问权限，再执行：

```bash
git clone git@github.com:YOUR_ACCOUNT/OpenWAM-private.git OpenWAM
cd OpenWAM
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
```

上述 PyTorch/CUDA 组合来自现有 README，不保证适合所有服务器。
也可以按 `assets/openwam_usage_docs/docker.md` 使用 Docker 路线。
模型与基准数据通过 `scripts/download_assets/` 中的下载脚本准备；
RoboTwin 等模拟器按相应 benchmark 文档安装。
检查配置和启动脚本中的绝对路径、GPU 编号、网络端口及资产目录后再运行。
现有依赖声明并非完整环境锁定，重建环境不保证与当前机器逐版本一致。

## 验收

- 在独立目录重新克隆远程仓库，核对 `git rev-parse HEAD` 与导出仓库一致。
- 确认源码和许可证存在，虚拟环境、密钥、模型和训练输出未被追踪。
- 需要 Cosmos 时验证上游源码下载及安装；不需要时可跳过。
- 在目标 GPU 服务器验证依赖导入及选定的推理或评测任务。
- Git 推送成功仅代表代码传输完成，不代表环境、模型和 GPU 运行已验证。

## 后续同步

本导出目录与原工作目录相互独立，不会自动同步后续修改。
首次上传完成后，建议以新的 Git 仓库作为代码修改入口，并通过
`git add`、`git commit`、`git push` 同步；其他服务器使用 `git pull`。
