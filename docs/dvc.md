# DVC 数据版本控制使用指南

> 本文档说明本项目的 DVC (Data Version Control) 用法：如何追踪数据、推送/拉取数据、切换数据集版本。

## 1. 概述

DVC 用于对 `data/` 目录中的数据集（原始视频、关键点、标注、SQLite 数据库等）进行版本控制，
避免将大文件直接提交到 Git，从而让 Git 仓库保持轻量、数据集可回溯。

- **被追踪内容**：`data/` 目录（通过 `data.dvc` 追踪）
- **远程存储（Local Remote）**：`D:\WorkSpace\coding\fall-risk-prediction\dvc-storage`
  （默认远程，名为 `storage`；已在 `.gitignore` 中忽略）
- **安装方式**：`dvc` 已加入 `pyproject.toml` 的 `[project.optional-dependencies] dev` 列表
  （`dvc>=3.0.0`），并在项目虚拟环境 `venv` 中安装，通过 `venv\Scripts\dvc.exe` 调用。

> 注意：`data.dvc` 与 `.dvc/` 目录由 DVC 创建，需提交到 Git；而 `data/` 实际文件
> 与 `dvc-storage/` 均已在 `.gitignore` 中忽略，不会进入 Git。

## 2. 常用命令速查

| 场景 | 命令 | 说明 |
|------|------|------|
| 追踪新数据/数据变更 | `dvc add data` | 生成/更新 `data.dvc` |
| 查看状态 | `dvc status` | 显示数据与缓存是否同步 |
| 推送数据到远程 | `dvc push` | 将本地缓存推送到 `storage` 远程 |
| 拉取数据 | `dvc pull` | 从远程拉取缺失的数据到本地 |
| 切换数据版本 | `dvc checkout` | 恢复到 `data.dvc` 记录的版本 |
| 查看远程 | `dvc remote list` / `dvc remote default` | 列出/查看默认远程 |
| 数据浏览 | `dvc list` | 列出 DVC 追踪的文件 |

在 Windows PowerShell 下，请使用虚拟环境中的 DVC：

```powershell
& "venv\Scripts\dvc.exe" status
# 或先激活虚拟环境后直接使用 dvc
venv\Scripts\activate
dvc status
```

## 3. 添加 / 更新数据

每当 `data/` 下新增或修改数据文件后：

```powershell
dvc add data
```

- 若首次追踪，会生成 `data.dvc`（记录数据内容的 MD5 校验和与文件清单）；
- 之后数据变更时再次执行，`data.dvc` 会更新为新的校验和；
- 最后提交 Git：`git add data.dvc && git commit -m "data: update dataset"`。

> 提示：`dvc add` 会自动把被追踪路径写入 `.gitignore`（本项目已添加 `/data`），
> 数据本体不会进入 Git 仓库。

## 4. 推送 / 拉取数据

### 推送（本地 → 远程）

```powershell
dvc push
```

将本地缓存推送到默认远程 `storage`（即 `dvc-storage/` 目录）。

### 拉取（远程 → 本地）

```powershell
dvc pull
```

在**新克隆的仓库**上，先执行 `git pull` 拿到 `data.dvc`，再执行 `dvc pull`
即可还原 `data/` 目录（远程存储路径见 `.dvc/config`）。

## 5. 切换数据版本

配合 Git tag，可以精确回溯到某个历史数据集版本：

```powershell
# 1. 为当前数据集打标签（先提交 data.dvc）
git add data.dvc
git commit -m "data: update dataset"
git tag v1.0-dataset

# 2. 切换到历史版本
git checkout v1.0-dataset        # 切换 data.dvc 到历史版本
dvc checkout                     # 将 data/ 目录恢复到该版本对应的内容
# 若本地缓存无该版本数据，改用:
dvc pull                         # 从远程拉取对应版本
```

说明：
- `dvc checkout`：从**本地缓存**恢复（快速，前提是缓存中存在该版本）。
- `dvc pull`：从**远程**拉取（适合缓存缺失的场景）。
- 切换到最新开发状态：`git checkout master && dvc checkout`。

## 6. 远程存储位置

- **远程名称**：`storage`（默认远程）
- **本地路径**：`D:\WorkSpace\coding\fall-risk-prediction\dvc-storage`
- **配置位置**：`.dvc/config`

```ini
[core]
    remote = storage
['remote "storage"']
    url = D:\WorkSpace\coding\fall-risk-prediction\dvc-storage
```

`dvc-storage/` 已加入 `.gitignore`，不会被提交到 Git。如需迁移到共享存储
（NAS / 对象存储），可通过 `dvc remote modify storage url <新路径>` 更换。

## 7. 注意事项

- 所有 DVC 命令请在项目根目录执行。
- `data/` 下的文件变动（尤其是 `baseline.db` 等数据库）会导致 `dvc status`
  显示 `modified`，此时重新执行 `dvc add data && dvc push` 即可同步。
- 本项目当前 Python 环境为 `venv`（Python 3.14），DVC 3.67.1 已安装其中；
  `pyproject.toml` 已声明 `dvc>=3.0.0`，新成员执行 `make install-dev` 即可自动安装。
