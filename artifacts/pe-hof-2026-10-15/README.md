# PE/HOF proposal 材料，2026-10-15

这些材料对应中文 proposal 附录 A–D，保存在 proposal 仓库本地。

## 卷积代码

- 仓库：https://github.com/hsqStephenZhang/MimIR
- 基线：`09bfc92ae80bef5a2acdbcb449bbd909a84afdbf`
- 增量：`conv-winograd.patch`
- 补丁 SHA-256：`c7c584ce2ebd4f5cc441f64a9656fef42a044f9d65b417e7765fc963dd3fa12c`

补丁包括当前五个已跟踪源码文件的修改、新增 Winograd/常量 buffer 测试，以及测量所需的 build.py、run.py、summarize.py、bench.cpp。Im2col 已在基线中。基线提交目前在本地分支；`mimir-conv-base.bundle` 包含该提交，前置为 `c56d58681549d478afd893ed5b8915d2bfbda979`（远端 feature/par）。已从该前置和 bundle 在新仓库恢复精确的基线 hash。补丁在独立的基线 archive 上通过 `git apply --check`，应用后的每个文件与来源逐字一致。原 worktree、index、分支与提交均未修改。

```sh
git clone https://github.com/hsqStephenZhang/MimIR.git mimir-conv
cd mimir-conv
# BASE_BUNDLE 指向 mimir-conv-base.bundle
git bundle verify "$BASE_BUNDLE"
git fetch "$BASE_BUNDLE" HEAD
git checkout --detach 09bfc92ae80bef5a2acdbcb449bbd909a84afdbf
git apply --check "$PATCH"
git apply "$PATCH"
git submodule update --init --recursive
```

`BASE_BUNDLE` 应设为本材料 Git bundle 的位置，`PATCH` 应设为本材料 `conv-winograd.patch` 的绝对位置。随后按基线构建说明构建 Release 编译器到 `build/`。

原测量脚本 `docs/benchmarks/conv-winograd/build.py` 固定了当时的本地 clang++ 路径。复现者应将第二条编译命令的首元素改为 LLVM 22.1.4 的 clang++ 路径，其余 `-O3 -mcpu=native -std=c++23` 等选项保持一致；这是环境路径适配。不要把材料中的二进制 hash 当作跨机器重建的相等要求。

```sh
python3 docs/benchmarks/conv-winograd/build.py 1 1
python3 docs/benchmarks/conv-winograd/build.py 1 4
python3 docs/benchmarks/conv-winograd/build.py 4 1
python3 docs/benchmarks/conv-winograd/build.py 4 4
python3 docs/benchmarks/conv-winograd/run.py
python3 docs/benchmarks/conv-winograd/summarize.py
```

本轮验证了 bundle 恢复、补丁应用与文件一致性，未重建或重新 benchmark。历史验证条件与测试结果见 `conv/README.md`。`conv/results/` 是原始独立运行的 CSV，每个数值已是一次运行内七轮的中位数；全部单轮样本不可恢复。`conv/table.md` 已独立按每组三次运行重算核对。

## 前端结果与依赖缺口

仓库：https://github.com/hsqStephenZhang/mimir-frontend

- 作者确认的运行源码版本：`619a15776d91a645fba997c059a3001e8e778dea`
- 检查时 MimIR：`3a31afcdef1e00db51ea0de175de4c4b2228443e`
- 检查时 Torch 插件：https://github.com/hsqStephenZhang/mimir-torch-plugin ，`6c41f8eb2313cbb923c2cb92a0a5a6b4d4f9a23d`
- 现有 `frontend/level1.json` 和 `level2.json`：100 PASS；99 PASS/1 FAIL/0 INVALID。日志逐字归档。

结果 JSON 没有嵌入完整命令、环境变量或依赖 lock；`frontend/manifest.json` 的 `run_arguments` 与 `run_dependency_lock` 由运行者事后补记，事后检查的 build hash 仍不能代替运行时记录。前端请求 `torch.scan.diff`/`diff_i64`，所列插件源版本没有该接口；作者报告六个前端测试因此失败。这些版本是当前检查对象，尚未构成完整兼容的前端测试复现配方。卷积补丁与这个前端子模块是独立实验材料。

## 文件校验

`manifest.json` 列出所有归档文件的 SHA-256 和补丁文件清单；校验时对相对路径读取字节后计算 SHA-256。运行环境补齐后需保留本组原始结果，并为新的环境和复跑结果新建记录。
