# COMP90086 报告

正式改稿入口是 `main.tex` 和 `sections/*.tex`（共 5 个章节文件）；`reportdraft.en.md` 仅保留原稿，不会与 TeX 自动同步。当前报告按 IEEE A4 双栏排版，正文 10pt、标准页边距，共 5 章、6 张表（表 IV 分 a/b）、2 张图和 4 条引用；无摘要和关键词。作者为 Hongze Li（1820462）和 Jiatong Wu（1822875）。当前 `build/main.pdf` 共 4 页，包含参考文献。

## 保存后自动更新

本机已有 MacTeX / TeX Live 的 `latexmk`、`pdflatex` 和 VS Code 的 **LaTeX Workshop**，无需付费软件或额外安装。

1. 在 VS Code 中打开整个 `final_project` 或单独打开 `report` 文件夹，两处均已配置 `.vscode/settings.json`。
2. 打开 `report/main.tex`（单独打开 report 时就是 `main.tex`）。首次使用可运行 **LaTeX Workshop: Build LaTeX project**。
3. 运行 **LaTeX Workshop: View LaTeX PDF file**，使用编辑器内的 PDF 标签页。
4. 保存 `main.tex`、`sections/*.tex`、已引用的 `.bib` 或图片后，依赖文件变化会触发编译；编译完成后 PDF 预览自动刷新。不是每次按键都编译。

也可以在 Codex 内置 LaTeX 编辑器中打开 `report` 项目；保存源文件后会自动编译并更新 PDF 预览。当前桌面编译已成功，并已实际验证修改、保存章节 TeX 后自动重编译；当前会话中的监听进程保持运行。

配置使用 `onFileChange`，因此也支持其他编辑器保存到磁盘后的变化。所有章节都指定了根文件 `../main.tex`。PDF 和辅助文件集中在 `build/`，文件路径含空格也可编译。

如果使用其他编辑器，或没有打开 VS Code，可在项目根目录启动：

```bash
bash report/scripts/watch.sh
```

保持该终端运行，即会监听保存并重新编译；按 `Ctrl+C` 停止。单独打开 report 时使用 `bash scripts/watch.sh`。这不会安装常驻服务，也不需要同时开启多个监听器。外部 PDF 阅读器是否自动刷新取决于阅读器；推荐使用编辑器内置预览。

仅编译一次：

```bash
bash report/scripts/build.sh
```

## 文件组织

- `main.tex`：标题、作者、章节入口、参考文献入口。
- `sections/`：引言、方法、实验、讨论、结论，共 5 章。
- `references.bib`：BibTeX 文献；正文当前引用 4 条。
- `images/cases/`：4 张案例原图；TeX 横向排版，保留完整画幅和原始比例。
- `images/`：报告图像资源；图 1 使用此目录中的矢量 PDF。
- `build/main.pdf`：导出的报告；其余 build 文件均为编译缓存。
- `IEEEtran.cls`、`IEEEtran.bst`：IEEE 排版与文献样式。
- `vendor/`：原始模板、发行包和下载来源。
- `scripts/`：编译一次或连续监听的入口。

## 作业限制

- 正文最多 **4 页 A4**；只有参考文献可以延伸至第 5 页。模板不会通过缩小字号或页边距规避限制，也不会自动保证你添加后的页数。
- 写清方法动机、对照实验、实验设置、结果与误差分析；最终报告读者无需看代码即可理解方法。
- 最终提交截止 **2026-10-25 23:59**，以英文作业和后续 LMS 更新为准。

编译成功时 `build/main.pdf` 更新；如果某次保存有 LaTeX 错误，编辑器会显示错误，旧 PDF 不代表最新源文件。查看 LaTeX Workshop 或 Codex 编辑器的日志修正后再次保存。

模板下载详情见 `vendor/SOURCE.md`。研究方案与数据审计放在项目根目录的 `planning/`，不占正式报告篇幅。
