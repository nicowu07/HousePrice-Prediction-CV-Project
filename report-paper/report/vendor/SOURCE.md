# 模板来源

- 作业指定入口：https://www.ieee.org/conferences/publishing/templates.html
- IEEE Author Center：https://conferences.ieeeauthorcenter.ieee.org/write-your-paper/authoring-tools-and-templates/
- 实际下载：CTAN 的 IEEEtran 官方维护发行包，https://mirrors.ctan.org/macros/latex/contrib/IEEEtran.zip
- 包说明：https://ctan.org/pkg/ieeetran
- 版本：IEEEtran 1.8b，维护者 Michael Shell；许可证 LPPL 1.3。
- 下载日期：2026-09-22。
- ZIP SHA-256：`e0cd4f5afbd42c8076092280e72b3e0a5111efe501d35de9f715cfb8da313cb4`。

IEEE 原站的 `conference-latex-template_10-17-19.zip` 在本次访问中返回空文件，因此改用 CTAN 的 IEEEtran 发行包。它包含标准会议模板 `bare_conf.tex`，并非自行仿制 IEEE 排版。

`IEEEtran.zip` 和 `bare_conf.tex` 原样保存；项目目录的 `IEEEtran.cls` 与 `IEEEtran.bst` 也直接来自该压缩包。`main.tex` 是重新命名的项目骨架，采用 `conference,a4paper` 选项，不更改类文件、字号或页边距。原始说明和许可证信息保留在 `bare_conf.tex`、`IEEEtran-README.txt` 及压缩包中。
