# GeoMAR 论文主页

这是一个无需构建的英文静态网页。所有图片、样式、脚本都在本目录中，不依赖外部字体、CDN 或后端服务。

页面参考 SONIC（https://ubc-vision.github.io/sonic/）的论文主页排版：白色背景、居中标题与作者、深色圆角资源按钮、宽幅结果展示，以及 Abstract / Overview / Qualitative Comparisons / Citation 四个章节。保留论文图、拖动对比和数据集切换。

## 本地查看

直接双击 `index.html` 即可查看；拖动对比、数据集切换均可本地使用。若浏览器限制剪贴板，复制引用按钮会自动尝试兼容方式，也支持手动复制。

也可以在本目录运行 `python -m http.server 8000`，然后访问 `http://localhost:8000`。

## 上传到现有论文代码仓库（推荐）

1. 将整个 `docs` 文件夹上传至仓库根目录，保持文件结构。
2. 在 GitHub 仓库中进入 **Settings → Pages**。
3. Source 选择 **Deploy from a branch**。
4. Branch 选择你的默认分支（通常是 `main`），目录选择 **/docs**，点击 **Save**。
5. 部署完成后，在 Pages 设置页查看最终网址。后续向这个目录推送修改即可更新。

官方说明：https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site

若使用单独的网站仓库，也可以把 `docs` 文件夹内部的全部文件上传到仓库根目录，并把 Pages 目录设为 **/(root)**。保留 `.nojekyll` 文件。

## 修改内容

- `index.html`：论文标题、作者、单位、按钮链接、摘要、方法文案、指标和 BibTeX。
- `style.css`：颜色、字体、页面布局和移动端样式。
- `script.js`：对比滑块、数据集切换和引用复制。
- `assets/`：论文实际示例、方法图和补充材料 PDF。
- `assets/fonts/`：本地 Noto Sans 字体及许可信息，离线也可保持排版。

当前链接来自本地论文和 README：arXiv `2608.03923`、代码仓库 `BRL-SYSU/GeoMAR`、DOI `10.1145/3767308.3836198`。上传前可按需调整。

## 图片与内容来源

- 方法图来自原有 `assets/geomar_architecture.png`。
- 结果图提取自 `latex/GeoMAR_MM2026_cameraready/Figures/fig3.pdf` 中的原始嵌入图像，包含 LFW、WebPhoto、WIDER 和 CelebA severe 示例。PDF 导出中包含了多张论文图的素材，已按论文中的实际排列核对。
- 示例图保留论文原有的红色区域标注和局部放大图。它们是预计算的论文结果，页面不进行在线模型推理。
- 定量表格展示主论文的 10 个方法：三个真实数据集的 FID / NIQE / MANIQA，以及 CelebA-Test 两种退化设置的 FID / NIQE / MANIQA / PSNR / SSIM / LMD。数值直接从 `latex/GeoMAR_arXiv (2)` 的 LaTeX 表格读取；网页不展示运行时间。
- 作者、单位、摘要、实验数值和引用来自本地论文 LaTeX 与仓库 README；补充材料由更新后的 `latex/GeoMAR_arXiv (2)/sigconf.tex` 和 `supp.tex` 编译，提取完整稿件第 11–20 页的附录，共 10 页，完整保留原文、图表和版式。

页面采用相对路径，适配 GitHub Pages 的仓库子路径，也能直接通过本地文件打开。

如需更新补充材料，直接替换 `assets/supplementary.pdf`，网页链接无需修改。

更新论文中的定量表格后，可在仓库根目录运行 `python scripts/build_project_tables.py` 同步网页数值。上传网页只需要 `docs` 文件夹，无需上传或运行此生成脚本。
