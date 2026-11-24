"""Build the current five-input engineering review from maintained source chapters.

Document tooling only. Optional dependencies: beautifulsoup4 and Markdown.
No imports of the production runtime, Provider calls, or analysis-data writes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import textwrap
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

import markdown
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CODE = "d6a400e7010d62ff5dda56efd8215f23729bb3ec"
DOCS = "master"
GITHUB = "https://github.com/dmh045/Hara/"
APPENDICES = {
    "05_部署配置运行与恢复.md": ("appendix-c", "附录 A｜安装、配置、运行与恢复操作手册"),
    "06_开发测试与维护.md": ("appendix-d", "附录 B｜开发维护、测试地图与修改方法"),
    "08_资产移交与接手清单.md": ("appendix-e", "附录 C｜五项目资产、检查点与接手清单"),
}
DOC_ANCHORS = {
    "README.md": "section-0",
    "01_架构与端到端流程.md": "section-2",
    "02_输入抽取数据契约与方法治理.md": "section-6",
    "03_场景物理与风险计算.md": "wf-08",
    "04_SG安全状态FTTI与报告.md": "wf-12",
    "07_验证结果与已知限制.md": "section-11",
    **{name: value[0] for name, value in APPENDICES.items()},
}


def sections(text: str, prefix: str) -> dict[str, tuple[str, str]]:
    pattern = re.compile(r"^## (" + re.escape(prefix) + r"[\w-]+) (.+)$", re.M)
    matches = list(pattern.finditer(text))
    return {
        match[1]: (match[2], text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip())
        for i, match in enumerate(matches)
    }


def chapter(path: Path, number: int) -> str:
    matches = list(re.finditer(r"^## (\d+)\. (.+)$", path.read_text(), re.M))
    text = path.read_text()
    for i, match in enumerate(matches):
        if int(match[1]) == number:
            return text[match.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)].strip()
    raise ValueError(f"Missing chapter {number}: {path}")


def verified_examples(text: str, source: Path, section: str) -> list[dict]:
    """Check maintained excerpts against both the worktree and pinned source."""
    pattern = re.compile(
        r"<!-- source-example: ([^\n]+?):L(\d+)-L(\d+) -->\s*"
        r"```python\n(.*?)\n```", re.S
    )
    result = []
    for match in pattern.finditer(text):
        path, start, end = match[1], int(match[2]), int(match[3])
        headings = list(re.finditer(r"^#{2,4} 实现例 ([A-Z]\d+)｜([^\n]+)$", text[:match.start()], re.M))
        if not headings:
            raise ValueError(f"Code example has no indexed heading: {source.name}:{path}")
        heading = headings[-1]
        current = (ROOT / path).read_text(encoding="utf-8")
        pinned = subprocess.check_output(["git", "show", f"{CODE}:{path}"], cwd=ROOT, text=True)
        for version, contents in (("worktree", current), (CODE[:7], pinned)):
            lines = contents.splitlines()
            if not 1 <= start <= end <= len(lines):
                raise ValueError(f"Code range outside {version}: {path}:{start}-{end}")
            expected = textwrap.dedent("\n".join(lines[start - 1:end])).strip("\n")
            actual = textwrap.dedent(match[4]).strip("\n")
            if actual != expected:
                raise ValueError(f"Stale code excerpt in {source.name}: {path}:{start}-{end} ({version})")
        result.append({"id": heading[1], "title": heading[2], "section": section,
                       "source": path, "start": start, "end": end})
    if text.count("<!-- source-example:") != len(result):
        raise ValueError(f"Malformed source-example marker in {source.name}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa-output", type=Path)
    args = parser.parse_args()
    soup = BeautifulSoup('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title></title></head><body><header></header><div class="layout"><nav></nav><main></main></div></body></html>', "html.parser")
    main, nav = soup.main, soup.nav
    main["id"] = "document-content"
    nav["id"] = "document-toc"
    nav["aria-label"] = "文档章节目录"
    local_links: list[dict] = []
    examples: list[dict] = []
    source_files: set[Path] = {Path(__file__).resolve(), HERE / "report_style.css", HERE / "report_navigation.js"}

    def tag(name: str, text: str = "", **attrs):
        element = soup.new_tag(name, attrs=attrs)
        element.string = text
        return element

    def fragment(text: str, source: Path, prefix: str):
        source_files.add(source)
        examples.extend(verified_examples(text, source, prefix))
        if "<!-- code-example-index -->" in text:
            rows = ["| 实例 | 解决的问题 | 所在步骤 | 固定版本源码 |", "|---|---|---|---|"]
            for example in examples:
                ident, path = example["id"], example["source"]
                location = f"L{example['start']}-L{example['end']}"
                url = GITHUB + "blob/" + CODE + "/" + path + "#" + location
                rows.append(f"| [{ident}](#code-{ident.lower()}) | {example['title']} | [{example['section']}](#{example['section']}) | [{Path(path).name}:{location}]({url}) |")
            text = text.replace("<!-- code-example-index -->", "\n".join(rows))
        rendered = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
        node = soup.new_tag("div", attrs={"class": "current-detail", "data-source": source.name})
        parsed = BeautifulSoup(rendered, "html.parser")
        for child in list(parsed.contents):
            node.append(child.extract())
        shift = 1 if prefix.startswith(("wf-", "appendix-")) or prefix == "section-5" else 0
        for i, heading in enumerate(node.select("h2,h3,h4,h5")):
            heading.name = "h" + str(min(6, int(heading.name[1]) + shift))
            example = re.match(r"实现例 ([A-Z]\d+)｜", heading.get_text())
            heading["id"] = "code-" + example[1].lower() if example else f"{prefix}-part-{i + 1}"
        for anchor in node.select("a[href]"):
            href = anchor["href"]
            parts = urlsplit(href)
            if parts.scheme or href.startswith("#"):
                if href.startswith(GITHUB + "blob/" + CODE):
                    anchor["class"] = anchor.get("class", []) + ["source-reference"]
                continue
            target = (source.parent / unquote(parts.path)).resolve()
            exists = target.exists()
            local_links.append({"source": source.name, "href": href, "target": str(target), "exists": exists})
            if not exists:
                raise ValueError(f"Broken local link: {source.name} -> {href}")
            current_chapter = {
                "00_入口抽取与失效.md": "wf-00", "01_场景组合驾驶员与物理.md": "wf-05",
                "02_因果评分与下游报告.md": "wf-09", "03_当前结果与交接说明.md": "section-8",
                "04_代码实现与问题定位索引.md": "appendix-g", "05_语义任务与执行职责.md": "section-5",
                "06_项目概念与分析示例.md": "start-purpose",
                "README.md": "appendix-f",
            }
            if target == args.output.resolve():
                anchor["href"] = "#section-0"
            elif target.parent == HERE and target.name in current_chapter:
                anchor["href"] = "#" + current_chapter[target.name]
            elif target.suffix == ".md" and target.name in DOC_ANCHORS and target.parent == HERE.parent:
                anchor["href"] = "#" + DOC_ANCHORS[target.name]
                anchor["title"] = f"同页主题入口；完整源章：{target.relative_to(ROOT)}"
            elif target.is_relative_to(ROOT / "output") or target.is_relative_to(ROOT / "runtime"):
                anchor["href"] = target.as_uri() + ("#" + parts.fragment if parts.fragment else "")
                anchor["class"] = anchor.get("class", []) + ["local-evidence"]
                anchor["title"] = "本机运行证据；异机需取得对应资产包"
            else:
                relative = target.relative_to(ROOT)
                version = DOCS if relative.parts[0] == "docs" else CODE
                route = "tree/" if target.is_dir() else "blob/"
                anchor["href"] = GITHUB + route + version + "/" + quote(str(relative), safe="/") + ("#" + parts.fragment if parts.fragment else "")
                anchor["title"] = f"{relative} @ {version[:7]}"
            if anchor["href"].startswith(GITHUB + "blob/" + CODE):
                anchor["class"] = anchor.get("class", []) + ["source-reference"]
        for pre in list(node.select("pre")):
            code = pre.select_one("code.language-python")
            heading = pre.find_previous(["h3", "h4", "h5", "h6"])
            if code is None or heading is None or not heading.get("id", "").startswith("code-"):
                continue
            disclosure = soup.new_tag("details", attrs={"class": "source-code"})
            summary = tag("summary", f"查看 Python 源码摘录 · {len(code.get_text().splitlines())} 行")
            disclosure.append(summary)
            pre.replace_with(disclosure)
            disclosure.append(pre)
        for table in list(node.select("table")):
            table.wrap(soup.new_tag("div", attrs={"class": "table-scroll"}))
        return node

    def section(ident, title, content, source):
        main.append(tag("h2", title, id=ident))
        main.append(fragment(content, source, ident))
        nav.append(tag("a", title, href="#" + ident))

    def source_chapters(ident, title, filename, numbers):
        source = HERE.parent / filename
        content = []
        for number in numbers:
            heading = re.search(rf"^## {number}\. (.+)$", source.read_text(), re.M)
            body = re.sub(r"^(#{3,5}) ", r"\1# ", chapter(source, number), flags=re.M)
            content.append(f"### {heading[1]}\n\n" + body)
        section(ident, title, "\n\n".join(content), source)

    diagram_source = HERE / "workflow_diagrams.html"
    diagrams = BeautifulSoup(diagram_source.read_text(), "html.parser")
    source_files.add(diagram_source)

    def diagram(ident):
        node = diagrams.find("figure", id=ident)
        if node is None:
            raise ValueError(f"Missing documentation diagram: {ident}")
        return node.extract()

    current_source = HERE / "03_当前结果与交接说明.md"
    current = sections(current_source.read_text(), "section-")
    expected_projects = ["AVP", "IPA-High", "IPA-Main", "RPA", "rTPA"]
    project_metrics = []
    for line in current["section-8"][1].splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) == 8 and cells[0] in expected_projects:
            values = [int(cell.replace(",", "")) for cell in cells[1:]]
            project_metrics.append(dict(zip(["project", "risk", "main", "sg", "c_known", "c_missing", "asil_known", "asil_missing"], [cells[0], *values])))
    if [x["project"] for x in project_metrics] != expected_projects:
        raise ValueError("Five-input result table is missing or has changed schema")
    for row in project_metrics:
        if row["risk"] != row["c_known"] + row["c_missing"] or row["risk"] != row["asil_known"] + row["asil_missing"]:
            raise ValueError(f"Inconsistent project totals: {row}")

    soup.title.string = "HARA 开发与技术交接｜项目原理、完整流程与实现详解"
    soup.header.append(tag("b", "HARA / ENGINEERING HANDOVER"))
    soup.header.append(tag("span", "五份输入最新修订结果 · 2026-11-24 · 代码 d6a400e"))
    soup.header.append(tag("button", "打印 / 保存 PDF", onclick="window.print()", type="button"))
    style = soup.new_tag("style")
    style.string = (HERE / "report_style.css").read_text()
    soup.head.append(style)
    main.append(tag("h1", "HARA 开发与技术交接"))
    main.append(tag("h2", "项目原理 · 完整流程、代码与五份输入最新结果", id="section-0"))
    nav.append(tag("b", "报告目录"))
    nav.append(tag("a", "文档用途与阅读入口", href="#section-0"))
    diagram_nav = soup.new_tag("details")
    diagram_nav.append(tag("summary", "流程图与结果图索引"))
    for ident, label in [("main", "1 端到端流程"), ("authority", "2 输入权威"),
                         ("result-state", "3 数值与审批"), ("identities", "4 场景与身份"),
                         ("physics", "5 物理任务分流"), ("time", "6 时间与响应"),
                         ("report", "7 报告与证据映射"), ("results", "8 五项目C数值"), ("recovery", "9 恢复与复用")]:
        diagram_nav.append(tag("a", label, href="#diagram-" + ident, **{"class": "subnav"}))
    nav.append(diagram_nav)
    cover = soup.new_tag("div", attrs={"class": "handover-cover"})
    cover.append(tag("p", "文档核对：2026-11-24 ｜ 业务结果：2026-11-23完整修订 ｜ 生产代码：d6a400e", **{"class": "edition"}))
    cover.append(tag("p", "本报告仅呈现AVP、IPA-High、IPA-Main、RPA、rTPA五份输入的最新可核查结果，以及当前13步生产流程、17个代码实例和开发交接说明。五份结果来自已完成的全量与后续定向修订，不是本次又从头运行五次全量。"))
    cover.append(tag("p", "有数值、计算完成、条件性结果与工程批准分别标明。报告中的本机证据链接需对应运行资产；正文与图表可以离线阅读。本次文档更新不调用Provider、不重新评分。"))
    cover.append(tag("p", "文档从项目用途、输入边界和核心概念展开，以端到端示例贯通分析流程，再逐步说明数据变化、执行职责、实现代码和验证依据。业务解释、计算条件与维护细节相互对应，便于工程评审和开发交接。"))
    links = soup.new_tag("div", attrs={"class": "quick-links"})
    for ident, title in [("start-purpose", "项目用途与边界"), ("diagram-main", "端到端主流程"), ("start-walkthrough", "贯穿分析示例"), ("section-8", "五份最新结果"), ("appendix-g", "代码与问题索引")]:
        links.append(tag("a", title, href="#" + ident))
    cover.append(links)
    main.append(cover)
    cards = soup.new_tag("div", attrs={"class": "metrics"})
    for value, label in [(5, "输入项目"), (sum(x["risk"] for x in project_metrics), "活动风险成员"), (sum(x["main"] for x in project_metrics), "HARA主表审核组"), (sum(x["sg"] for x in project_metrics), "已关联SG · 非工程批准数")]:
        card = soup.new_tag("div", attrs={"class": "metric"})
        card.append(tag("strong", f"{value:,}")); card.append(tag("span", label)); cards.append(card)
    # Metrics are introduced only after the reader has learned their meaning.

    concept_source = HERE / "06_项目概念与分析示例.md"
    concepts = sections(concept_source.read_text(), "start-")
    nav.append(tag("div", "项目概念与分析过程", **{"class": "nav-group"}))
    for ident, (title, text) in concepts.items():
        section(ident, title, text, concept_source)
        if ident == "start-purpose":
            main.append(diagram("diagram-main"))
        elif ident == "start-input":
            main.append(diagram("diagram-authority"))
        elif ident == "start-result":
            main.append(diagram("diagram-result-state"))
    main.append(cards)

    def current_section(ident):
        title, text = current[ident]
        section(ident, title, text, current_source)

    current_section("section-1")
    source_chapters("section-2", "程序架构、状态与职责边界", "01_架构与端到端流程.md", [1, 4, 5, 6, 7])

    main.append(tag("h2", "当前工作流：逐步处理、组合与结果流转", id="workflow"))
    main.append(tag("p", "00—12是业务讲解顺序，不是13个独立WorkflowGraph节点。场景、物理、因果和风险事实集中于MALFUNCTIONS复合节点；各阶段按各自身份计数。", **{"class": "scope-note"}))
    nav.append(tag("a", "13步生产工作流", href="#workflow"))
    step_nav = soup.new_tag("details"); step_nav.append(tag("summary", "展开13步")); nav.append(step_nav)
    step_index = soup.new_tag("div", attrs={"class": "workflow-index"}); main.append(step_index)
    step_diagrams = {"wf-06": "diagram-identities", "wf-08": "diagram-physics",
                     "wf-11": "diagram-time", "wf-12": "diagram-report"}
    for source in sorted(HERE.glob("0[012]_*.md")):
        for ident, (title, prose) in sections(source.read_text(), "wf-").items():
            card = soup.new_tag("section", attrs={"class": "workflow-step", "id": ident})
            head = soup.new_tag("div", attrs={"class": "step-head"})
            head.append(tag("span", ident[-2:], **{"class": "step-no"})); head.append(tag("h3", title)); card.append(head)
            detail = fragment(prose, source, ident)
            toc = soup.new_tag("div", attrs={"class": "step-toc"})
            for h in detail.select("h4"):
                toc.append(tag("a", h.get_text(), href="#" + h["id"]))
            card.append(toc)
            if ident in step_diagrams:
                drawing = diagram(step_diagrams[ident])
                drawing.h3.name = "h4"
                card.append(drawing)
            card.append(detail); main.append(card)
            step_index.append(tag("a", ident[-2:] + " " + title, href="#" + ident))
            step_nav.append(tag("a", ident[-2:] + " " + title, href="#" + ident, **{"class": "subnav"}))

    current_section("section-3")
    main.append(tag("p", "以下为技术专题：把前述流程涉及的筛选、复用、方法契约和评分条件集中说明，便于按问题定位。执行顺序仍以前面的13步流程为准，不代表另有一套分析链。", **{"class": "scope-note"}))
    source_chapters("section-4", "筛选、逻辑校验、复用与结果去向", "03_场景物理与风险计算.md", [3, 6, 9])
    source = HERE / "05_语义任务与执行职责.md"
    section("section-5", "模型任务、上下文与生成内容", source.read_text().split("\n", 1)[1], source)
    source_chapters("section-6", "数据接口、来源与跨项目范围", "02_输入抽取数据契约与方法治理.md", [5, 6, 7, 8, 9])
    source_chapters("section-7", "S/E/C、分支ASIL及控制通道", "03_场景物理与风险计算.md", [7, 8])
    nav.append(tag("div", "五份输入的实际结果", **{"class": "nav-group"}))
    current_section("section-8")
    chart = soup.new_tag("figure", attrs={"class": "process-figure result-chart", "id": "diagram-results", "aria-labelledby": "diagram-results-title"})
    chart.append(tag("h3", "图 8｜五份最新结果：C有数值和无数值的比例", id="diagram-results-title"))
    chart.append(tag("p", "深色表示已有数值（含条件性待审核）；浅色表示尚无数值。每条横条的总长代表该项目全部活动风险，不表示项目之间风险数量相同。", **{"class": "figure-intro"}))
    for row in project_metrics:
        item = soup.new_tag("div", attrs={"class": "result-bar-row"})
        item.append(tag("b", row["project"]))
        track = soup.new_tag("div", attrs={"class": "result-bar", "aria-hidden": "true"})
        track.append(tag("span", style=f"width:{100 * row['c_known'] / row['risk']:.6f}%"))
        item.append(track)
        item.append(tag("span", f"有值 {row['c_known']} / 无值 {row['c_missing']} / 共 {row['risk']}"))
        chart.append(item)
    chart.append(tag("figcaption", "图与本节表格来自同一份项目统计；有C数值不等于工程审批已经完成，也不表示剩余缺口都应通过重新调用模型解决。"))
    main.append(chart)
    for ident in range(9, 16):
        current_section(f"section-{ident}")

    nav.append(tag("div", "运行、开发与代码实例", **{"class": "nav-group"}))
    for name, (ident, title) in APPENDICES.items():
        source = HERE.parent / name
        text = source.read_text(); text = text[text.index("\n## ") + 1:]
        section(ident, title, text, source)
        if ident == "appendix-c":
            main.append(diagram("diagram-recovery"))
    source = HERE / "README.md"
    section("appendix-f", "附录 D｜当前文档构建、验证与交付边界", source.read_text().split("\n", 1)[1], source)
    source = HERE / "04_代码实现与问题定位索引.md"
    section("appendix-g", "附录 E｜真实代码实例、问题定位与数据对象索引", source.read_text().split("\n", 1)[1], source)
    main.append(tag("p", "HARA 开发与技术交接 · 五份输入最新结果版 · 2026-11-24", **{"class": "bottom"}))

    location = soup.new_tag("div", attrs={"class": "reading-location"})
    location.append(tag("span", "当前位置"))
    location.append(tag("strong", id="reading-chapter"))
    location.append(tag("small", id="reading-subsection"))
    nav.insert(0, location)
    for link in list(nav.select("a[href]")):
        ident = link["href"][1:]
        target = main.find(id=ident)
        if target is None or not (target.name == "h2" or "workflow-step" in target.get("class", [])):
            continue
        entry = soup.new_tag("div", attrs={"class": "toc-entry", "data-chapter": ident})
        link.wrap(entry)
        link["class"] = link.get("class", []) + ["toc-chapter"]
        link["data-reading-target"] = ident
        headings = []
        if target.name == "section":
            headings = target.select("h3[id],h4[id],h5[id],h6[id]")
        else:
            for sibling in target.next_siblings:
                if getattr(sibling, "name", None) == "h2" or "workflow-step" in getattr(sibling, "attrs", {}).get("class", []):
                    break
                if hasattr(sibling, "select"):
                    headings.extend(sibling.select("h3[id],h4[id],h5[id],h6[id]"))
        children = soup.new_tag("div", attrs={"class": "toc-subsections"})
        for heading in headings:
            children.append(tag("a", heading.get_text(), href="#" + heading["id"],
                                **{"data-reading-target": heading["id"], "class": "toc-detail"}))
        entry.append(children)
    script = soup.new_tag("script", attrs={"id": "report-navigation"})
    script.string = (HERE / "report_navigation.js").read_text()
    soup.body.append(script)

    manifest = {"revision": "detailed-readable-workflow-20261124", "production_code": CODE,
                "documentation_base": DOCS, "result_date": "2026-11-23", "project_metrics": project_metrics,
                "new_provider_requests": 0, "scope": "Existing full populations plus targeted revisions; no new full run or engineering approval.",
                "document_sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)}}
    metadata = soup.new_tag("script", attrs={"type": "application/json", "id": "handover-evidence"})
    metadata.string = json.dumps(manifest, ensure_ascii=False); soup.body.append(metadata)
    all_ids = [x["id"] for x in soup.select("[id]")]
    duplicate_ids = sorted({i for i in all_ids if all_ids.count(i) > 1})
    required_ids = {"workflow", *concepts, *[f"section-{i}" for i in range(16)], *[f"wf-{i:02d}" for i in range(13)], *["code-"+e["id"].lower() for e in examples],
                    *["diagram-"+name for name in ("main", "authority", "identities", "physics", "result-state", "time", "report", "recovery", "results")]}
    missing_ids = sorted(required_ids - set(all_ids))
    broken_anchors = [x["href"] for x in soup.select("a[href^='#']") if unquote(x["href"][1:]) not in all_ids]
    legacy = soup.select(".historical-section,#report-evidence,#toggle-implementation,img")
    if duplicate_ids or missing_ids or broken_anchors or legacy:
        raise ValueError({"duplicate_ids": duplicate_ids, "missing_ids": missing_ids, "broken_anchors": broken_anchors, "legacy_nodes": len(legacy)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(line.rstrip() for line in str(soup).splitlines()) + "\n", encoding="utf-8")
    text = " ".join(str(x).strip() for x in soup.find_all(string=True) if x.parent.name not in {"style", "script"} and str(x).strip())
    report = {"output": str(args.output.resolve()), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "html_bytes": args.output.stat().st_size,
              "visible_text_chars": len(text), "workflow_steps": len(soup.select(".workflow-step")), "workflow_subsections": len(soup.select(".workflow-step .current-detail h4")),
              "tables": len(soup.select("table")), "svg_figures": len(soup.select("svg")), "legacy_sections": len(legacy), "local_source_links_checked": len(local_links),
              "readable_figures": len(soup.select("figure.process-figure")), "concept_chapters": len(concepts),
              "navigation_chapters": len(nav.select(".toc-entry")), "navigation_targets": len(nav.select("[data-reading-target]")),
              "collapsible_source_examples": len(soup.select("details.source-code")),
              "source_examples_checked": len(examples), "source_examples": examples, "project_metrics": project_metrics,
              "broken_anchors": broken_anchors, "duplicate_ids": duplicate_ids, "missing_ids": missing_ids, "new_provider_requests": 0, "local_links": local_links}
    if args.qa_output:
        args.qa_output.parent.mkdir(parents=True, exist_ok=True)
        args.qa_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"local_links", "source_examples"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
