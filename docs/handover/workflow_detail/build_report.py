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
    "05_部署配置运行与恢复.md": ("appendix-c", "附录 C｜安装、配置、运行与恢复操作手册"),
    "06_开发测试与维护.md": ("appendix-d", "附录 D｜开发维护、测试地图与修改方法"),
    "08_资产移交与接手清单.md": ("appendix-e", "附录 E｜五项目资产、检查点与接手清单"),
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
    local_links: list[dict] = []
    examples: list[dict] = []
    source_files: set[Path] = set()

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
        for i, heading in enumerate(node.select("h2,h3,h4,h5")):
            heading.name = "h4" if prefix.startswith("wf-") else "h3" if heading.name == "h2" else heading.name
            example = re.match(r"实现例 ([A-Z]\d+)｜", heading.get_text())
            heading["id"] = "code-" + example[1].lower() if example else f"{prefix}-part-{i + 1}"
        for anchor in node.select("a[href]"):
            href = anchor["href"]
            parts = urlsplit(href)
            if parts.scheme or href.startswith("#"):
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
            content.append(f"### {heading[1]}\n\n" + chapter(source, number))
        section(ident, title, "\n\n".join(content), source)

    def figure(svg, caption):
        box = soup.new_tag("figure")
        scroll = soup.new_tag("div", attrs={"class": "diagram-scroll"})
        parsed = BeautifulSoup(svg, "html.parser").svg
        parsed["viewBox"] = parsed.attrs.pop("viewbox")
        scroll.append(parsed)
        box.append(scroll)
        box.append(tag("figcaption", caption))
        return box

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

    soup.title.string = "HARA 开发与技术交接｜五份输入最新测试结果"
    soup.header.append(tag("b", "HARA / ENGINEERING HANDOVER"))
    soup.header.append(tag("span", "五份输入最新修订结果 · 2026-11-24 · 代码 d6a400e"))
    soup.header.append(tag("button", "打印 / 保存 PDF", onclick="window.print()", type="button"))
    style = soup.new_tag("style")
    style.string = (HERE / "report_style.css").read_text()
    soup.head.append(style)
    main.append(tag("h1", "HARA 开发与技术交接"))
    main.append(tag("h2", "工作流工程详解 · 五份输入最新测试结果", id="section-0"))
    nav.append(tag("b", "报告目录"))
    nav.append(tag("a", "五份输入与当前基线", href="#section-0"))
    cover = soup.new_tag("div", attrs={"class": "handover-cover"})
    cover.append(tag("p", "文档核对：2026-11-24 ｜ 业务结果：2026-11-23完整修订 ｜ 生产代码：d6a400e", **{"class": "edition"}))
    cover.append(tag("p", "本报告仅呈现AVP、IPA-High、IPA-Main、RPA、rTPA五份输入的最新可核查结果，以及当前13步生产流程、17个代码实例和开发交接说明。五份结果来自已完成的全量与后续定向修订，不是本次又从头运行五次全量。"))
    cover.append(tag("p", "有数值、计算完成、条件性结果与工程批准分别标明。报告中的本机证据链接需对应运行资产；正文与图表可以离线阅读。本次文档更新不调用Provider、不重新评分。"))
    links = soup.new_tag("div", attrs={"class": "quick-links"})
    for ident, title in [("section-8", "五份最新结果"), ("section-11", "当前剩余问题"), ("workflow", "13步生产流程"), ("appendix-g", "代码与问题索引"), ("section-15", "五份证据索引")]:
        links.append(tag("a", title, href="#" + ident))
    cover.append(links)
    main.append(cover)
    cards = soup.new_tag("div", attrs={"class": "metrics"})
    for value, label in [(5, "输入项目"), (sum(x["risk"] for x in project_metrics), "活动风险成员"), (sum(x["main"] for x in project_metrics), "HARA主表审核组"), (sum(x["sg"] for x in project_metrics), "已关联SG · 非工程批准数")]:
        card = soup.new_tag("div", attrs={"class": "metric"})
        card.append(tag("strong", f"{value:,}")); card.append(tag("span", label)); cards.append(card)
    main.append(cards)

    def current_section(ident):
        title, text = current[ident]
        section(ident, title, text, current_source)

    current_section("section-1")
    source_chapters("section-2", "程序架构、状态与职责边界", "01_架构与端到端流程.md", [1, 4, 5, 6, 7])
    main.append(figure('''<svg viewBox="0 0 960 340" role="img" aria-label="当前三类输入与统一分析链">
    <defs><marker id="arch-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8Z" fill="#087e87"/></marker></defs>
    <g fill="#edf7f4" stroke="#8ebeb8"><rect x="20" y="20" width="270" height="58" rx="8"/><rect x="345" y="20" width="270" height="58" rx="8"/><rect x="670" y="20" width="270" height="58" rx="8"/><rect x="20" y="120" width="270" height="58" rx="8"/><rect x="345" y="120" width="270" height="58" rx="8"/><rect x="670" y="120" width="270" height="58" rx="8"/><rect x="150" y="230" width="405" height="70" rx="8"/><rect x="650" y="230" width="290" height="70" rx="8"/></g>
    <g text-anchor="middle" font-size="17"><text x="155" y="55">产品输入：DOCX / PDF</text><text x="480" y="55">方法来源：模板或治理YAML</text><text x="805" y="55">报告模板：工作簿布局</text><text x="155" y="155">功能与有来源的项目事实</text><text x="480" y="155">唯一 MethodContract</text><text x="805" y="155">ReportContract</text><text x="352" y="260">统一场景、物理、因果与风险分析</text><text x="352" y="286">HARAState 与完整追溯</text><text x="795" y="260">报告投影与模板写入</text><text x="795" y="286">Excel 与关联明细</text></g>
    <g fill="none" stroke="#087e87" stroke-width="2" marker-end="url(#arch-arrow)"><path d="M155 78V120"/><path d="M480 78V120"/><path d="M805 78V120"/><path d="M155 178V205H270V230"/><path d="M480 178V230"/><path d="M805 178V230"/><path d="M555 265H650"/></g></svg>''', "三类输入的权威分开：项目资料提供事实，活动方法决定计算，报告契约决定呈现。"))

    main.append(tag("h2", "当前工作流：逐步处理、组合与结果流转", id="workflow"))
    main.append(tag("p", "00—12是业务讲解顺序，不是13个独立WorkflowGraph节点。场景、物理、因果和风险事实集中于MALFUNCTIONS复合节点；各阶段按各自身份计数。", **{"class": "scope-note"}))
    nav.append(tag("a", "13步生产工作流", href="#workflow"))
    step_nav = soup.new_tag("details"); step_nav.append(tag("summary", "展开13步")); nav.append(step_nav)
    step_index = soup.new_tag("div", attrs={"class": "workflow-index"}); main.append(step_index)
    for source in sorted(HERE.glob("0[012]_*.md")):
        for ident, (title, prose) in sections(source.read_text(), "wf-").items():
            card = soup.new_tag("section", attrs={"class": "workflow-step", "id": ident})
            head = soup.new_tag("div", attrs={"class": "step-head"})
            head.append(tag("span", ident[-2:], **{"class": "step-no"})); head.append(tag("h3", title)); card.append(head)
            detail = fragment(prose, source, ident)
            toc = soup.new_tag("div", attrs={"class": "step-toc"})
            for h in detail.select("h4"):
                toc.append(tag("a", h.get_text(), href="#" + h["id"]))
            card.append(toc); card.append(detail); main.append(card)
            step_index.append(tag("a", ident[-2:] + " " + title, href="#" + ident))
            step_nav.append(tag("a", ident[-2:] + " " + title, href="#" + ident, **{"class": "subnav"}))

    current_section("section-3")
    source_chapters("section-4", "筛选、逻辑校验、复用与结果去向", "03_场景物理与风险计算.md", [3, 6, 9])
    source = HERE / "05_语义任务与执行职责.md"
    section("section-5", "模型任务、上下文与生成内容", source.read_text().split("\n", 1)[1], source)
    source_chapters("section-6", "数据接口、来源与跨项目范围", "02_输入抽取数据契约与方法治理.md", [5, 6, 7, 8, 9])
    source_chapters("section-7", "S/E/C、分支ASIL及控制通道", "03_场景物理与风险计算.md", [7, 8])
    nav.append(tag("div", "五份输入的实际结果", **{"class": "nav-group"}))
    current_section("section-8")
    bars = []
    for index, row in enumerate(project_metrics):
        y = 60 + index * 55
        width = 500 * row["c_known"] / row["risk"]
        bars.append(f'<text x="20" y="{y+20}" font-size="17">{row["project"]}</text><rect x="145" y="{y}" width="500" height="28" rx="3" fill="#f3d5b5"/><rect x="145" y="{y}" width="{width:.3f}" height="28" fill="#087e87"/><text x="665" y="{y+19}" font-size="16">有值 {row["c_known"]} / 无值 {row["c_missing"]} / 共 {row["risk"]}</text>')
    chart = '<svg viewBox="0 0 1000 365" role="img" aria-label="五份最新结果的C数值完整度"><rect x="145" y="16" width="18" height="16" fill="#087e87"/><text x="172" y="30" font-size="14">已有C数值（含条件性待审核）</text><rect x="470" y="16" width="18" height="16" fill="#f3d5b5"/><text x="497" y="30" font-size="14">尚无C数值</text>' + ''.join(bars) + '</svg>'
    main.append(figure(chart, "图中分母为各项目的活动风险成员，数据与本节表格同源；有C数值不等于方法适用性和工程审批已经全部完成。"))
    for ident in range(9, 16):
        current_section(f"section-{ident}")

    nav.append(tag("div", "运行、开发与代码实例", **{"class": "nav-group"}))
    for name, (ident, title) in APPENDICES.items():
        source = HERE.parent / name
        text = source.read_text(); text = text[text.index("\n## ") + 1:]
        section(ident, title, text, source)
    source = HERE / "README.md"
    section("appendix-f", "附录 F｜当前文档构建、验证与交付边界", source.read_text().split("\n", 1)[1], source)
    source = HERE / "04_代码实现与问题定位索引.md"
    section("appendix-g", "附录 G｜真实代码实例、问题定位与数据对象索引", source.read_text().split("\n", 1)[1], source)
    main.append(tag("p", "HARA 开发与技术交接 · 五份输入最新结果版 · 2026-11-24", **{"class": "bottom"}))

    manifest = {"revision": "five-input-current-results-20261124", "production_code": CODE,
                "documentation_base": DOCS, "result_date": "2026-11-23", "project_metrics": project_metrics,
                "new_provider_requests": 0, "scope": "Existing full populations plus targeted revisions; no new full run or engineering approval.",
                "document_sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_files)}}
    metadata = soup.new_tag("script", attrs={"type": "application/json", "id": "handover-evidence"})
    metadata.string = json.dumps(manifest, ensure_ascii=False); soup.body.append(metadata)
    all_ids = [x["id"] for x in soup.select("[id]")]
    duplicate_ids = sorted({i for i in all_ids if all_ids.count(i) > 1})
    required_ids = {"workflow", *[f"section-{i}" for i in range(16)], *[f"wf-{i:02d}" for i in range(13)], *["code-"+e["id"].lower() for e in examples]}
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
              "source_examples_checked": len(examples), "source_examples": examples, "project_metrics": project_metrics,
              "broken_anchors": broken_anchors, "duplicate_ids": duplicate_ids, "missing_ids": missing_ids, "new_provider_requests": 0, "local_links": local_links}
    if args.qa_output:
        args.qa_output.parent.mkdir(parents=True, exist_ok=True)
        args.qa_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"local_links", "source_examples"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
