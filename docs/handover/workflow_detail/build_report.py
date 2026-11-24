"""Extend the supplied HTML review with maintained, code-linked handover prose.

Document tooling only. Optional dependencies: beautifulsoup4 and Markdown.
No imports of the production runtime, Provider calls, or analysis-data writes.
"""

from __future__ import annotations

import argparse
import copy
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
DOCS = "4429a2d1cd095a6fa699f0dba135bcff8816defa"
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
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qa-output", type=Path)
    args = parser.parse_args()
    if args.reference.resolve() == args.output.resolve():
        raise ValueError("Keep the user-supplied reference unchanged; use a new output.")
    original = args.reference.read_bytes()
    soup = BeautifulSoup(original.decode("utf-8"), "html.parser")
    original_ids = {x["id"] for x in soup.select("[id]")}
    original_images = [hashlib.sha256(x["src"].encode()).hexdigest() for x in soup.select("img")]
    local_links: list[dict] = []
    examples: list[dict] = []

    def tag(name: str, text: str = "", **attrs):
        element = soup.new_tag(name, attrs=attrs)
        element.string = text
        return element

    def fragment(text: str, source: Path, prefix: str):
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
            if target.suffix == ".md" and target.name in DOC_ANCHORS and target.parent == HERE.parent:
                anchor["href"] = "#" + DOC_ANCHORS[target.name]
                anchor["title"] = f"同页主题入口；完整源章：{target.relative_to(ROOT)}"
            elif target.is_relative_to(ROOT / "output") or target.is_relative_to(ROOT / "runtime"):
                anchor["href"] = target.as_uri() + ("#" + parts.fragment if parts.fragment else "")
                anchor["class"] = anchor.get("class", []) + ["local-evidence"]
                anchor["title"] = "本机历史证据；不包含在HTML中，异机需取得资产包"
            else:
                relative = target.relative_to(ROOT)
                version = DOCS if relative.parts[0] == "docs" else CODE
                route = "tree/" if target.is_dir() else "blob/"
                anchor["href"] = GITHUB + route + version + "/" + quote(str(relative), safe="/") + ("#" + parts.fragment if parts.fragment else "")
                anchor["title"] = f"{relative} @ {version[:7]}"
        for table in list(node.select("table")):
            wrapper = soup.new_tag("div", attrs={"class": "table-scroll"})
            table.wrap(wrapper)
        return node

    def historical(nodes, label="原稿记录｜2026-10-22 · 652f0774 · 非当前结论"):
        box = soup.new_tag("details", attrs={"class": "historical-section"})
        box.append(tag("summary", label))
        box.append(tag("p", "以下保留原稿文字与图表。其‘当前’、‘本轮’均指原稿时点；当前实现及交接结论请以上方增补为准。", **{"class": "history-notice"}))
        for node in nodes:
            box.append(node.extract())
        return box

    def body_of(section_id: str):
        heading = soup.find(id=section_id)
        result = []
        for node in list(heading.next_siblings):
            if getattr(node, "name", None) == "h2" or (getattr(node, "name", None) and "bottom" in node.get("class", [])):
                break
            result.append(node)
        return heading, result

    soup.title.string = "HARA 开发与技术交接｜工作流工程详解增强版"
    soup.main.h1.string = "HARA 开发与技术交接"
    soup.find(id="section-0").string = "工作流工程详解增强版 · 沿原报告13步逐层读代码"
    top_heading, old_top = body_of("section-0")
    top_history = historical(old_top, "原稿封面与适用范围｜2026-10-22")
    top_heading.insert_after(top_history)
    cover = tag("div", **{"class": "handover-cover"})
    cover.clear()
    cover.append(tag("p", "更新：2026-11-24 ｜ 生产代码 d6a400e ｜ 文档起点 4429a2d ｜ 既有证据截至2026-11-23", **{"class": "edition"}))
    cover.append(tag("p", "本版保留原稿的目录、13步工作流、图示放大与打印。每一步补入实际调用、字段与来源、条件分支、错误去向、成本及恢复机制；重点展开物理方案、因果资格、S/E/C、SG、Safe State、FTTI和关联报告。"))
    cover.append(tag("p", "正文为当前核对说明；橙色折叠区保留历史原文。代码链接锁定生产提交；带↗本机证据的链接需要原项目资料，转发HTML本身不包含这些大文件。本次仅编写文档，0 Provider，不改变计算或批准。"))
    jump = soup.new_tag("div", attrs={"class": "quick-links"})
    for ident, label in [("workflow", "13步调用详解"), ("appendix-g", "代码实例与问题索引"), ("wf-08", "物理方案与模型"), ("wf-12", "SG / Safe State / FTTI / Excel"), ("appendix-c", "运行与恢复命令"), ("section-11", "真实未决与边界")]:
        jump.append(tag("a", label, href="#" + ident))
    cover.append(jump)
    top_heading.insert_after(cover)
    header_label = soup.header.find("span")
    if header_label:
        header_label.string = "工程详解增强版 · 2026-11-24 · 代码 d6a400e"
    toggle = soup.find(id="toggle-implementation")
    toggle.string = "显示历史原稿"

    # Maintain all original workflow anchors and enrich those exact cards.
    for source in sorted(HERE.glob("0[012]_*.md")):
        for ident, (title, prose) in sections(source.read_text(), "wf-").items():
            card = soup.find(id=ident)
            head, lead = card.select_one(".step-head"), card.select_one(".step-lead")
            old_nodes = [x for x in list(card.contents) if x is not head and x is not lead]
            history = historical(old_nodes, f"对照原稿 {ident}｜旧摘要和旧实现说明")
            title_tag = head.find(re.compile("^h[1-6]$"))
            title_tag.name = "h3"
            # Original role badge may be nested in heading: preserve it separately.
            badge = title_tag.select_one(".role-badge")
            if badge:
                badge.extract()
            title_tag.clear()
            title_tag.append(title)
            if badge:
                title_tag.append(badge)
            current = fragment(prose, source, ident)
            toc = soup.new_tag("div", attrs={"class": "step-toc", "aria-label": "本步详细目录"})
            for h in current.select("h4"):
                toc.append(tag("a", h.get_text(), href="#" + h["id"]))
            card.append(tag("p", "当前生产实现 · 调用 / 字段 / 分支 / 恢复 / 例子 / 测试", **{"class": "current-marker"}))
            card.append(toc)
            card.append(current)
            card.append(history)

    # Keep historical quantity graphics, but do not let old counts describe new code.
    old_flow = soup.find(id="workflow").find_next("h3")
    flow_nodes = [old_flow]
    for x in list(old_flow.next_siblings):
        if getattr(x, "name", None) and "workflow-index" in x.get("class", []):
            break
        flow_nodes.append(x)
    wrapper = historical([], "原v8数量流转图｜408 → 816 → 1596 → 1000，仅为历史运行")
    old_flow.insert_before(wrapper)
    for x in flow_nodes:
        wrapper.append(x.extract())
    wrapper.insert_before(tag("p", "阅读方法：00—12是业务讲解顺序，不是13个独立WorkflowGraph节点。场景、物理、因果和风险事实集中在MALFUNCTIONS复合节点。不同阶段按各自身份计数，不能把参数点、驾驶员分支、HTTP尝试和Excel行数相加。", **{"class": "scope-note"}))
    matrix = soup.select_one(".matrix-block")
    if matrix:
        box = historical([], "原v8功能×引导词矩阵｜9功能／61失效，仅作示例")
        matrix.insert_before(box)
        box.append(matrix.extract())

    extras = HERE / "03_当前结果与交接说明.md"
    for ident, (title, prose) in sections(extras.read_text(), "section-").items():
        heading, old_nodes = body_of(ident)
        history = historical(old_nodes, "原稿对应章节与图表｜保留作历史／规划对照")
        heading.string = title
        heading.insert_after(history)
        heading.insert_after(fragment(prose, extras, ident))

    # Current scoring explanation replaces historical numeric policy descriptions.
    h, old = body_of("section-7")
    h.insert_after(historical(old, "原10月评分方法说明｜不能作为当前策略依据"))
    source = HERE.parent / "03_场景物理与风险计算.md"
    h.insert_after(fragment(chapter(source, 7) + "\n\n### 控制通道的职责\n\n" + chapter(source, 8), source, "current-risk"))
    h.string = "S/E/C、分支ASIL及控制通道：当前执行边界"

    # Add checked detail to timeless architecture/source/task explanations.
    insertions = {
        "section-2": (HERE.parent / "01_架构与端到端流程.md", [4, 6, 7]),
        "section-4": (HERE.parent / "03_场景物理与风险计算.md", [9]),
        "section-6": (HERE.parent / "02_输入抽取数据契约与方法治理.md", [5, 6, 8, 9]),
    }
    for ident, (source, numbers) in insertions.items():
        heading, body = body_of(ident)
        addition = fragment("\n\n".join(chapter(source, n) for n in numbers), source, ident + "-current")
        addition.insert(0, tag("h3", "2026-11-24：对应代码职责与交接细节"))
        if body:
            body[-1].insert_after(addition)
        else:
            heading.insert_after(addition)
    # Replace only the obsolete architecture operations table, preserving both diagrams.
    for h3 in soup.select("main > h3"):
        if h3.get_text().startswith("2.4"):
            table = h3.find_next("div", class_="table-scroll")
            table.insert_before(tag("p", "当前普通生产默认8路；有界样本为1路。全量硬预算不由默认CLI自动创建，需核对实际客户端装配。下表原v8运行条件保留在历史区，当前配置及恢复命令见附录C。", **{"class": "scope-note"}))
            box = historical([], "原v8并发与预算说明｜非当前全量CLI配置")
            table.insert_before(box)
            box.append(table.extract())
    # Older schema illustration is useful, but explicitly not a current callable payload.
    old_contract = soup.select_one("details.json-contract")
    if old_contract:
        introduction = old_contract.find_previous("p")
        introduction.clear()
        introduction.append("下面保留原稿的物理方案返回节选，只说明历史字段形状；当前完整请求与输出字段见wf-08。")
        old_contract["class"] = ["historical-section", "json-contract"]
        old_contract.summary.string = "原稿契约节选｜仅历史示意；当前字段看wf-08"
    for h3 in soup.select("main > h3"):
        if h3.get_text().startswith("5.3"):
            table = h3.find_next("div", class_="table-scroll")
            current_table = copy.deepcopy(table)
            for cell in current_table.select("td"):
                text = cell.get_text()
                if "当前 v8" in text:
                    cell.string = "子场景仍须重新绑定并通过证据校验；没有依赖证明不能复制完整判断。实际收益按本次任务账本核对。"
                elif "1,000" in text:
                    cell.string = "风险明细及原始身份不因主表归组而消失；缩表不是节省Provider请求的证明。"
            box = historical([], "原5.3对照表｜v8复用与1000条／60组为历史口径")
            table.insert_before(current_table)
            table.insert_before(box)
            box.append(table.extract())

    for ident in ["section-2", "section-4", "section-5", "section-6", "section-14"]:
        soup.find(id=ident).insert_after(tag("p", "本节沿用原稿中仍成立的通用架构与职责说明；S/W编号为历史引用，最新字段和调用边界由步骤详解及本节增补的d6a400e源码链接说明。原稿中提及v8的数量仅属于10月历史运行。", **{"class": "scope-note"}))
    soup.find(id="section-5").insert_after(tag("p", "当前控制输入补全、保持固定轨迹的响应纠正都由既有PhysicalScenarioAgent承担，不是新增第九个Agent。remaining/failed/unresolved三态、人员范围和required command path分别校验；完整运动、控制补全和响应纠正的请求内容见wf-08。"))
    soup.find(id="section-15").string = "附录 B｜原稿历史证据索引（652f0774）"

    # Append full operational chapters inside this same HTML (not links-only).
    footer = soup.select_one("main > .bottom")
    for name, (ident, title) in APPENDICES.items():
        source = HERE.parent / name
        text = source.read_text()
        text = text[text.index("\n## ") + 1:]
        heading = tag("h2", title, id=ident)
        footer.insert_before(heading)
        footer.insert_before(fragment(text, source, ident))
    appendix = tag("h2", "附录 F｜本版扩写范围、证据版本与验证", id="appendix-f")
    footer.insert_before(appendix)
    footer.insert_before(tag("p", "本版完整读取原HTML可见正文，沿原13步扩写；保留原图和原记录，当前正文以锁定生产代码及已保存五项目证据为准。新加三个完整操作附录。生产代码、方法资产、原始输入、checkpoint、响应、审批和.env均未更改；没有新增业务运行。"))
    footer.insert_before(tag("p", "代码链接固定d6a400e；既有交接章节固定4429a2d。原稿S/W索引仍指其历史提交与工作簿。两套证据不混合。默认方法与五份报告冻结方法并不相同，重导必须使用proof给出的快照；具体路径和哈希见附录E。"))
    footer.insert_before(tag("p", "完整静态检查、实际门禁及预览范围见配套扩写说明；不把历史视觉选区验证说成本次新截图。HTML无需联网即可阅读，源码链接需GitHub访问权限，本机证据链接需移交对应资产。", id="validation-note"))
    footer.insert_before(tag("p", "2026-11-24首版扩写验证：pytest 2176 passed、1 skipped（历史守卫opt-in），136.21秒；doctor、compileall和diff检查通过。doctor为ready_for_draft=true、ready_for_release=false。浏览器曾成功打开新正文并取得可访问性内容，截图超时；后续页面变化触发受限URL，预览停止。没有完成截图级、窄屏或打印分页验收。后续代码实例增补的检查另见配套README与验证记录，不由重新构建HTML自动产生。", **{"class": "scope-note"}))
    source = HERE / "04_代码实现与问题定位索引.md"
    heading = tag("h2", "附录 G｜真实代码实例、问题定位与数据对象索引", id="appendix-g")
    footer.insert_before(heading)
    footer.insert_before(fragment(source.read_text().split("\n", 1)[1], source, "appendix-g"))
    footer.string = "HARA 开发与技术交接 · 工作流工程详解增强版 · 2026-11-24 · 保留原稿及历史证据"

    # Current chapter titles in the existing sidebar plus new operational appendices.
    for a in soup.select("nav a[href^='#']"):
        target = soup.find(id=a["href"][1:])
        if target and target.name == "h2":
            a.string = target.get_text()
    nav = soup.select_one("nav")
    nav.append(tag("div", "交接操作附录", **{"class": "nav-group-title"}))
    for ident in ["appendix-c", "appendix-d", "appendix-e", "appendix-f", "appendix-g"]:
        nav.append(tag("a", soup.find(id=ident).get_text(), href="#" + ident))

    # Local-only reading controls: original diagrams/zoom retained; history off by default.
    for script in soup.select("script:not([type='application/json'])"):
        content = script.string or script.get_text()
        content = content.replace("details.implementation", "details.historical-section")
        content = content.replace("收起实现细节", "隐藏历史原稿").replace("隐藏实现细节", "隐藏历史原稿").replace("显示实现细节", "显示历史原稿")
        script.string = content
    behavior = soup.new_tag("script")
    behavior.string = """function openTarget(){const id=decodeURIComponent(location.hash.slice(1));const el=document.getElementById(id);if(!el)return;let p=el.parentElement;while(p){if(p.tagName==='DETAILS')p.open=true;p=p.parentElement;}} window.addEventListener('hashchange',openTarget);openTarget();"""
    soup.body.append(behavior)
    style = soup.new_tag("style")
    style.string = """
    .handover-cover{border:1px solid #99cbbd;border-left:6px solid #137c68;background:#f2faf7;padding:22px 26px;margin:20px 0}
    .edition,.current-marker{color:#0b6f5e;font-weight:700;font-size:13px;letter-spacing:.3px}
    .quick-links,.step-toc{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0}
    .quick-links a,.step-toc a{display:inline-block;background:#edf5f3;border:1px solid #cfe2dc;border-radius:5px;padding:6px 10px;font-size:13px;text-decoration:none}
    .workflow-step{overflow:visible;scroll-margin-top:90px;break-inside:auto}
    .step-head h3{margin:0;font-size:22px;line-height:1.45;border:0;padding:0}
    .current-detail h4{font-size:18px;color:#174c43;margin:30px 0 12px;border-bottom:1px solid #deebe6;padding-bottom:8px;scroll-margin-top:100px}
    .current-detail h3{scroll-margin-top:100px}.current-detail{min-width:0}.current-detail p,.current-detail li{line-height:1.9}
    .current-detail code{overflow-wrap:anywhere;font-size:.88em}.current-detail pre code{white-space:pre-wrap;overflow-wrap:anywhere}
    .current-detail pre{white-space:pre-wrap;max-width:100%;overflow-x:auto;font-size:12px;line-height:1.7}
    .current-detail table{font-size:13px;min-width:600px}.current-detail td{vertical-align:top;overflow-wrap:anywhere}
    .historical-section{margin:22px 0;padding:14px 18px;border:1px solid #ddcaa9;background:#fffaf1;border-radius:8px}
    .historical-section>summary{color:#7b551d;font-weight:650;cursor:pointer;line-height:1.6}
    .historical-section[open]>summary{margin-bottom:14px}.history-notice{font-size:12px!important;background:#fff0d9;padding:8px 10px;color:#795521}
    .local-evidence::after{content:' ↗本机证据';font-size:10px;color:#7b705d}.nav-group-title{font-weight:700;margin:22px 10px 6px;color:#4d6d63}
    nav>a{display:block}h2{scroll-margin-top:90px}
    @media(max-width:800px){.handover-cover{padding:14px}.step-head h3{font-size:18px}.current-detail table{font-size:12px}.step-toc a{font-size:12px}.workflow-step{padding:16px}}
    @media print{.historical-section{display:none!important}.step-toc,.quick-links{display:none!important}.workflow-step{break-inside:auto!important}.current-detail h3,.current-detail h4{break-after:avoid}.current-detail table{min-width:0;font-size:9pt}.current-detail pre{font-size:8pt}.table-scroll{overflow:visible!important}.current-detail a{color:inherit}.local-evidence::after{content:'（本机证据）'}}
    """
    soup.head.append(style)
    metadata = soup.new_tag("script", attrs={"type": "application/json", "id": "handover-evidence"})
    metadata.string = json.dumps({"revision": "workflow-engineering-detail-20261124", "production_code": CODE, "documentation_base": DOCS, "reference_sha256": hashlib.sha256(original).hexdigest(), "new_provider_requests": 0, "scope": "Documentation only; conditional historical results are not engineering approval."}, ensure_ascii=False)
    soup.body.append(metadata)
    # Original SVG/XML viewBox casing must survive HTML parsing for scalable diagrams.
    for svg in soup.select("svg"):
        if "viewbox" in svg.attrs:
            svg["viewBox"] = svg.attrs.pop("viewbox")

    all_ids = [x["id"] for x in soup.select("[id]")]
    duplicate_ids = sorted({i for i in all_ids if all_ids.count(i) > 1})
    lost_ids = sorted(original_ids - set(all_ids))
    broken_anchors = [x["href"] for x in soup.select("a[href^='#']") if unquote(x["href"][1:]) not in all_ids]
    new_images = [hashlib.sha256(x["src"].encode()).hexdigest() for x in soup.select("img")]
    if duplicate_ids or lost_ids or broken_anchors or new_images != original_images:
        raise ValueError({"duplicate_ids": duplicate_ids, "lost_ids": lost_ids, "broken_anchors": broken_anchors, "images_unchanged": new_images == original_images})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(line.rstrip() for line in str(soup).splitlines()) + "\n", encoding="utf-8")
    text = " ".join(str(x).strip() for x in soup.find_all(string=True) if x.parent.name not in {"style", "script"} and str(x).strip())
    report = {"output": str(args.output.resolve()), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "reference_sha256": hashlib.sha256(original).hexdigest(), "html_bytes": args.output.stat().st_size, "visible_text_chars": len(text), "chinese_characters": len(re.findall(r"[\u4e00-\u9fff]", text)), "workflow_steps": len(soup.select(".workflow-step")), "workflow_subsections": len(soup.select(".workflow-step .current-detail h4")), "tables": len(soup.select("table")), "original_images_preserved": len(new_images), "svg_figures": len(soup.select("svg")), "local_source_links_checked": len(local_links), "source_examples_checked": len(examples), "source_examples": examples, "broken_anchors": broken_anchors, "duplicate_ids": duplicate_ids, "lost_original_ids": lost_ids, "reference_unchanged": args.reference.read_bytes() == original, "new_provider_requests": 0, "local_links": local_links}
    if args.qa_output:
        args.qa_output.parent.mkdir(parents=True, exist_ok=True)
        args.qa_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"local_links", "source_examples"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
