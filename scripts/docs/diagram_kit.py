"""Small diagram kit: one definition -> a draw.io file and a matching PNG.

GitHub cannot show .drawio files, so every diagram is saved twice: the editable .drawio and a .png rendered with
matplotlib from the same coordinates. Shapes: external entity, process, data store (Gane-Sarson), ER table,
sequence lifeline; arrows with labels, crow's-foot relationships and sequence messages.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Polygon, Rectangle  # noqa: E402

FONT = "DejaVu Sans"
MONO = "DejaVu Sans Mono"
PT = 0.72  # pixel size -> matplotlib points at 100 dpi

STYLE = {
    "entity": ("#dae8fc", "#6c8ebf"),
    "process": ("#d5e8d4", "#82b366"),
    "store": ("#fff2cc", "#d6b656"),
    "table": ("#dae8fc", "#6c8ebf"),
    "ref": ("#f5f5f5", "#999999"),
    "lifeline": ("#f5f5f5", "#666666"),
}


@dataclass
class Node:
    id: str
    kind: str
    x: float
    y: float
    w: float
    h: float
    label: str
    number: str = ""  # process number "1.0" or store id "D1"
    rows: list[tuple[str, str, str]] = field(default_factory=list)  # ER table rows: (key, name, type)

    def at(self, fx: float, fy: float) -> tuple[float, float]:
        return self.x + fx * self.w, self.y + fy * self.h


@dataclass
class Edge:
    src: str
    exit: tuple[float, float]
    dst: str
    entry: tuple[float, float]
    label: str = ""
    points: list[tuple[float, float]] = field(default_factory=list)
    kind: str = "flow"  # flow | relation
    label_at: tuple[float, float] | None = None  # explicit label position (pixels)


@dataclass
class Message:
    x1: float
    x2: float
    y: float
    label: str
    reply: bool = False


class Diagram:
    def __init__(self, name: str, width: int, height: int, title: str = "") -> None:
        self.name, self.W, self.H, self.title = name, width, height, title
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.messages: list[Message] = []

    # ---------------- building ----------------
    def node(self, *args, **kwargs) -> Node:
        n = Node(*args, **kwargs)
        self.nodes[n.id] = n
        return n

    def edge(self, *args, **kwargs) -> Edge:
        e = Edge(*args, **kwargs)
        self.edges.append(e)
        return e

    def message(self, *args, **kwargs) -> Message:
        m = Message(*args, **kwargs)
        self.messages.append(m)
        return m

    def path(self, e: Edge) -> list[tuple[float, float]]:
        return [self.nodes[e.src].at(*e.exit), *e.points, self.nodes[e.dst].at(*e.entry)]

    # ---------------- PNG ----------------
    def save_png(self, path: Path) -> None:
        fig = plt.figure(figsize=(self.W / 100, self.H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.W)
        ax.set_ylim(self.H, 0)
        ax.axis("off")
        fig.patch.set_facecolor("white")
        if self.title:
            ax.text(20, 16, self.title, fontsize=15 * PT, fontweight="bold", family=FONT, va="top")
        for n in self.nodes.values():
            getattr(self, f"_draw_{n.kind}")(ax, n)
        for e in self.edges:
            self._draw_edge(ax, e)
        for m in self.messages:
            self._draw_message(ax, m)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)

    @staticmethod
    def _text(ax, x, y, s, size=12, bold=False, ha="center", va="center", family=FONT, color="#000"):
        ax.text(x, y, s, fontsize=size * PT, fontweight="bold" if bold else "normal", ha=ha, va=va,
                family=family, color=color, linespacing=1.25, zorder=5)

    def _draw_entity(self, ax, n: Node) -> None:
        fc, ec = STYLE["entity"]
        ax.add_patch(Rectangle((n.x, n.y), n.w, n.h, fc=fc, ec=ec, lw=1.6, zorder=2))
        self._text(ax, n.x + n.w / 2, n.y + n.h / 2, n.label, 13, bold=True)

    def _draw_ref(self, ax, n: Node) -> None:
        fc, ec = STYLE["ref"]
        ax.add_patch(FancyBboxPatch((n.x, n.y), n.w, n.h, boxstyle="round,pad=0,rounding_size=12", fc=fc,
                                    ec=ec, lw=1.4, ls="--", zorder=2))
        self._text(ax, n.x + n.w / 2, n.y + n.h / 2, f"{n.number}\n{n.label}", 12, color="#444")

    def _draw_process(self, ax, n: Node) -> None:
        fc, ec = STYLE["process"]
        ax.add_patch(FancyBboxPatch((n.x, n.y), n.w, n.h, boxstyle="round,pad=0,rounding_size=14", fc=fc,
                                    ec=ec, lw=1.6, zorder=2))
        ax.plot([n.x, n.x + n.w], [n.y + 28, n.y + 28], color=ec, lw=1.2, zorder=3)
        self._text(ax, n.x + n.w / 2, n.y + 14, n.number, 12, bold=True)
        self._text(ax, n.x + n.w / 2, n.y + 28 + (n.h - 28) / 2, n.label, 12)

    def _draw_store(self, ax, n: Node) -> None:
        fc, ec = STYLE["store"]
        ax.add_patch(Rectangle((n.x, n.y), n.w, n.h, fc=fc, ec="none", zorder=2))
        ax.plot([n.x + n.w, n.x, n.x, n.x + n.w], [n.y, n.y, n.y + n.h, n.y + n.h], color=ec, lw=1.6, zorder=3)
        ax.plot([n.x + 42, n.x + 42], [n.y, n.y + n.h], color=ec, lw=1.2, zorder=3)
        self._text(ax, n.x + 21, n.y + n.h / 2, n.number, 12, bold=True)
        self._text(ax, n.x + 52, n.y + n.h / 2, n.label, 12, ha="left", family=MONO)

    def _draw_table(self, ax, n: Node) -> None:
        fc, ec = STYLE["table"]
        rows_h = len(n.rows) * 22
        ax.add_patch(Rectangle((n.x, n.y), n.w, 30, fc=fc, ec=ec, lw=1.6, zorder=2))
        ax.add_patch(Rectangle((n.x, n.y + 30), n.w, rows_h + 6, fc="white", ec=ec, lw=1.6, zorder=2))
        self._text(ax, n.x + n.w / 2, n.y + 15, n.label, 13, bold=True, family=MONO)
        for i, (key, name, typ) in enumerate(n.rows):
            y = n.y + 30 + 3 + i * 22 + 11
            self._text(ax, n.x + 8, y, key, 11, bold=True, ha="left", family=MONO, color="#b85450")
            self._text(ax, n.x + 52, y, name, 11, bold=key.startswith("PK"), ha="left", family=MONO)
            self._text(ax, n.x + n.w - 8, y, typ, 10.5, ha="right", family=MONO, color="#555")

    def _draw_lifeline(self, ax, n: Node) -> None:
        fc, ec = STYLE["lifeline"]
        ax.add_patch(Rectangle((n.x, n.y), n.w, 44, fc=fc, ec=ec, lw=1.4, zorder=2))
        self._text(ax, n.x + n.w / 2, n.y + 22, n.label, 11.5, bold=True)
        cx = n.x + n.w / 2
        ax.plot([cx, cx], [n.y + 44, n.y + n.h], color="#888", lw=1.1, ls=(0, (5, 4)), zorder=1)

    def _draw_edge(self, ax, e: Edge) -> None:
        pts = self.path(e)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="#333", lw=1.3, zorder=4, solid_joinstyle="miter")
        if e.kind == "flow":
            _arrow_head(ax, pts[-2], pts[-1])
        else:  # relation: "one" at the source, "many" at the target
            _crow_one(ax, pts[1], pts[0])
            _crow_many(ax, pts[-2], pts[-1])
        if e.label:
            lx, ly = e.label_at or _midpoint(pts)
            ax.text(lx, ly, e.label, fontsize=10.5 * PT, family=FONT, ha="center", va="center", zorder=6,
                    linespacing=1.2, bbox=dict(fc="white", ec="none", pad=1.5))

    def _draw_message(self, ax, m: Message) -> None:
        style = (0, (5, 3)) if m.reply else "-"
        if m.x1 == m.x2:  # self message
            pts = [(m.x1, m.y), (m.x1 + 40, m.y), (m.x1 + 40, m.y + 22), (m.x1 + 3, m.y + 22)]
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color="#333", lw=1.3, ls=style, zorder=4)
            _arrow_head(ax, pts[-2], pts[-1])
            self._text(ax, m.x1 + 48, m.y + 10, m.label, 10.5, ha="left")
            return
        ax.plot([m.x1, m.x2], [m.y, m.y], color="#333", lw=1.3, ls=style, zorder=4)
        _arrow_head(ax, (m.x1, m.y), (m.x2, m.y), filled=not m.reply)
        self._text(ax, (m.x1 + m.x2) / 2, m.y - 9, m.label, 10.5)

    # ---------------- draw.io ----------------
    def save_drawio(self, path: Path) -> None:
        cells: list[str] = ['<mxCell id="0"/>', '<mxCell id="1" parent="0"/>']
        if self.title:
            cells.append(_vertex("title", self.title, 20, 4, 900, 28,
                                 "text;html=1;fontSize=15;fontStyle=1;align=left;verticalAlign=middle;"))
        for n in self.nodes.values():
            cells.extend(self._drawio_node(n))
        for i, e in enumerate(self.edges):
            cells.append(self._drawio_edge(i, e))
        for i, m in enumerate(self.messages):
            cells.append(self._drawio_message(i, m))
        body = "".join(cells)
        xml = (
            '<mxfile host="drawio" type="device">'
            f'<diagram id="{self.name}" name="{html.escape(self.name)}">'
            f'<mxGraphModel dx="{self.W}" dy="{self.H}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" '
            f'arrows="1" fold="1" page="1" pageScale="1" pageWidth="{self.W}" pageHeight="{self.H}" math="0" '
            f'shadow="0"><root>{body}</root></mxGraphModel></diagram></mxfile>'
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(xml, encoding="utf-8")

    def _drawio_node(self, n: Node) -> list[str]:
        fc, ec = STYLE.get(n.kind, ("#ffffff", "#000000"))
        base = f"html=1;whiteSpace=wrap;fillColor={fc};strokeColor={ec};fontSize=12;"
        label = n.label.replace("\n", "<br>")
        if n.kind == "entity":
            return [_vertex(n.id, label, n.x, n.y, n.w, n.h, base + "rounded=0;fontStyle=1;fontSize=13;")]
        if n.kind == "ref":
            return [_vertex(n.id, f"{n.number}<br>{label}", n.x, n.y, n.w, n.h,
                            base + "rounded=1;dashed=1;fontColor=#444444;")]
        if n.kind == "process":
            return [_vertex(n.id, f"<b>{n.number}</b><hr>{label}", n.x, n.y, n.w, n.h,
                            base + "rounded=1;arcSize=12;verticalAlign=top;spacingTop=4;")]
        if n.kind == "store":
            return [
                _vertex(n.id, label, n.x, n.y, n.w, n.h,
                        base + "shape=partialRectangle;right=0;align=left;spacingLeft=52;fontFamily=Courier New;"),
                _vertex(n.id + "_no", f"<b>{n.number}</b>", n.x, n.y, 42, n.h,
                        f"html=1;shape=partialRectangle;top=0;bottom=0;left=0;fillColor=none;strokeColor={ec};"),
            ]
        if n.kind == "table":
            h = 30 + len(n.rows) * 22 + 6
            out = [_vertex(n.id, label, n.x, n.y, n.w, h,
                           base + "swimlane;fontStyle=1;startSize=30;horizontal=1;childLayout=stackLayout;"
                           "horizontalStack=0;resizeParent=1;collapsible=0;marginBottom=0;fontFamily=Courier New;"
                           "fontSize=13;swimlaneFillColor=#ffffff;")]
            for i, (key, name, typ) in enumerate(n.rows):
                value = (f'<font color="#b85450"><b>{key or "&nbsp;"}</b></font>&nbsp;&nbsp;'
                         f'{"<b>" + name + "</b>" if key.startswith("PK") else name}&nbsp;&nbsp;'
                         f'<font color="#555555">{typ}</font>')
                out.append(_vertex(f"{n.id}_r{i}", value, 0, 30 + 3 + i * 22, n.w, 22,
                                   "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;"
                                   "spacingLeft=8;fontFamily=Courier New;fontSize=11;", parent=n.id))
            return out
        if n.kind == "lifeline":
            return [_vertex(n.id, label, n.x, n.y, n.w, n.h,
                            "shape=umlLifeline;perimeter=lifelinePerimeter;whiteSpace=wrap;html=1;container=0;"
                            f"collapsible=0;size=44;fillColor={fc};strokeColor={ec};fontStyle=1;fontSize=12;")]
        raise ValueError(n.kind)

    def _drawio_edge(self, i: int, e: Edge) -> str:
        if e.kind == "flow":
            ends = "endArrow=block;endFill=1;startArrow=none;"
        else:
            ends = "endArrow=ERmany;endFill=0;startArrow=ERmandOne;startFill=0;"
        style = (f"edgeStyle=none;html=1;rounded=0;{ends}strokeColor=#333333;fontSize=11;"
                 "labelBackgroundColor=#ffffff;"
                 f"exitX={e.exit[0]};exitY={e.exit[1]};exitDx=0;exitDy=0;"
                 f"entryX={e.entry[0]};entryY={e.entry[1]};entryDx=0;entryDy=0;")
        pts = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in e.points)
        pts = f'<Array as="points">{pts}</Array>' if pts else ""
        offset = ""
        if e.label_at:  # draw.io places labels relative to the path middle; move it to the same spot
            mx, my = _midpoint(self.path(e))
            offset = f'<mxPoint x="{e.label_at[0] - mx}" y="{e.label_at[1] - my}" as="offset"/>'
        value = html.escape(e.label).replace("\n", "&lt;br&gt;")
        return (f'<mxCell id="e{i}" value="{value}" style="{style}" edge="1" parent="1" source="{e.src}" '
                f'target="{e.dst}"><mxGeometry relative="1" as="geometry">{pts}{offset}</mxGeometry></mxCell>')

    def _drawio_message(self, i: int, m: Message) -> str:
        dash = "dashed=1;endArrow=open;endFill=0;" if m.reply else "endArrow=block;endFill=1;"
        style = f"html=1;rounded=0;{dash}strokeColor=#333333;fontSize=11;verticalAlign=bottom;labelBackgroundColor=none;"
        value = html.escape(m.label)
        if m.x1 == m.x2:
            pts = (f'<Array as="points"><mxPoint x="{m.x1 + 40}" y="{m.y}"/><mxPoint x="{m.x1 + 40}" '
                   f'y="{m.y + 22}"/></Array>')
            style = style.replace("verticalAlign=bottom", "align=left;spacingLeft=48;verticalAlign=middle")
            geo = (f'<mxPoint x="{m.x1}" y="{m.y}" as="sourcePoint"/><mxPoint x="{m.x1 + 3}" y="{m.y + 22}" '
                   f'as="targetPoint"/>{pts}')
        else:
            geo = f'<mxPoint x="{m.x1}" y="{m.y}" as="sourcePoint"/><mxPoint x="{m.x2}" y="{m.y}" as="targetPoint"/>'
        return (f'<mxCell id="m{i}" value="{value}" style="{style}" edge="1" parent="1">'
                f'<mxGeometry relative="1" as="geometry">{geo}</mxGeometry></mxCell>')

    def save(self, folder: Path, stem: str) -> None:
        self.save_drawio(folder / f"{stem}.drawio")
        self.save_png(folder / f"{stem}.png")


# ---------------- helpers ----------------
def _vertex(id_, value, x, y, w, h, style, parent="1") -> str:
    value = html.escape(value)
    return (f'<mxCell id="{id_}" value="{value}" style="{style}" vertex="1" parent="{parent}">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')


def _midpoint(pts):
    import math

    lengths = [math.dist(a, b) for a, b in zip(pts, pts[1:])]
    half = sum(lengths) / 2
    for (a, b), length in zip(zip(pts, pts[1:]), lengths):
        if half <= length and length > 0:
            t = half / length
            return a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        half -= length
    return pts[-1]


def _unit(a, b):
    import math

    dx, dy = b[0] - a[0], b[1] - a[1]
    d = math.hypot(dx, dy) or 1
    return dx / d, dy / d


def _arrow_head(ax, a, b, size=11, filled=True):
    ux, uy = _unit(a, b)
    px, py = -uy, ux
    tip = b
    left = (b[0] - ux * size + px * size * 0.42, b[1] - uy * size + py * size * 0.42)
    right = (b[0] - ux * size - px * size * 0.42, b[1] - uy * size - py * size * 0.42)
    if filled:
        ax.add_patch(Polygon([tip, left, right], closed=True, fc="#333", ec="#333", zorder=5))
    else:
        ax.plot([left[0], tip[0], right[0]], [left[1], tip[1], right[1]], color="#333", lw=1.3, zorder=5)


def _crow_one(ax, toward, end):
    """Two bars across the line near `end` (mandatory one)."""
    ux, uy = _unit(toward, end)
    px, py = -uy, ux
    for d in (10, 16):
        cx, cy = end[0] - ux * d, end[1] - uy * d
        ax.plot([cx + px * 7, cx - px * 7], [cy + py * 7, cy - py * 7], color="#333", lw=1.3, zorder=5)


def _crow_many(ax, toward, end):
    """Crow's foot at `end` (many)."""
    ux, uy = _unit(toward, end)
    px, py = -uy, ux
    base = (end[0] - ux * 14, end[1] - uy * 14)
    for s in (-1, 0, 1):
        ax.plot([base[0], end[0] + px * 8 * s], [base[1], end[1] + py * 8 * s], color="#333", lw=1.3, zorder=5)
