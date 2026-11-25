"""Wireframe kit: one screen definition -> a draw.io file and a matching PNG.

Low-fidelity, grey wireframes with the five interface colours only (see README "Interface"):
ink for text, blue for actions and "in flight", green for "accepted", red for "rejected" and errors,
grey for "ended" and inactive parts.
"""

from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

FONT = "DejaVu Sans"
PT = 0.72  # pixel size -> matplotlib points at 100 dpi

INK, BLUE, GREEN, RED, GREY = "#1f2328", "#0969da", "#1a7f37", "#cf222e", "#8c959f"
PAPER, LINE = "#f6f8fa", "#d0d7de"  # neutral greys for backgrounds and borders
STATUS = {"accepted": GREEN, "rejected": RED, "in flight": BLUE, "ended": GREY}


class Screen:
    def __init__(self, name: str, title: str, width: int = 960, height: int = 640, user: str = "",
                 back: str = "") -> None:
        self.name, self.W, self.H = name, width, height
        self._png: list = []  # drawing callbacks
        self._xml: list[str] = []
        self._n = 0
        # window frame, app bar, screen title and the way back
        self.rect(0, 0, width, height, fill="#ffffff", stroke=GREY)
        self.rect(0, 0, width, 48, fill=PAPER, stroke=GREY)
        self.text(20, 0, 300, 48, "Drone Vision", 15, bold=True)
        if user:
            self.text(width - 340, 0, 200, 48, user, 12, color=INK, align="right")
            self.text(width - 120, 0, 100, 48, "Sign out", 12, color=BLUE, align="right")
        if back:
            self.text(24, 58, 200, 24, f"← {back}", 12, color=BLUE)
        self.text(24, 82, width - 48, 34, title, 20, bold=True)

    # ---------------- primitives ----------------
    def _id(self) -> str:
        self._n += 1
        return f"c{self._n}"

    def rect(self, x, y, w, h, fill="#ffffff", stroke=LINE, rounded=False, dashed=False, lw=1.2) -> None:
        def draw(ax):
            kw = dict(fc=fill, ec=stroke, lw=lw, ls="--" if dashed else "-", zorder=1)
            if rounded:
                ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=6", **kw))
            else:
                ax.add_patch(Rectangle((x, y), w, h, **kw))
        self._png.append(draw)
        style = (f"html=1;whiteSpace=wrap;fillColor={fill};strokeColor={stroke};"
                 f"{'rounded=1;arcSize=12;' if rounded else 'rounded=0;'}{'dashed=1;' if dashed else ''}")
        self._xml.append(_vertex(self._id(), "", x, y, w, h, style))

    def text(self, x, y, w, h, s, size=12, bold=False, color=INK, align="left") -> None:
        def draw(ax):
            tx = {"left": x + 2, "center": x + w / 2, "right": x + w - 2}[align]
            ax.text(tx, y + h / 2, s, fontsize=size * PT, fontweight="bold" if bold else "normal", ha=align,
                    va="center", family=FONT, color=color, linespacing=1.3, zorder=5)
        self._png.append(draw)
        style = (f"text;html=1;whiteSpace=wrap;align={align};verticalAlign=middle;fontSize={size};"
                 f"fontColor={color};{'fontStyle=1;' if bold else ''}spacingLeft=2;spacingRight=2;")
        self._xml.append(_vertex(self._id(), html.escape(s).replace("\n", "<br>"), x, y, w, h, style))

    def line(self, x1, y1, x2, y2, color=LINE, lw=1.2) -> None:
        self._png.append(lambda ax: ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=2))
        self._xml.append(
            f'<mxCell id="{self._id()}" style="endArrow=none;html=1;strokeColor={color};" edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry"><mxPoint x="{x1}" y="{y1}" as="sourcePoint"/>'
            f'<mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')

    # ---------------- widgets ----------------
    def button(self, x, y, w, label, primary=False, color=BLUE) -> None:
        self.rect(x, y, w, 36, fill=color if primary else "#ffffff", stroke=color, rounded=True)
        self.text(x, y, w, 36, label, 12.5, bold=True, color="#ffffff" if primary else color, align="center")

    def field(self, x, y, w, label, value="", kind="text", error="", hint="", readonly=False,
              required=False) -> None:
        """Labelled input: text | select | datetime | readonly. Returns nothing; 74 px tall with a message."""
        self.text(x, y, w, 20, label + (" *" if required else ""), 12, bold=True)
        stroke = RED if error else (LINE if readonly else GREY)
        self.rect(x, y + 22, w, 34, fill=PAPER if readonly else "#ffffff", stroke=stroke, rounded=True,
                  lw=1.8 if error else 1.2)
        self.text(x + 8, y + 22, w - 46, 34, value, 12, color=GREY if not value or readonly else INK)
        icon = {"select": "▾", "datetime": "\U0001F4C5"}.get(kind, "")
        if icon:
            self.text(x + w - 34, y + 22, 28, 34, icon if kind == "select" else "▦", 13, color=GREY,
                      align="center")
        if error:
            self.text(x, y + 58, w, 20, f"⚠ {error}", 11, color=RED)
        elif hint:
            self.text(x, y + 58, w, 20, hint, 11, color=GREY)

    def checkbox(self, x, y, label, checked=False, error="") -> None:
        self.rect(x, y + 3, 18, 18, fill=BLUE if checked else "#ffffff", stroke=RED if error else GREY,
                  rounded=True, lw=1.8 if error else 1.2)
        if checked:
            self.text(x, y + 3, 18, 18, "✓", 12, bold=True, color="#ffffff", align="center")
        self.text(x + 28, y, 700, 24, label, 12)
        if error:
            self.text(x + 28, y + 24, 700, 20, f"⚠ {error}", 11, color=RED)

    def chip(self, x, y, status, w=86) -> None:
        color = STATUS[status]
        self.rect(x, y, w, 22, fill="#ffffff", stroke=color, rounded=True)
        self.text(x, y, w, 22, status, 11, bold=True, color=color, align="center")

    def table(self, x, y, columns: list[tuple[str, int]], rows: list[list[str]], row_h=30,
              chip_col: int | None = None) -> None:
        total = sum(w for _, w in columns)
        self.rect(x, y, total, 32, fill=PAPER, stroke=LINE)
        cx = x
        for name, w in columns:
            self.text(cx + 8, y, w - 10, 32, name, 11.5, bold=True)
            cx += w
        for r, row in enumerate(rows):
            ry = y + 32 + r * row_h
            self.line(x, ry + row_h, x + total, ry + row_h)
            cx = x
            for c, ((_, w), value) in enumerate(zip(columns, row)):
                if c == chip_col:
                    self.chip(cx + 6, ry + (row_h - 22) / 2, value, w=min(86, w - 12))
                else:
                    self.text(cx + 8, ry, w - 10, row_h, value, 11)
                cx += w
        self.rect(x, y, total, 32 + len(rows) * row_h, fill="none", stroke=LINE)

    def image(self, x, y, w, h, caption) -> None:
        self.rect(x, y, w, h, fill=PAPER, stroke=GREY)
        self.line(x, y, x + w, y + h, color=LINE)
        self.line(x, y + h, x + w, y, color=LINE)
        self.rect(x + w / 2 - 150, y + h / 2 - 18, 300, 36, fill="#ffffff", stroke=LINE, rounded=True)
        self.text(x + w / 2 - 150, y + h / 2 - 18, 300, 36, caption, 12, color=GREY, align="center")

    def banner(self, x, y, w, h, title, body, color=RED) -> None:
        self.rect(x, y, w, h, fill="#ffffff", stroke=color, rounded=True, lw=2)
        self.rect(x, y, 8, h, fill=color, stroke=color)
        self.text(x + 22, y + 8, w - 40, 26, title, 13.5, bold=True, color=color)
        self.text(x + 22, y + 36, w - 40, h - 44, body, 12, color=INK)

    # ---------------- output ----------------
    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(self.W / 100, self.H / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.W)
        ax.set_ylim(self.H, 0)
        ax.axis("off")
        for draw in self._png:
            draw(ax)
        fig.savefig(folder / f"{self.name}.png", dpi=200, facecolor="white")
        plt.close(fig)
        body = '<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(self._xml)
        (folder / f"{self.name}.drawio").write_text(
            '<mxfile host="drawio" type="device">'
            f'<diagram id="{self.name}" name="{self.name}"><mxGraphModel dx="{self.W}" dy="{self.H}" grid="1" '
            f'gridSize="10" guides="1" page="1" pageWidth="{self.W}" pageHeight="{self.H}">'
            f"<root>{body}</root></mxGraphModel></diagram></mxfile>", encoding="utf-8")


def _vertex(id_, value, x, y, w, h, style) -> str:
    return (f'<mxCell id="{id_}" value="{html.escape(value)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
