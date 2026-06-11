# Copyright (c) 2007-2025 by the Sphinx team (see AUTHORS file)
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: BSD-2-Clause
#
# This file has been modified by Bytedance Ltd. and/or its affiliates. on 2026-05-12.
#
# Original file was released under the BSD-2-Clause license, with the full
# license text available at licenses/Sphinx-BSD-2-Clause.txt.
#
# This modified file is released under the same license.

from docutils import nodes
from sphinx.locale import _
from sphinx.util.math import get_node_equation_number
from sphinx.writers.html5 import HTML5Translator


def html_visit_displaymath(self: HTML5Translator, node: nodes.math_block) -> None:
    self.body.append(
        self.starttag(node, "span", CLASS="math notranslate nohighlight d-block")
    )
    if node.get("no-wrap", node.get("nowrap", False)):
        self.body.append(self.encode(node.astext()))
        self.body.append("</span>")
        raise nodes.SkipNode

    # necessary to e.g. set the id property correctly
    if node["number"]:
        number = get_node_equation_number(self, node)
        self.body.append(f'<span class="eqno">({number})')
        self.add_permalink_ref(node, _("Link to this equation"))
        self.body.append("</span>")
    self.body.append(self.builder.config.mathjax_display[0])
    parts = [prt for prt in node.astext().split("\n\n") if prt.strip()]
    if len(parts) > 1:  # Add alignment if there are more than 1 equation
        self.body.append(r" \begin{align}\begin{aligned}")
    for i, raw_part in enumerate(parts):
        part = self.encode(raw_part)
        if r"\\" in part:
            self.body.append(r"\begin{split}" + part + r"\end{split}")
        else:
            self.body.append(part)
        if i < len(parts) - 1:  # append new line if not the last equation
            self.body.append(r"\\")
    if len(parts) > 1:  # Add alignment if there are more than 1 equation
        self.body.append(r"\end{aligned}\end{align} ")
    self.body.append(self.builder.config.mathjax_display[1])
    self.body.append("</span>\n")
    raise nodes.SkipNode


def setup(app):
    app.setup_extension("sphinx.ext.mathjax")
    app.registry.html_block_math_renderers["mathjax"] = (html_visit_displaymath, None)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
