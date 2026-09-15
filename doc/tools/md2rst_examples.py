#!/usr/bin/env python
"""
One-shot converter: jupytext py:percent examples -> sphinx-gallery format.

sphinx-gallery requires a module-level docstring and interprets text blocks as
reST, while the examples were written as markdown cells. This performs the
mechanical part of the conversion:

  * drops the '#!/usr/bin/env python' shebang
  * turns the first '# %% [markdown]' cell into the module docstring, with the
    markdown h1 becoming a reST title
  * rewrites markdown headings in the remaining text cells as reST headings
  * rewrites single-backtick code spans as double-backtick reST literals

'**bold**' and '*italic*' are left alone -- they are valid reST already.

Usage:
    python md2rst_examples.py [--dry-run] FILE [FILE ...]
"""

import argparse
import re
import sys

# reST has no fixed heading hierarchy; this is the convention used here.
HEADING_CHARS = {1: '=', 2: '-', 3: '~', 4: '^', 5: '"'}

RE_CELL = re.compile(r'^#\s*%%(.*)$')
RE_HEADING = re.compile(r'^(#{1,4})\s+(.*)$')
# Deliberately liberal: '1a.' appears alongside '1.' in some examples, and
# treating it as a list item keeps such a list contiguous instead of splitting
# it into stray paragraphs.
RE_LIST_ITEM = re.compile(r'^\s*([-*+]|\d+[a-z]?[.)])\s+')
# A single-backtick span, not part of a double-backtick span.
RE_CODE_SPAN = re.compile(r'(?<!`)`([^`\n]+)`(?!`)')
RE_FENCE = re.compile(r'^\s*```\s*(\w*)\s*$')
RE_TABLE_ROW = re.compile(r'^\s*\|(.+)\|\s*$')
RE_TABLE_SEP = re.compile(r'^\s*\|[\s:|-]+\|\s*$')
# Markdown code blocks in these examples are indented by anything from 2 to 8
# spaces. reST needs an explicit '::' marker in front of any of them.
RE_INDENTED = re.compile(r'^\s{2,}\S')


def inline_md_to_rst(text):
    """Convert inline markdown markup to reST within a single line."""
    return RE_CODE_SPAN.sub(r'``\1``', text)


def decomment(lines):
    """Strip the leading '# ' from a block of comment lines."""
    out = []
    for line in lines:
        if line.startswith('# '):
            out.append(line[2:])
        elif line.strip() == '#':
            out.append('')
        else:
            out.append(line.lstrip('#').lstrip())
    return out


def dedent_stray_indents(lines):
    """Remove a single leading space.

    Markdown ignores an indent of one space; reST turns it into a block quote.
    Anything indented further is left alone -- that is a deliberate code block,
    handled by fix_literal_blocks.
    """
    return [line[1:] if re.match(r'^ \S', line) else line for line in lines]


def recomment(lines):
    """Prefix a block of text lines with '# ' to make them comments again."""
    return ['# %s' % line if line.strip() else '#' for line in lines]


def convert_text(lines, title_only_first=False, base_level=2):
    """Convert de-commented markdown lines to reST lines.

    ``base_level`` is the reST level the shallowest body heading maps to. The
    docstring supplies the document title (level 1), so body headings must start
    at level 2 or Sphinx reports 'Inconsistent title style'. Authors are not
    consistent about whether they used '#' or '##' for body sections, so the
    caller measures the shallowest level actually used and everything is shifted
    relative to that.
    """
    out = []
    first_heading_done = False
    for line in lines:
        match = RE_HEADING.match(line)
        if match:
            level = len(match.group(1))
            text = inline_md_to_rst(match.group(2).strip())
            # The docstring's first heading is the gallery title.
            if title_only_first and not first_heading_done:
                level = 1
                first_heading_done = True
            else:
                level = min(level - base_level + 2, max(HEADING_CHARS))
            if out and out[-1].strip():
                out.append('')
            out.append(text)
            out.append(HEADING_CHARS[level] * max(len(text), 3))
            out.append('')
            continue
        out.append(inline_md_to_rst(line))
    return out


def fix_literal_blocks(lines):
    """Introduce indented markdown blocks with a reST literal-block marker.

    Markdown treats a 4-space indent as preformatted text; reST treats it as a
    block quote unless the preceding paragraph ends in '::'.
    """
    out = []
    i = 0
    while i < len(lines):
        if not RE_INDENTED.match(lines[i]):
            out.append(lines[i])
            i += 1
            continue
        j = len(out) - 1
        while j >= 0 and not out[j].strip():
            j -= 1
        # An indented run under a list item is that item's continuation text,
        # which reST already understands -- turning it into a literal block
        # would silently reformat prose as code.
        if j >= 0 and RE_LIST_ITEM.match(out[j]):
            while i < len(lines) and (RE_INDENTED.match(lines[i]) or not lines[i].strip()):
                out.append(lines[i])
                i += 1
            continue
        if j >= 0:
            intro = out[j].rstrip()
            if not intro.endswith('::'):
                if intro.endswith(':'):
                    out[j] = intro + ':'
                else:
                    out.append('')
                    out.append('::')
                    out.append('')
        while i < len(lines) and (RE_INDENTED.match(lines[i]) or not lines[i].strip()):
            out.append(lines[i])
            i += 1
    return out


def convert_fences(lines):
    """Convert ``` fenced code blocks to reST code-block directives."""
    out = []
    i = 0
    while i < len(lines):
        match = RE_FENCE.match(lines[i])
        if not match:
            out.append(lines[i])
            i += 1
            continue
        lang = match.group(1)
        i += 1
        body = []
        while i < len(lines) and not RE_FENCE.match(lines[i]):
            body.append(lines[i])
            i += 1
        i += 1  # closing fence
        if out and out[-1].strip():
            out.append('')
        out.append('.. code-block:: %s' % lang if lang else '::')
        out.append('')
        out.extend('   %s' % line if line.strip() else '' for line in body)
        out.append('')
    return out


def split_table_row(line):
    inner = line.strip()
    if inner.startswith('|'):
        inner = inner[1:]
    if inner.endswith('|'):
        inner = inner[:-1]
    return [cell.strip() for cell in inner.split('|')]


def convert_tables(lines):
    """Convert markdown pipe tables to reST list-table directives."""
    out = []
    i = 0
    while i < len(lines):
        is_table = (RE_TABLE_ROW.match(lines[i])
                    and i + 1 < len(lines)
                    and RE_TABLE_SEP.match(lines[i + 1]))
        if not is_table:
            out.append(lines[i])
            i += 1
            continue
        rows = [split_table_row(lines[i])]
        i += 2  # header plus separator
        while i < len(lines) and RE_TABLE_ROW.match(lines[i]):
            rows.append(split_table_row(lines[i]))
            i += 1
        if out and out[-1].strip():
            out.append('')
        out.append('.. list-table::')
        out.append('   :header-rows: 1')
        out.append('')
        for row in rows:
            for n, cell in enumerate(row):
                out.append('%s%s' % ('   * - ' if n == 0 else '     - ', cell))
        out.append('')
    return out


def fix_list_spacing(lines):
    """reST needs a blank line before and after a list block."""
    out = []
    for line in lines:
        is_item = bool(RE_LIST_ITEM.match(line))
        prev = out[-1] if out else ''
        prev_is_item = bool(RE_LIST_ITEM.match(prev))
        indent_changed = (is_item and prev_is_item
                          and len(line) - len(line.lstrip())
                          != len(prev) - len(prev.lstrip()))
        if is_item and prev.strip() and (not prev_is_item or indent_changed):
            out.append('')
        elif prev_is_item and line.strip() and not is_item and not line.startswith(' '):
            out.append('')
        out.append(line)
    return out


def strip_trailing_blanks(lines):
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def collapse_blanks(lines):
    """Collapse runs of blank lines down to a single blank line."""
    out = []
    for line in lines:
        if not line.strip() and out and not out[-1].strip():
            continue
        out.append(line)
    return out


def split_cells(lines):
    """Split source lines into (kind, marker, body) cells.

    kind is 'markdown' or 'code'; marker is the '# %%' line (or None for any
    leading content before the first marker).
    """
    cells = []
    marker = None
    kind = 'code'
    body = []
    for line in lines:
        match = RE_CELL.match(line)
        if match:
            if marker is not None or body:
                cells.append((kind, marker, body))
            marker = line
            kind = 'markdown' if 'markdown' in match.group(1) else 'code'
            body = []
        else:
            body.append(line)
    if marker is not None or body:
        cells.append((kind, marker, body))
    return cells


def take_comment_block(body):
    """Split a markdown cell body into (comment lines, trailing other lines)."""
    comment, rest = [], []
    for i, line in enumerate(body):
        if line.startswith('#') or (not line.strip() and not rest):
            if line.startswith('#'):
                comment.append(line)
            else:
                rest = body[i:]
                break
        else:
            rest = body[i:]
            break
    return comment, rest


def prepare(comment_lines):
    """De-comment a markdown cell and rewrite its block-level markup as reST.

    Order matters: literal blocks and list spacing are decided while the text is
    still markdown, because the fence and table conversions below introduce
    indented lines of their own that those passes would otherwise mangle.
    """
    lines = decomment(comment_lines)
    lines = dedent_stray_indents(lines)
    lines = fix_literal_blocks(lines)
    lines = fix_list_spacing(lines)
    lines = convert_fences(lines)
    lines = convert_tables(lines)
    return lines


def convert_file(path, dry_run=False):
    with open(path, encoding='utf-8') as f:
        source = f.read()

    if source.lstrip().startswith('"""'):
        print('  SKIP (already has a docstring): %s' % path)
        return False

    lines = source.split('\n')
    if lines and lines[0].startswith('#!'):
        lines = lines[1:]

    cells = split_cells(lines)
    if not cells:
        print('  SKIP (no cells found): %s' % path)
        return False

    # The shallowest heading level used outside the docstring, so body headings
    # can be shifted to start at level 2 regardless of the author's choice.
    body_levels = []
    seen_first_markdown = False
    for kind, _marker, body in cells:
        if kind != 'markdown':
            continue
        if not seen_first_markdown:
            seen_first_markdown = True
            continue
        for line in decomment(take_comment_block(body)[0]):
            match = RE_HEADING.match(line)
            if match:
                body_levels.append(len(match.group(1)))
    base_level = min(body_levels) if body_levels else 2

    out = []
    docstring_done = False
    for kind, marker, body in cells:
        if kind == 'markdown' and not docstring_done:
            comment, rest = take_comment_block(body)
            text = convert_text(prepare(comment), title_only_first=True)
            text = strip_trailing_blanks(collapse_blanks(text))
            if '"""' in '\n'.join(text):
                print('  WARN  embedded triple quote, needs hand fixing: %s' % path)
            out.append('"""')
            out.extend(text)
            out.append('"""')
            out.extend(strip_trailing_blanks(rest))
            docstring_done = True
        elif kind == 'markdown':
            comment, rest = take_comment_block(body)
            text = convert_text(prepare(comment), base_level=base_level)
            text = strip_trailing_blanks(collapse_blanks(text))
            out.append('# %%')
            out.extend(recomment(text))
            out.extend(rest)
        else:
            if marker is not None:
                out.append(marker)
            out.extend(body)

    result = '\n'.join(out)
    if not result.endswith('\n'):
        result += '\n'

    if dry_run:
        print('  would convert: %s' % path)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(result)
        print('  converted: %s' % path)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files', nargs='+')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    n = 0
    for path in args.files:
        if convert_file(path, dry_run=args.dry_run):
            n += 1
    print('%d file(s) %s' % (n, 'to convert' if args.dry_run else 'converted'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
