# tifffile/setup.py

"""Tifffile package Setuptools script."""

import re
import sys

from setuptools import setup

buildnumber = ''


def search(pattern: str, string: str, flags: int = 0) -> str:
    """Return first match of pattern in string."""
    match = re.search(pattern, string, flags)
    if match is None:
        raise ValueError(f'{pattern!r} not found')
    return match.groups()[0]


def fix_docstring_examples(docstring: str) -> str:
    """Return docstring with examples fixed for GitHub."""
    start = True
    indent = False
    lines = ['..', '  This file is generated by setup.py', '']
    for line in docstring.splitlines():
        if not line.strip():
            start = True
            indent = False
        if line.startswith('>>> '):
            indent = True
            if start:
                lines.extend(['.. code-block:: python', ''])
                start = False
        lines.append(('    ' if indent else '') + line)
    return '\n'.join(lines)


with open('tifffile/tifffile.py', encoding='utf-8') as fh:
    code = fh.read().replace('\r\n', '\n').replace('\r', '\n')

version = search(r"__version__ = '(.*?)'", code).replace('.x.x', '.dev0')
version += ('.' + buildnumber) if buildnumber else ''

description = search(r'"""(.*)\.(?:\r\n|\r|\n)', code)

readme = search(
    r'(?:\r\n|\r|\n){2}r"""(.*)"""(?:\r\n|\r|\n){2}from __future__',
    code,
    re.MULTILINE | re.DOTALL,
)
readme = '\n'.join(
    [description, '=' * len(description)] + readme.splitlines()[1:]
)

if 'sdist' in sys.argv:
    # update README, LICENSE, and CHANGES files

    with open('README.rst', 'w', encoding='utf-8') as fh:
        fh.write(fix_docstring_examples(readme))

    license = search(
        r'(# Copyright.*?(?:\r\n|\r|\n))(?:\r\n|\r|\n)+r""',
        code,
        re.MULTILINE | re.DOTALL,
    )
    license = license.replace('# ', '').replace('#', '')

    with open('LICENSE', 'w', encoding='utf-8') as fh:
        fh.write('BSD 3-Clause License\n\n')
        fh.write(license)

    revisions = search(
        r'(?:\r\n|\r|\n){2}(Revisions.*)- …',
        readme,
        re.MULTILINE | re.DOTALL,
    ).strip()

    with open('CHANGES.rst', encoding='utf-8') as fh:
        old = fh.read()

    old = old.split(revisions.splitlines()[-1])[-1]
    with open('CHANGES.rst', 'w', encoding='utf-8') as fh:
        fh.write(revisions.strip())
        fh.write(old)

setup(
    version=version,
    description=description,
    long_description=readme,
    long_description_content_type='text/x-rst',
)
