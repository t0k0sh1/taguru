"""`_models.py`'s `__all__` versus its actual class definitions (issue
#735): `__init__.py`'s re-export surface is guarded by
`check_surface.py`, but nothing guarded `_models.py`'s own declared
surface, and 39 defined classes had silently dropped out of it. This
cross-checks the module's `__all__` against every class the file
defines, by AST, so the two can never drift again.
"""

from __future__ import annotations

import ast

from tests.unit._repo import repo_root

MODELS_PATH = repo_root() / "sdk" / "python" / "src" / "taguru" / "_models.py"


def test_models_all_names_every_defined_class() -> None:
    tree = ast.parse(MODELS_PATH.read_text())
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    declared: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            assert isinstance(node.value, ast.List)
            declared = {
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant)
            }
    assert declared, "_models.py must declare __all__"
    missing = sorted(classes - declared)
    assert not missing, f"defined but absent from __all__: {missing}"
    phantom = sorted(declared - classes)
    assert not phantom, f"in __all__ but not defined: {phantom}"
