#!/usr/bin/env python3
"""Validate the exact file boundary for CompassLM's public release."""

from __future__ import annotations

import argparse
import ast
import json
import re
import shlex
from pathlib import Path, PurePosixPath
from typing import Sequence


DEFAULT_MAX_BYTES = 90 * 1024 * 1024
MANIFEST_BASENAME = "PUBLIC_RELEASE_MANIFEST.txt"

FORBIDDEN_COMPONENTS = {
    ".agents",
    ".audit",
    ".cache",
    ".codex",
    ".git",
    ".idea",
    ".lumin",
    ".mypy_cache",
    ".pytest_cache",
    ".roo",
    ".ruff_cache",
    ".superpowers",
    ".tools",
    ".venv",
    ".vscode",
    ".worktrees",
    "__pycache__",
    "archive",
    "build",
    "cache",
    "compassvenv",
    "data",
    "dist",
    "evaluation_results",
    "fin_result",
    "logs",
    "models",
    "node_modules",
    "offline_assets",
    "offline_bundle",
    "offline_packages",
    "packages",
    "results",
    "runtime",
    "tools",
    "uploads",
    "venv",
    "wheels",
}

FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".db",
    ".dll",
    ".dylib",
    ".exe",
    ".gguf",
    ".jsonl",
    ".key",
    ".log",
    ".onnx",
    ".otf",
    ".p12",
    ".pem",
    ".pfx",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".sqlite",
    ".sqlite3",
    ".tar",
    ".tgz",
    ".ttf",
    ".whl",
    ".woff",
    ".woff2",
    ".zip",
}

FORBIDDEN_EXACT_BASENAMES = {
    ".ds_store",
    "desktop.ini",
    "merges.txt",
    "rpc-server",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "usage.md",
    "vocab.json",
    "workplan.md",
    "workflow.md",
    "workflow.xml",
}

KNOWN_PLACEHOLDER_VALUES = {
    "***",
    "replace-with-strong-secret",
    "change-me",
    "example",
    "placeholder",
}

ALLOWED_TEXT_SUFFIXES = {
    ".cfg",
    ".css",
    ".example",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

ALLOWED_TEXT_BASENAMES = {
    ".gitignore",
    "dockerfile",
    "license",
    "public_release_manifest.txt",
    "requirements.txt",
}

LOCAL_PATH_PATTERNS = (
    re.compile(r"/home/[A-Za-z0-9._-]+(?:/|$)"),
    re.compile(r"/Users/[A-Za-z0-9._-]+(?:/|$)"),
    re.compile(r"/mnt/[A-Za-z]/Users/[A-Za-z0-9._-]+(?:/|$)", re.IGNORECASE),
    re.compile(r"[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9._-]+(?:[\\/]|$)", re.IGNORECASE),
)

PRIVATE_IPV4_BODY = (
    r"(?:10(?:\.[0-9]{1,3}){3}"
    r"|192\.168(?:\.[0-9]{1,3}){2}"
    r"|172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2})"
)
PRIVATE_IPV4_PATTERN = re.compile(r"(?<![0-9])" + PRIVATE_IPV4_BODY + r"(?![0-9])")
PRIVATE_HOSTNAME_PATTERN = re.compile(
    r"\b[A-Za-z0-9.-]+\.(?:internal|lan|localdomain)\b",
    re.IGNORECASE,
)
PACKAGE_REQUIREMENT_LINE_PATTERN = re.compile(
    r"(?P<quote>['\"]?)[a-z0-9][a-z0-9._-]*"
    r"(?:\[[a-z0-9][a-z0-9._-]*(?:\s*,\s*[a-z0-9][a-z0-9._-]*)*\])?"
    r"\s*(?:==|~=|>=|<=|!=|>|<)\s*"
    r"(?:[0-9]+!)?(?P<version>[0-9]+(?:\.[0-9]+){3})"
    r"(?:[._+-][A-Za-z0-9]+)*(?P=quote)"
)
VERSION_ASSIGNMENT_LINE_PATTERN = re.compile(
    r"(?:export\s+)?[A-Z_][A-Z0-9_]*VERSION\s*=\s*"
    r"(?P<quote>['\"]?)(?:[0-9]+!)?(?P<version>[0-9]+(?:\.[0-9]+){3})"
    r"(?:[._+-][A-Za-z0-9]+)*(?P=quote)"
)
WHEEL_FILENAME_LINE_PATTERN = re.compile(
    r"(?P<quote>['\"]?)[A-Za-z0-9][A-Za-z0-9_.]*-"
    r"(?:[0-9]+!)?(?P<version>[0-9]+(?:\.[0-9]+){3})"
    r"(?:[._+][A-Za-z0-9]+)*-[A-Za-z0-9][A-Za-z0-9.]*-"
    r"[A-Za-z0-9][A-Za-z0-9.]*-[A-Za-z0-9][A-Za-z0-9_.]*\.whl(?P=quote)",
    re.IGNORECASE,
)
SETUP_WHEEL_GLOB_LINE_PATTERN = re.compile(
    r'require_offline_artifact\s+"[^"\r\n]*"\s+"[A-Za-z0-9][A-Za-z0-9_.]*-'
    r'(?P<version>[0-9]+(?:\.[0-9]+){3})-\*\.whl"\s+"[^"\r\n]*"'
)
ASSET_CHECK_WHEEL_GLOB_LINE_PATTERN = re.compile(
    r'ok_or_missing_glob\s+"[^"\r\n]*[A-Za-z0-9][A-Za-z0-9_.]*-'
    r'(?P<version>[0-9]+(?:\.[0-9]+){3})-\*\.whl"\s+"[^"\r\n]*"'
)
TYPED_VERSION_LINE_PATTERNS = (
    PACKAGE_REQUIREMENT_LINE_PATTERN,
    VERSION_ASSIGNMENT_LINE_PATTERN,
    WHEEL_FILENAME_LINE_PATTERN,
    SETUP_WHEEL_GLOB_LINE_PATTERN,
    ASSET_CHECK_WHEEL_GLOB_LINE_PATTERN,
)
NETWORK_ASSIGNMENT_PATTERN = re.compile(
    r"^\s*(?:export\s+)?(?P<key>[A-Z_][A-Z0-9_]*)\s*=\s*(?P<value>.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
NETWORK_KEY_PATTERN = re.compile(
    r"(?:^|_)(?:HOST|URL|IP|ADDR|ADDRESS)(?:_|$)",
    re.IGNORECASE,
)

NON_PYTHON_CREDENTIAL_ASSIGNMENT = re.compile(
    r"^\s*(?:(?:export|local|readonly|declare|typeset)(?:\s+(?:-[A-Za-z]+|--))*\s+)*"
    r"(?P<quote>['\"]?)(?P<variable>[A-Za-z_][A-Za-z0-9_-]*)(?P=quote)"
    r"\s*(?P<operator>\+?=)\s*(?P<value>.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
STRUCTURED_CREDENTIAL_ASSIGNMENT = re.compile(
    r"^\s*(?P<quote>['\"]?)(?P<variable>[A-Za-z_][A-Za-z0-9_-]*)(?P=quote)"
    r"\s*:\s*(?P<value>.*?)\s*$",
    re.MULTILINE,
)
FLOW_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?:[{,]\s*)(?P<quote>['\"]?)(?P<variable>[A-Za-z_][A-Za-z0-9_-]*)(?P=quote)"
    r"\s*:\s*(?P<value>[^,}]+)",
)

LOCAL_USERNAME_PATTERN = re.compile(r"\b" + "ae" + "lag" + r"\b", re.IGNORECASE)
PRIVATE_KEY_HEADER = re.compile(
    r"-{5}BEGIN (?:[A-Z0-9]+ )*" + "PRIVATE " + "KEY" + r"-{5}",
    re.IGNORECASE,
)
PLAIN_ENV_REFERENCE = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")
SHELL_DEFAULT_REFERENCE = re.compile(
    r"^\$\{[A-Za-z_][A-Za-z0-9_]*:-(?P<default>.*)\}$"
)
YOUR_PLACEHOLDER = re.compile(r"^your-[A-Za-z0-9][A-Za-z0-9._-]*$", re.IGNORECASE)
ENVIRONMENT_KEY = re.compile(r"^[A-Z_][A-Z0-9_]*$")
LOOKUP_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
LOOKUP_ARGUMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
SENSITIVE_LOOKUP_KEYS = {"api_key", "apikey", "key", "password", "secret", "token"}
SHELL_ASSIGNMENT_WORD = re.compile(
    r"(?P<variable>[A-Za-z_][A-Za-z0-9_-]*)(?P<operator>\+?=)(?P<value>.*)",
)
SHELL_PARAMETER_MUTATION = re.compile(
    r"\$\{(?P<variable>[A-Za-z_][A-Za-z0-9_]*)(?P<operator>:=|=)(?P<value>[^}]*)\}"
)
SHELL_HEREDOC_START = re.compile(
    r"(?<!<)<<(?P<strip_tabs>-)?(?!<)\s*(?:"
    r"\$'(?P<ansi>[^'\r\n]+)'|\$\"(?P<localized>[^\"\r\n]+)\"|"
    r"'(?P<single>[^'\r\n]+)'|\"(?P<double>[^\"\r\n]+)\"|"
    r"\\(?P<escaped>[A-Za-z_][A-Za-z0-9_]*)|(?P<bare>[A-Za-z_][A-Za-z0-9_]*))"
)
SAFE_CREDENTIAL_CALLS = {
    "_hash_session_token",
    "_normalize_query_term",
    "auth_store.create_session",
    "resolve_api_key",
    "secrets.token_urlsafe",
    "store.create_session",
}


def read_manifest(path: Path) -> list[str]:
    """Return manifest lines decoded strictly as UTF-8."""

    return path.read_text(encoding="utf-8").splitlines()


def _is_normalized_relative_path(entry: str) -> bool:
    if (
        not entry
        or entry != entry.strip()
        or "\\" in entry
        or any(character in entry for character in "*?[")
    ):
        return False
    path = PurePosixPath(entry)
    return (
        not path.is_absolute()
        and "." not in path.parts
        and ".." not in path.parts
        and path.as_posix() == entry
    )


def _path_policy_issues(entry: str) -> list[str]:
    issues: list[str] = []
    path = PurePosixPath(entry)
    basename = path.name
    lower_name = basename.lower()
    lower_parts = [part.lower() for part in path.parts[:-1]]

    forbidden = sorted(set(lower_parts) & FORBIDDEN_COMPONENTS)
    if forbidden:
        issues.append(f"{entry}: forbidden component: {', '.join(forbidden)}")

    is_env_example = lower_name.endswith(".env.example")
    is_live_env = (
        lower_name == ".env"
        or lower_name.endswith(".env")
        or ".env." in lower_name
    )
    if is_live_env and not is_env_example:
        issues.append(f"{entry}: live environment file is forbidden")

    if (
        lower_name in FORBIDDEN_EXACT_BASENAMES
        or lower_name.startswith("transfer_")
        or lower_name.endswith(".index.json")
        or lower_name.endswith(".tar.gz")
        or lower_name.endswith(".so")
        or ".so." in lower_name
        or any(lower_name.endswith(suffix) for suffix in FORBIDDEN_SUFFIXES)
        or ("." not in lower_name and lower_name.startswith("llama-"))
    ):
        issues.append(f"{entry}: forbidden filename or format")

    if lower_name.endswith("~") or lower_name.startswith(".#") or lower_name.endswith((".swp", ".swo", ".tmp")):
        issues.append(f"{entry}: editor temporary file is forbidden")

    if lower_name not in ALLOWED_TEXT_BASENAMES and path.suffix.lower() not in ALLOWED_TEXT_SUFFIXES:
        issues.append(f"{entry}: unsupported public text type")

    return issues


def _is_static_placeholder_credential(value: str) -> bool:
    if not value:
        return True
    if PLAIN_ENV_REFERENCE.fullmatch(value):
        return True
    shell_default = SHELL_DEFAULT_REFERENCE.fullmatch(value)
    if shell_default:
        default = shell_default.group("default")
        return not default or _is_static_placeholder_credential(default)
    lowered = value.lower()
    return lowered in KNOWN_PLACEHOLDER_VALUES or YOUR_PLACEHOLDER.fullmatch(value) is not None


def _credential_name_parts(name: str) -> list[str]:
    """Normalize env/config spellings without treating generic ``key`` as a secret."""

    camel_split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel_split)
    return [part.lower() for part in re.findall(r"[A-Za-z0-9]+", camel_split)]


def _is_credential_name(name: str) -> bool:
    parts = _credential_name_parts(name)
    if not parts:
        return False
    joined = "".join(parts)
    if joined == "apikey" or joined.endswith("apikey"):
        return True
    if any(part in {"password", "secret"} for part in parts):
        return True
    if any(left == "api" and right == "key" for left, right in zip(parts, parts[1:])):
        return True
    noncredential_suffixes = {"budget", "configured", "count", "est", "pattern"}
    return any(
        part == "token"
        and (index + 1 == len(parts) or parts[index + 1] not in noncredential_suffixes)
        for index, part in enumerate(parts)
    )


def _static_string(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        pieces: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                pieces.append(value.value)
                continue
            if isinstance(value, ast.FormattedValue):
                rendered = _static_string(value.value)
                if rendered is not None:
                    pieces.append(rendered)
                    continue
            return None
        return "".join(pieces)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string(node.left)
        right = _static_string(node.right)
        if left is not None and right is not None:
            return left + right
    return None


def _qualified_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and not node.id.startswith("__"):
        return node.id
    if isinstance(node, ast.Attribute) and not node.attr.startswith("__"):
        parent = _qualified_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
    return None


def _is_safe_default_node(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _is_static_placeholder_credential(node.value)
    if isinstance(node, ast.Call):
        return _is_safe_environment_getter(node)
    return False


def _is_safe_environment_getter(node: ast.Call) -> bool:
    qualified = _qualified_name(node.func)
    if qualified not in {"os.getenv", "os.environ.get"} or node.keywords:
        return False
    if not 1 <= len(node.args) <= 2:
        return False
    key = node.args[0]
    if not (
        isinstance(key, ast.Constant)
        and isinstance(key.value, str)
        and ENVIRONMENT_KEY.fullmatch(key.value)
    ):
        return False
    return len(node.args) == 1 or _is_safe_default_node(node.args[1])


def _is_safe_lookup_key(node: ast.AST, *, required_mapping: bool = False) -> bool:
    if isinstance(node, ast.Slice) and not required_mapping:
        return all(
            bound is None or _is_safe_call_argument(bound)
            for bound in (node.lower, node.upper, node.step)
        )
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return _is_safe_credential_expression(node)
    if required_mapping:
        return ENVIRONMENT_KEY.fullmatch(node.value) is not None
    return (
        LOOKUP_KEY.fullmatch(node.value) is not None
        and node.value.lower() not in SENSITIVE_LOOKUP_KEYS
    )


def _is_safe_call_argument(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return (isinstance(node.value, str) and not node.value) or type(node.value) in {
            int,
            float,
        }
    return _is_safe_credential_expression(node)


def _is_safe_credential_call(node: ast.Call) -> bool:
    qualified = _qualified_name(node.func)
    if qualified in {"os.getenv", "os.environ.get"}:
        return _is_safe_environment_getter(node)

    if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
        if node.keywords or not 1 <= len(node.args) <= 2:
            return False
        if _qualified_name(node.func.value) is None:
            return False
        lookup = node.args[0]
        if not (
            (
                isinstance(lookup, ast.Constant)
                and isinstance(lookup.value, str)
                and LOOKUP_ARGUMENT_KEY.fullmatch(lookup.value)
            )
            or (isinstance(lookup, ast.Name) and not lookup.id.startswith("__"))
        ):
            return False
        return len(node.args) == 1 or _is_safe_default_node(node.args[1])

    if qualified == "str":
        return (
            len(node.args) == 1
            and not node.keywords
            and _is_safe_credential_expression(node.args[0])
        )

    if isinstance(node.func, ast.Attribute) and node.func.attr in {"lower", "strip"}:
        return (
            not node.args
            and not node.keywords
            and _is_safe_credential_expression(node.func.value)
        )
    if isinstance(node.func, ast.Attribute) and node.func.attr == "group":
        return (
            _qualified_name(node.func.value) is not None
            and all(_is_safe_call_argument(argument) for argument in node.args)
            and not node.keywords
        )

    if qualified not in SAFE_CREDENTIAL_CALLS:
        return False
    if not all(_is_safe_call_argument(argument) for argument in node.args):
        return False
    return all(
        keyword.arg is not None and _is_safe_call_argument(keyword.value)
        for keyword in node.keywords
    )

def _is_safe_credential_expression(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return not node.id.startswith("__")
    if isinstance(node, ast.Attribute):
        return _qualified_name(node) is not None
    if isinstance(node, ast.Constant):
        return isinstance(node.value, str) and not node.value
    if isinstance(node, ast.BoolOp):
        return all(_is_safe_credential_expression(value) for value in node.values)
    if isinstance(node, ast.IfExp):
        return all(
            _is_safe_credential_expression(value)
            for value in (node.test, node.body, node.orelse)
        )
    if isinstance(node, ast.Call):
        return _is_safe_credential_call(node)
    if isinstance(node, ast.Subscript):
        mapping_name = _qualified_name(node.value)
        if mapping_name is None:
            return False
        return _is_safe_lookup_key(node.slice, required_mapping=mapping_name == "required")
    return False


def _is_safe_non_python_credential_assignment(variable: str, value: str) -> bool:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        literal = value[1:-1]
        return _is_static_placeholder_credential(literal) or (
            variable.upper() == "STDCXXFS_LINK_TOKEN" and literal == "-lstdc++fs"
        )
    if _is_static_placeholder_credential(value):
        return True
    return variable.upper() == "STDCXXFS_LINK_TOKEN" and value == "-lstdc++fs"


def _is_safe_python_credential(variable: str, value: ast.AST) -> bool:
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return _is_static_placeholder_credential(value.value) or (
            variable.upper() == "STDCXXFS_LINK_TOKEN" and value.value == "-lstdc++fs"
        )
    return _is_safe_credential_expression(value)


def _credential_target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name) and _is_credential_name(target.id):
        return [target.id]
    if isinstance(target, ast.Attribute) and _is_credential_name(target.attr):
        return [target.attr]
    if isinstance(target, ast.Subscript):
        key = _static_string(target.slice)
        if key is not None and _is_credential_name(key):
            return [key]
    if isinstance(target, (ast.List, ast.Tuple)):
        names: list[str] = []
        for element in target.elts:
            names.extend(_credential_target_names(element))
        return names
    return []


def _python_argument_defaults(arguments: ast.arguments) -> list[tuple[str, ast.AST]]:
    defaults: list[tuple[str, ast.AST]] = []
    positional = [*arguments.posonlyargs, *arguments.args]
    for argument, default in zip(positional[-len(arguments.defaults) :], arguments.defaults):
        if _is_credential_name(argument.arg):
            defaults.append((argument.arg, default))
    for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
        if default is not None and _is_credential_name(argument.arg):
            defaults.append((argument.arg, default))
    return defaults


def _sequence_nodes(node: ast.AST) -> list[ast.AST] | None:
    if isinstance(node, (ast.List, ast.Tuple)):
        return list(node.elts)
    return None


def _mapping_items(node: ast.AST) -> list[tuple[str, ast.AST]]:
    items: list[tuple[str, ast.AST]] = []
    if isinstance(node, ast.Dict):
        for key_node, value in zip(node.keys, node.values):
            if key_node is None:
                items.extend(_mapping_items(value))
                continue
            key = _static_string(key_node)
            if key is not None:
                items.append((key, value))
        return items
    if isinstance(node, (ast.List, ast.Tuple)):
        for element in node.elts:
            pair = _sequence_nodes(element)
            if pair is None or len(pair) != 2:
                continue
            key = _static_string(pair[0])
            if key is not None:
                items.append((key, pair[1]))
        return items
    if isinstance(node, ast.Call) and _qualified_name(node.func) == "zip" and len(node.args) >= 2:
        keys = _sequence_nodes(node.args[0])
        values = _sequence_nodes(node.args[1])
        if keys is None or values is None:
            return items
        for key_node, value in zip(keys, values):
            key = _static_string(key_node)
            if key is not None:
                items.append((key, value))
        return items
    if isinstance(node, ast.Call) and _qualified_name(node.func) == "dict":
        for argument in node.args:
            items.extend(_mapping_items(argument))
        for keyword in node.keywords:
            if keyword.arg is None:
                items.extend(_mapping_items(keyword.value))
            else:
                items.append((keyword.arg, keyword.value))
    return items


def _append_credential_item(
    credentials: list[tuple[str, ast.AST]], key_node: ast.AST, value: ast.AST
) -> None:
    key = _static_string(key_node)
    if key is not None and _is_credential_name(key):
        credentials.append((key, value))


def _python_credential_values(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    credentials: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                credentials.extend((name, node.value) for name in _credential_target_names(target))
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                credentials.extend(
                    (name, node.value) for name in _credential_target_names(node.target)
                )
        elif isinstance(node, ast.AugAssign):
            credentials.extend((name, node) for name in _credential_target_names(node.target))
        elif isinstance(node, ast.NamedExpr):
            credentials.extend((name, node.value) for name in _credential_target_names(node.target))
        elif isinstance(node, ast.Call):
            credentials.extend(
                (keyword.arg, keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None and _is_credential_name(keyword.arg)
            )
            qualified = _qualified_name(node.func)
            if qualified == "dict":
                for argument in node.args:
                    credentials.extend(
                        (key, value)
                        for key, value in _mapping_items(argument)
                        if _is_credential_name(key)
                    )
            elif qualified == "os.putenv" and len(node.args) >= 2:
                _append_credential_item(credentials, node.args[0], node.args[1])
            elif qualified == "setattr" and len(node.args) >= 3:
                _append_credential_item(credentials, node.args[1], node.args[2])
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "setdefault":
                if len(node.args) >= 2:
                    _append_credential_item(credentials, node.args[0], node.args[1])
            elif isinstance(node.func, ast.Attribute) and node.func.attr == "update":
                for argument in node.args:
                    credentials.extend(
                        (key, value)
                        for key, value in _mapping_items(argument)
                        if _is_credential_name(key)
                    )
        elif isinstance(node, ast.Dict):
            credentials.extend(
                (key, value)
                for key, value in _mapping_items(node)
                if _is_credential_name(key)
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            credentials.extend(_python_argument_defaults(node.args))
    return credentials


def _python_credential_issues(entry: str, content: str) -> list[str]:
    try:
        tree = ast.parse(content)
    except SyntaxError as exc:
        return [f"{entry}: cannot inspect invalid Python syntax: line {exc.lineno or 0}"]

    issues: list[str] = []
    for variable, value in _python_credential_values(tree):
        if not _is_safe_python_credential(variable, value):
            issues.append(f"{entry}: non-placeholder credential assignment for {variable}")
    return issues


def _non_python_credential_issues(entry: str, content: str) -> list[str]:
    issues: list[str] = []
    for match in NON_PYTHON_CREDENTIAL_ASSIGNMENT.finditer(content):
        variable = match.group("variable")
        if not _is_credential_name(variable):
            continue
        value = match.group("value").strip()
        if match.group("operator") != "=" or not _is_safe_non_python_credential_assignment(
            variable, value
        ):
            issues.append(f"{entry}: non-placeholder credential assignment for {variable}")

    suffix = PurePosixPath(entry).suffix.lower()
    if suffix in {".yaml", ".yml", ".ini", ".cfg"}:
        lines = content.splitlines()
        for index, line in enumerate(lines):
            match = STRUCTURED_CREDENTIAL_ASSIGNMENT.fullmatch(line)
            if not match or not _is_credential_name(match.group("variable")):
                continue
            variable = match.group("variable")
            value = match.group("value").strip()
            if suffix in {".yaml", ".yml"} and value in {"|", ">", "|-", ">-", "|+", ">+"}:
                indentation = len(line) - len(line.lstrip())
                block: list[str] = []
                for following in lines[index + 1 :]:
                    if not following.strip():
                        block.append("")
                        continue
                    following_indent = len(following) - len(following.lstrip())
                    if following_indent <= indentation:
                        break
                    block.append(following.strip())
                value = "\n".join(block)
            if not _is_safe_non_python_credential_assignment(variable, value):
                issues.append(f"{entry}: non-placeholder credential assignment for {variable}")

        if suffix in {".yaml", ".yml"}:
            for match in FLOW_CREDENTIAL_ASSIGNMENT.finditer(content):
                variable = match.group("variable")
                if not _is_credential_name(variable):
                    continue
                if not _is_safe_non_python_credential_assignment(
                    variable, match.group("value").strip()
                ):
                    issues.append(f"{entry}: non-placeholder credential assignment for {variable}")
    return issues


def _json_credential_issues(entry: str, content: str) -> list[str]:
    issues: list[str] = []

    def inspect_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        seen: set[str] = set()
        for key, value in pairs:
            if key in seen:
                issues.append(f"{entry}: duplicate JSON key: {key}")
            seen.add(key)
            if _is_credential_name(key):
                safe = value is None or (
                    isinstance(value, str) and _is_static_placeholder_credential(value)
                )
                if not safe:
                    issues.append(f"{entry}: non-placeholder credential assignment for {key}")
            result[key] = value
        return result

    try:
        json.loads(content, object_pairs_hook=inspect_pairs)
    except json.JSONDecodeError:
        issues.append(f"{entry}: invalid JSON")
    return issues


def _shell_words(line: str) -> list[str]:
    try:
        lexer = shlex.shlex(line, posix=True, punctuation_chars=";|<>")
        lexer.whitespace_split = True
        lexer.commenters = "#"
        return list(lexer)
    except ValueError:
        return []


def _shell_command_issues(entry: str, content: str) -> list[str]:
    issues: list[str] = []
    for mutation in SHELL_PARAMETER_MUTATION.finditer(content):
        variable = mutation.group("variable")
        if _is_credential_name(variable):
            issues.append(f"{entry}: non-placeholder credential assignment for {variable}")

    for line in content.splitlines():
        words = _shell_words(line)
        for word in words:
            assignment = SHELL_ASSIGNMENT_WORD.fullmatch(word)
            if not assignment:
                continue
            variable = assignment.group("variable")
            if not _is_credential_name(variable):
                continue
            value = assignment.group("value")
            if assignment.group("operator") != "=" or not _is_safe_non_python_credential_assignment(
                variable, value
            ):
                issues.append(f"{entry}: non-placeholder credential assignment for {variable}")

        for position, word in enumerate(words):
            if word == "printf" and position + 2 < len(words) and words[position + 1] == "-v":
                variable = words[position + 2]
                if _is_credential_name(variable):
                    values = words[position + 3 :]
                    assigned_value = values[-1] if values else ""
                    if not _is_safe_non_python_credential_assignment(variable, assigned_value):
                        issues.append(
                            f"{entry}: non-placeholder credential assignment for {variable}"
                        )
            if word == "read" and "<<<" in words[position + 1 :]:
                redirect = words.index("<<<", position + 1)
                assigned_value = words[redirect + 1] if redirect + 1 < len(words) else ""
                for variable in words[position + 1 : redirect]:
                    if variable.startswith("-") or not _is_credential_name(variable):
                        continue
                    if not _is_safe_non_python_credential_assignment(variable, assigned_value):
                        issues.append(
                            f"{entry}: non-placeholder credential assignment for {variable}"
                        )
    return issues


def _heredoc_delimiter(match: re.Match[str]) -> str:
    for group in ("ansi", "localized", "single", "double", "escaped", "bare"):
        value = match.group(group)
        if value is not None:
            return value
    raise AssertionError("heredoc delimiter regex matched without a delimiter")


def _is_python_heredoc_command(command: str) -> bool:
    if re.search(
        r"(?<![A-Za-z0-9_])(?:[A-Za-z0-9_.-]+/)*python(?:3(?:\.[0-9]+)?)?(?![A-Za-z0-9_])",
        command,
    ):
        return True
    return re.search(
        r"\$(?:PYTHON_BIN)\b|\$\{(?:PYTHON_BIN|VENV_PY|python_bin|JUPYTER_PYTHON|python_cmd)\}",
        command,
    ) is not None


def _shell_credential_issues(entry: str, content: str) -> list[str]:
    shell_lines: list[str] = []
    embedded_blocks: list[tuple[bool, str]] = []
    lines = content.splitlines(keepends=True)
    index = 0
    while index < len(lines):
        line = lines[index]
        starts = list(SHELL_HEREDOC_START.finditer(line.rstrip("\r\n")))
        if not starts:
            shell_lines.append(line)
            index += 1
            continue

        shell_lines.append(line)
        index += 1
        for start in starts:
            delimiter = _heredoc_delimiter(start)
            strip_tabs = start.group("strip_tabs") is not None
            block: list[str] = []
            while index < len(lines):
                candidate = lines[index].rstrip("\r\n")
                comparable = candidate.lstrip("\t") if strip_tabs else candidate
                if comparable == delimiter:
                    break
                block.append(lines[index].lstrip("\t") if strip_tabs else lines[index])
                shell_lines.append("\n")
                index += 1
            if index >= len(lines):
                return [f"{entry}: unterminated heredoc: {delimiter}"]
            embedded_blocks.append(
                (_is_python_heredoc_command(line[: start.start()]), "".join(block))
            )
            shell_lines.append(lines[index])
            index += 1

    shell_content = "".join(shell_lines)
    issues = _non_python_credential_issues(entry, shell_content)
    issues.extend(_shell_command_issues(entry, shell_content))
    for is_python, block in embedded_blocks:
        if is_python:
            issues.extend(_python_credential_issues(entry, block))
        else:
            issues.extend(_non_python_credential_issues(entry, block))
            issues.extend(_shell_command_issues(entry, block))
    return issues


def _binary_content_issues(entry: str, data: bytes) -> list[str]:
    issues: list[str] = []
    disallowed_controls = sorted(
        {byte for byte in data if (byte < 32 and byte not in (9, 10, 13)) or byte == 127}
    )
    if disallowed_controls:
        rendered = ", ".join(f"0x{byte:02x}" for byte in disallowed_controls)
        issues.append(f"{entry}: disallowed control byte: {rendered}")
    if data.startswith(b"%PDF-"):
        issues.append(f"{entry}: binary signature is forbidden")
    return issues


def _allowed_private_version_spans(content: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    offset = 0
    for raw_line in content.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        stripped_offset = offset + len(line) - len(line.lstrip())
        for pattern in TYPED_VERSION_LINE_PATTERNS:
            match = pattern.fullmatch(stripped)
            if match:
                start, end = match.span("version")
                spans.append((stripped_offset + start, stripped_offset + end))
                break
        offset += len(raw_line)
    return spans


def _contains_private_network(content: str) -> bool:
    if PRIVATE_HOSTNAME_PATTERN.search(content):
        return True

    for assignment in NETWORK_ASSIGNMENT_PATTERN.finditer(content):
        if NETWORK_KEY_PATTERN.search(assignment.group("key")) and PRIVATE_IPV4_PATTERN.search(
            assignment.group("value")
        ):
            return True

    package_version_spans = _allowed_private_version_spans(content)
    for match in PRIVATE_IPV4_PATTERN.finditer(content):
        if any(
            version_start <= match.start() and match.end() <= version_end
            for version_start, version_end in package_version_spans
        ):
            continue
        return True
    return False


def _content_policy_issues(entry: str, content: str) -> list[str]:
    issues: list[str] = []
    if any(pattern.search(content) for pattern in LOCAL_PATH_PATTERNS) or LOCAL_USERNAME_PATTERN.search(content):
        issues.append(f"{entry}: private content contains a local user path")
    if _contains_private_network(content):
        issues.append(f"{entry}: private content contains an internal network host")

    if PRIVATE_KEY_HEADER.search(content):
        issues.append(f"{entry}: private-key header is forbidden")

    suffix = PurePosixPath(entry).suffix.lower()
    if suffix == ".py":
        issues.extend(_python_credential_issues(entry, content))
    elif suffix == ".sh":
        issues.extend(_shell_credential_issues(entry, content))
    elif suffix == ".json":
        issues.extend(_json_credential_issues(entry, content))
    else:
        issues.extend(_non_python_credential_issues(entry, content))

    return issues


def validate_manifest(
    root: Path,
    manifest_path: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[str]:
    """Return every release-boundary violation found in the manifest."""

    issues: list[str] = []
    try:
        entries = read_manifest(manifest_path)
    except UnicodeDecodeError:
        return [f"{manifest_path}: manifest must be UTF-8"]
    except OSError as exc:
        return [f"{manifest_path}: cannot read manifest: {exc}"]

    if any(not entry for entry in entries):
        issues.append("manifest contains a blank entry")
    if entries != sorted(entries):
        issues.append("manifest entries must be sorted")
    if len(entries) != len(set(entries)):
        issues.append("manifest contains a duplicate entry")
    if MANIFEST_BASENAME not in entries:
        issues.append(f"manifest must include {MANIFEST_BASENAME}")

    for entry in entries:
        if not _is_normalized_relative_path(entry):
            issues.append(f"{entry!r}: entry must be a normalized POSIX-relative path")
            continue

        issues.extend(_path_policy_issues(entry))
        candidate = root / PurePosixPath(entry)
        current = root
        has_symlinked_parent = False
        for component in PurePosixPath(entry).parts[:-1]:
            current /= component
            if current.is_symlink():
                issues.append(f"{entry}: symlinked parent component is forbidden")
                has_symlinked_parent = True
                break
        if has_symlinked_parent:
            continue
        if candidate.is_symlink():
            issues.append(f"{entry}: symlink entries are forbidden")
            continue
        if not candidate.exists():
            issues.append(f"{entry}: missing file")
            continue
        if not candidate.is_file():
            issues.append(f"{entry}: entry must be a regular file")
            continue

        size = candidate.stat().st_size
        if size > max_bytes:
            issues.append(f"{entry}: file exceeds the {max_bytes}-byte size limit")

        try:
            data = candidate.read_bytes()
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            issues.append(f"{entry}: file must be UTF-8")
            continue
        except OSError as exc:
            issues.append(f"{entry}: cannot read file: {exc}")
            continue
        issues.extend(_binary_content_issues(entry, data))
        issues.extend(_content_policy_issues(entry, content))

    return issues


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path(MANIFEST_BASENAME))
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.root.resolve()
    manifest = args.manifest
    if not manifest.is_absolute():
        manifest = root / manifest

    issues = validate_manifest(root, manifest, max_bytes=args.max_bytes)
    if issues:
        for issue in issues:
            print(f"[PUBLIC_RELEASE][ERROR] {issue}")
        return 1

    entries = read_manifest(manifest)
    total_bytes = sum((root / PurePosixPath(entry)).stat().st_size for entry in entries)
    print(f"[PUBLIC_RELEASE][OK] {len(entries)} files, {total_bytes} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
