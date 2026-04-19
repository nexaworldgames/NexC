#!/usr/bin/env python3
from __future__ import annotations

import ast
import dataclasses
import math as _math
import os
import re
import shutil
import sys
import time as _time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


VERSION = "1.3.0"

KEYWORDS = {
    "let", "var", "const", "mut", "print", "input",
    "if", "else", "elseif", "then", "end",
    "while", "for", "in", "to", "step", "do",
    "break", "continue", "return",
    "function", "fn",
    "true", "false", "nil", "null",
    "and", "or", "not",
    "try", "catch",
    "import", "from", "as",
    "int", "float", "bool", "string", "list", "map", "any", "number", "void",
    "range", "len", "size",
    "module", "export", "class", "new", "this", "super",
}

TYPE_ALIASES = {
    "int": int,
    "float": float,
    "bool": bool,
    "string": str,
    "str": str,
    "list": list,
    "array": list,
    "map": dict,
    "dict": dict,
    "any": object,
    "number": (int, float),
    "void": type(None),
}

BUILTIN_MODULE_NAMES = {"math", "strings", "files", "system", "time"}

class NexCError(Exception):
    def __init__(self, message: str, line: Optional[int] = None, source: Optional[str] = None):
        self.message = message
        self.line = line
        self.source = source
        super().__init__(self.__str__())

    def __str__(self) -> str:
        if self.line is None:
            return self.message
        suffix = f" at line {self.line}"
        if self.source:
            suffix += f" in {self.source}"
        return f"{self.message}{suffix}"

class NexCParseError(NexCError):
    pass

class NexCRuntimeError(NexCError):
    pass

class _ReturnSignal(Exception):
    def __init__(self, value: Any):
        self.value = value

class _BreakSignal(Exception):
    pass

class _ContinueSignal(Exception):
    pass

@dataclass
class Line:
    text: str
    number: int

@dataclass
class FunctionDef:
    name: str
    params: List[str]
    param_types: Dict[str, str]
    return_type: Optional[str]
    body: List[Line]
    closure: "Scope"

@dataclass
class ModuleNamespace:
    name: str
    values: Dict[str, Any]

    def __getattr__(self, item: str) -> Any:
        try:
            return self.values[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __repr__(self) -> str:
        return f"<module {self.name}>"

class Scope:
    def __init__(self, parent: Optional["Scope"] = None):
        self.parent = parent
        self.values: Dict[str, Any] = {}
        self.types: Dict[str, str] = {}

    def has_local(self, name: str) -> bool:
        return name in self.values

    def contains(self, name: str) -> bool:
        if name in self.values:
            return True
        return self.parent.contains(name) if self.parent else False

    def resolve_scope(self, name: str) -> Optional["Scope"]:
        if name in self.values:
            return self
        return self.parent.resolve_scope(name) if self.parent else None

    def get(self, name: str) -> Any:
        if name in self.values:
            return self.values[name]
        if self.parent:
            return self.parent.get(name)
        raise KeyError(name)

    def get_type(self, name: str) -> Optional[str]:
        if name in self.types:
            return self.types[name]
        if self.parent:
            return self.parent.get_type(name)
        return None

    def set_local(self, name: str, value: Any, type_name: Optional[str] = None) -> None:
        self.values[name] = value
        if type_name is not None:
            self.types[name] = type_name

    def set_existing_or_local(self, name: str, value: Any) -> None:
        scope = self.resolve_scope(name)
        if scope is None:
            self.values[name] = value
        else:
            scope.values[name] = value

    def flatten(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.parent:
            out.update(self.parent.flatten())
        out.update(self.values)
        return out

def strip_comment(line: str) -> str:
    out = []
    in_single = False
    in_double = False
    i = 0
    while i < len(line):
        ch = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""
        if ch == "\\":
            out.append(ch)
            if i + 1 < len(line):
                out.append(line[i + 1])
                i += 2
                continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "-" and nxt == "-" and not in_single and not in_double:
            break
        out.append(ch)
        i += 1
    return "".join(out).rstrip()

def preprocess(code: str) -> List[Line]:
    lines: List[Line] = []
    for idx, raw in enumerate(code.splitlines(), start=1):
        cleaned = strip_comment(raw).rstrip()
        if cleaned.strip():
            lines.append(Line(cleaned, idx))
    return lines

def is_identifier(name: str) -> bool:
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is not None and name not in KEYWORDS

def split_args(argstr: str) -> List[str]:
    args = []
    current = []
    depth = 0
    in_single = False
    in_double = False
    i = 0
    while i < len(argstr):
        ch = argstr[i]
        if ch == "\\":
            current.append(ch)
            if i + 1 < len(argstr):
                current.append(argstr[i + 1])
                i += 2
                continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch in "([{":
            if not in_single and not in_double:
                depth += 1
        elif ch in ")]}":
            if not in_single and not in_double:
                depth -= 1
        elif ch == "," and depth == 0 and not in_single and not in_double:
            item = "".join(current).strip()
            if item:
                args.append(item)
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args

def split_colon_annotation(text: str) -> Tuple[str, Optional[str]]:
    depth = 0
    in_single = False
    in_double = False
    for i, ch in enumerate(text):
        if ch == "\\":
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch in "([{":
            if not in_single and not in_double:
                depth += 1
        elif ch in ")]}":
            if not in_single and not in_double:
                depth -= 1
        elif ch == ":" and depth == 0 and not in_single and not in_double:
            return text[:i].strip(), text[i + 1 :].strip()
    return text.strip(), None

def nxc_to_python_expr(expr: str) -> str:
    expr = expr.strip()
    expr = re.sub(r"\btrue\b", "True", expr)
    expr = re.sub(r"\bfalse\b", "False", expr)
    expr = re.sub(r"\b(nil|null)\b", "None", expr)
    expr = re.sub(r"\bthen\b", "", expr)
    expr = re.sub(r"\bdo\b", "", expr)
    expr = re.sub(r"\band\b", "and", expr)
    expr = re.sub(r"\bor\b", "or", expr)
    expr = re.sub(r"\bnot\b", "not", expr)
    return expr.strip()

def safe_eval(expr: str, scope: Scope) -> Any:
    expr = nxc_to_python_expr(expr)
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise NexCRuntimeError(f"Invalid expression: {expr}") from exc
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.IfExp, ast.Compare,
        ast.Call, ast.Name, ast.Load, ast.Constant, ast.List, ast.Tuple, ast.Dict,
        ast.Subscript, ast.Slice, ast.Attribute, ast.keyword,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
        ast.UAdd, ast.USub, ast.Not, ast.And, ast.Or,
        ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    )
    for sub in ast.walk(node):
        if not isinstance(sub, allowed):
            raise NexCRuntimeError(f"Unsupported expression element: {type(sub).__name__}")
        if isinstance(sub, ast.Name) and not scope.contains(sub.id):
            raise NexCRuntimeError(f"Unknown name: {sub.id}")
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and not scope.contains(sub.func.id):
            raise NexCRuntimeError(f"Unknown function: {sub.func.id}")
    env = scope.flatten()
    env["range"] = range
    env["int"] = int
    env["float"] = float
    env["str"] = str
    env["bool"] = bool
    env["len"] = len
    env["list"] = list
    env["dict"] = dict
    env["abs"] = abs
    env["min"] = min
    env["max"] = max
    env["round"] = round
    env["sum"] = sum
    env["sorted"] = sorted
    compiled = compile(node, "<nexc>", "eval")
    return eval(compiled, {"__builtins__": {}}, env)

def value_matches_type(value: Any, type_name: Optional[str]) -> bool:
    if not type_name or type_name == "any":
        return True
    expected = TYPE_ALIASES.get(type_name)
    if expected is None:
        return True
    if expected is type(None):
        return value is None
    if expected is object:
        return True
    return isinstance(value, expected)

def format_type_name(type_name: Optional[str]) -> str:
    return type_name or "any"

def collect_block(lines: Sequence[Line], start_idx: int) -> int:
    depth = 0
    for i in range(start_idx, len(lines)):
        text = lines[i].text.strip()
        head = text.split()[0] if text.split() else ""
        if re.match(r"^(if|while|for|function|fn|try)\b", text):
            depth += 1
        elif text == "end":
            depth -= 1
            if depth == 0:
                return i
    raise NexCParseError("Missing 'end'", line=lines[start_idx].number)

def detect_assignment(text: str) -> bool:
    if text.startswith("let "):
        return True
    if text.startswith("import ") or text.startswith("from "):
        return False
    if text in {"break", "continue"}:
        return False
    if text.startswith(("if ", "while ", "for ", "function ", "fn ", "try", "return", "catch", "else", "elseif ")):
        return False
    return "=" in text and not re.search(r"[=!<>]=|=>|<=|>=", text)

class Interpreter:
    def __init__(self, code: str, *, source_name: str = "<stdin>", base_dir: Optional[Path] = None, argv: Optional[List[str]] = None, module_mode: bool = False):
        self.source_name = source_name
        self.base_dir = base_dir or Path.cwd()
        self.argv = argv or []
        self.module_mode = module_mode
        self.lines = preprocess(code)
        self.functions: Dict[str, FunctionDef] = {}
        self.import_stack: List[str] = []
        self.loaded_modules: Dict[str, ModuleNamespace] = {}
        self.scope = Scope()
        self.exit_code = 0
        self._install_builtins(self.scope)
        self.scope.set_local("args", self.argv)
        self.scope.set_local("argv", self.argv)
        self.scope.set_local("version", VERSION)
        self.scope.set_local("true", True)
        self.scope.set_local("false", False)
        self.scope.set_local("nil", None)
        self.scope.set_local("null", None)

    def _install_builtins(self, scope: Scope) -> None:
        scope.set_local("print", self._builtin_print)
        scope.set_local("input", self._builtin_input)
        scope.set_local("len", len)
        scope.set_local("size", len)
        scope.set_local("push", self._builtin_push)
        scope.set_local("pop", self._builtin_pop)
        scope.set_local("shift", self._builtin_shift)
        scope.set_local("unshift", self._builtin_unshift)
        scope.set_local("get", self._builtin_get)
        scope.set_local("keys", self._builtin_keys)
        scope.set_local("values", self._builtin_values)
        scope.set_local("items", self._builtin_items)
        scope.set_local("sort", self._builtin_sort)
        scope.set_local("reverse", self._builtin_reverse)
        scope.set_local("slice", self._builtin_slice)
        scope.set_local("join", self._builtin_join)
        scope.set_local("split", self._builtin_split)
        scope.set_local("trim", self._builtin_trim)
        scope.set_local("lower", self._builtin_lower)
        scope.set_local("upper", self._builtin_upper)
        scope.set_local("replace", self._builtin_replace)
        scope.set_local("match", self._builtin_match)
        scope.set_local("search", self._builtin_search)
        scope.set_local("sleep", self._builtin_sleep)
        scope.set_local("now", self._builtin_now)
        scope.set_local("read", self._builtin_read)
        scope.set_local("write", self._builtin_write)
        scope.set_local("append", self._builtin_append)
        scope.set_local("exists", self._builtin_exists)
        scope.set_local("mkdir", self._builtin_mkdir)
        scope.set_local("delete", self._builtin_delete)
        scope.set_local("copy", self._builtin_copy)
        scope.set_local("move", self._builtin_move)
        scope.set_local("path", self._builtin_path)
        scope.set_local("file", self._builtin_file)
        scope.set_local("dir", self._builtin_dir)
        scope.set_local("env", self._builtin_env)
        scope.set_local("exit", self._builtin_exit)
        scope.set_local("range", range)
        scope.set_local("str", str)
        scope.set_local("int", int)
        scope.set_local("float", float)
        scope.set_local("bool", bool)
        scope.set_local("abs", abs)
        scope.set_local("min", min)
        scope.set_local("max", max)
        scope.set_local("round", round)
        scope.set_local("sum", sum)
        scope.set_local("sorted", sorted)

    def _builtin_print(self, *args):
        print(*args)

    def _builtin_input(self, prompt: str = ""):
        return input(prompt)

    def _builtin_push(self, arr, value):
        arr.append(value)
        return arr

    def _builtin_pop(self, arr):
        return arr.pop()

    def _builtin_shift(self, arr):
        if not arr:
            raise NexCRuntimeError("shift() on empty list")
        return arr.pop(0)

    def _builtin_unshift(self, arr, value):
        arr.insert(0, value)
        return arr

    def _builtin_get(self, obj, key, default=None):
        if isinstance(obj, (list, tuple)):
            return obj[key]
        return obj.get(key, default)

    def _builtin_keys(self, obj):
        return list(obj.keys())

    def _builtin_values(self, obj):
        return list(obj.values())

    def _builtin_items(self, obj):
        return list(obj.items())

    def _builtin_sort(self, arr):
        arr.sort()
        return arr

    def _builtin_reverse(self, arr):
        arr.reverse()
        return arr

    def _builtin_slice(self, obj, start=None, end=None):
        return obj[start:end]

    def _builtin_join(self, arr, sep=""):
        return sep.join(arr)

    def _builtin_split(self, s, sep=None):
        return s.split(sep)

    def _builtin_trim(self, s):
        return s.strip()

    def _builtin_lower(self, s):
        return s.lower()

    def _builtin_upper(self, s):
        return s.upper()

    def _builtin_replace(self, s, old, new):
        return s.replace(old, new)

    def _builtin_match(self, pattern, s):
        return re.fullmatch(pattern, s) is not None

    def _builtin_search(self, pattern, s):
        return re.search(pattern, s) is not None

    def _builtin_sleep(self, seconds):
        _time.sleep(float(seconds))
        return None

    def _builtin_now(self):
        from datetime import datetime
        return datetime.now().isoformat()

    def _builtin_read(self, path):
        return Path(path).read_text(encoding="utf-8")

    def _builtin_write(self, path, content):
        Path(path).write_text(str(content), encoding="utf-8")
        return True

    def _builtin_append(self, path, content):
        with open(path, "a", encoding="utf-8") as f:
            f.write(str(content))
        return True

    def _builtin_exists(self, path):
        return Path(path).exists()

    def _builtin_mkdir(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return True

    def _builtin_delete(self, path):
        p = Path(path)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
        return True

    def _builtin_copy(self, src, dst):
        shutil.copy2(src, dst)
        return True

    def _builtin_move(self, src, dst):
        shutil.move(src, dst)
        return True

    def _builtin_path(self, *parts):
        return str(Path(*map(str, parts)))

    def _builtin_file(self, path):
        return Path(path).name

    def _builtin_dir(self, path):
        return str(Path(path).parent)

    def _builtin_env(self, key, default=None):
        return os.getenv(str(key), default)

    def _builtin_exit(self, code=0):
        self.exit_code = int(code)
        raise SystemExit(self.exit_code)

    def error(self, message: str, line: Optional[Line] = None, exc: type = NexCRuntimeError):
        raise exc(message, line.number if line else None, self.source_name)

    def run(self) -> int:
        try:
            self._execute_block(self.lines, self.scope)
        except _ReturnSignal as signal:
            return 0 if signal.value is None else 0
        return self.exit_code

    def _execute_block(self, lines: Sequence[Line], scope: Scope) -> Any:
        i = 0
        while i < len(lines):
            line = lines[i]
            text = line.text.strip()

            if text.startswith("let "):
                self._handle_let(line, scope)
                i += 1
                continue

            if text.startswith("function ") or text.startswith("fn "):
                end_idx = collect_block(lines, i)
                body = list(lines[i + 1:end_idx])
                self._register_function(line, body, scope)
                i = end_idx + 1
                continue

            if text.startswith("if "):
                end_idx = self._execute_if_chain(lines, i, scope)
                i = end_idx + 1
                continue

            if text.startswith("while "):
                end_idx = collect_block(lines, i)
                self._execute_while(line, list(lines[i + 1:end_idx]), scope)
                i = end_idx + 1
                continue

            if text.startswith("for "):
                end_idx = collect_block(lines, i)
                self._execute_for(line, list(lines[i + 1:end_idx]), scope)
                i = end_idx + 1
                continue

            if text == "try":
                end_idx = collect_block(lines, i)
                self._execute_try(list(lines[i + 1:end_idx]), scope, source_line=line)
                i = end_idx + 1
                continue

            if text.startswith("import "):
                self._handle_import(line, scope)
                i += 1
                continue

            if text.startswith("from "):
                self._handle_from_import(line, scope)
                i += 1
                continue

            if text == "break":
                raise _BreakSignal()

            if text == "continue":
                raise _ContinueSignal()

            if text.startswith("return"):
                expr = text[6:].strip()
                value = None if not expr else safe_eval(expr, scope)
                raise _ReturnSignal(value)

            if text in {"end", "else"} or text.startswith("elseif ") or text.startswith("catch"):
                return None

            if detect_assignment(text):
                self._handle_assignment(line, scope)
                i += 1
                continue

            self._handle_expression_statement(line, scope)
            i += 1
        return None

    def _handle_let(self, line: Line, scope: Scope) -> None:
        payload = line.text.strip()[4:].strip()
        name_part, expr = payload.split("=", 1) if "=" in payload else (None, None)
        if expr is None:
            self.error("Invalid let statement", line, NexCParseError)
        lhs = name_part.strip()
        name, type_name = split_colon_annotation(lhs)
        if not is_identifier(name):
            self.error(f"Invalid variable name: {name}", line, NexCParseError)
        value = safe_eval(expr, scope)
        if type_name and not value_matches_type(value, type_name):
            self.error(f"Type mismatch for '{name}': expected {type_name}, got {type(value).__name__}", line)
        scope.set_local(name, value, type_name)

    def _handle_assignment(self, line: Line, scope: Scope) -> None:
        text = line.text.strip()
        if text.startswith("let "):
            return
        lhs, expr = text.split("=", 1)
        lhs = lhs.strip()
        expr = expr.strip()
        index_match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(\s*\[[^\]]+\])?", lhs)
        if not index_match:
            self.error(f"Invalid assignment: {text}", line, NexCParseError)
        name = index_match.group(1)
        index_part = index_match.group(2)
        value = safe_eval(expr, scope)
        if index_part:
            target = scope.get(name) if scope.contains(name) else None
            if target is None:
                self.error(f"Unknown variable: {name}", line)
            idx_expr = index_part.strip()[1:-1]
            idx = safe_eval(idx_expr, scope)
            try:
                target[idx] = value
            except Exception as exc:
                self.error(f"Index assignment failed: {exc}", line)
        else:
            existing_type = scope.get_type(name)
            if existing_type and not value_matches_type(value, existing_type):
                self.error(f"Type mismatch for '{name}': expected {existing_type}, got {type(value).__name__}", line)
            scope.set_existing_or_local(name, value)

    def _register_function(self, header: Line, body: List[Line], scope: Scope) -> None:
        match = re.match(r"^(?:function|fn)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*(?::\s*([A-Za-z_][A-Za-z0-9_]*))?\s*$", header.text.strip())
        if not match:
            self.error(f"Invalid function declaration: {header.text.strip()}", header, NexCParseError)
        name = match.group(1)
        param_blob = match.group(2).strip()
        return_type = match.group(3)
        params: List[str] = []
        param_types: Dict[str, str] = {}
        if param_blob:
            for raw in split_args(param_blob):
                p_name, p_type = split_colon_annotation(raw)
                if not is_identifier(p_name):
                    self.error(f"Invalid parameter name: {p_name}", header, NexCParseError)
                params.append(p_name)
                if p_type:
                    param_types[p_name] = p_type
        self.functions[name] = FunctionDef(name, params, param_types, return_type, body, scope)

    def _call_function(self, name: str, args: List[Any], *, calling_scope: Scope) -> Any:
        fn = self.functions.get(name)
        if not fn:
            raise NexCRuntimeError(f"Unknown function: {name}", None, self.source_name)
        if len(args) != len(fn.params):
            raise NexCRuntimeError(f"{name} expects {len(fn.params)} args, got {len(args)}", None, self.source_name)
        local = Scope(parent=fn.closure)
        local.values.update(self._module_safe_bindings())
        for p, a in zip(fn.params, args):
            expected = fn.param_types.get(p)
            if expected and not value_matches_type(a, expected):
                raise NexCRuntimeError(f"Type mismatch for parameter '{p}': expected {expected}, got {type(a).__name__}", None, self.source_name)
            local.set_local(p, a, expected)
        try:
            return self._execute_block(fn.body, local)
        except _ReturnSignal as signal:
            if fn.return_type and not value_matches_type(signal.value, fn.return_type):
                raise NexCRuntimeError(f"Return type mismatch in {name}: expected {fn.return_type}, got {type(signal.value).__name__}", None, self.source_name)
            return signal.value

    def _module_safe_bindings(self) -> Dict[str, Any]:
        return {
            "print": self._builtin_print,
            "input": self._builtin_input,
            "len": len,
            "size": len,
            "push": self._builtin_push,
            "pop": self._builtin_pop,
            "shift": self._builtin_shift,
            "unshift": self._builtin_unshift,
            "get": self._builtin_get,
            "keys": self._builtin_keys,
            "values": self._builtin_values,
            "items": self._builtin_items,
            "sort": self._builtin_sort,
            "reverse": self._builtin_reverse,
            "slice": self._builtin_slice,
            "join": self._builtin_join,
            "split": self._builtin_split,
            "trim": self._builtin_trim,
            "lower": self._builtin_lower,
            "upper": self._builtin_upper,
            "replace": self._builtin_replace,
            "match": self._builtin_match,
            "search": self._builtin_search,
            "sleep": self._builtin_sleep,
            "now": self._builtin_now,
            "read": self._builtin_read,
            "write": self._builtin_write,
            "append": self._builtin_append,
            "exists": self._builtin_exists,
            "mkdir": self._builtin_mkdir,
            "delete": self._builtin_delete,
            "copy": self._builtin_copy,
            "move": self._builtin_move,
            "path": self._builtin_path,
            "file": self._builtin_file,
            "dir": self._builtin_dir,
            "env": self._builtin_env,
            "exit": self._builtin_exit,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "abs": abs,
            "min": min,
            "max": max,
            "round": round,
            "sum": sum,
            "sorted": sorted,
            "true": True,
            "false": False,
            "nil": None,
            "null": None,
            "version": VERSION,
        }

    def _execute_if_chain(self, lines: Sequence[Line], start_idx: int, scope: Scope) -> int:
        branches: List[Tuple[Optional[str], List[Line], Line]] = []
        i = start_idx
        current_cond = lines[i].text.strip()[2:].strip()  # after 'if'
        current_body: List[Line] = []
        depth = 0
        i += 1
        while i < len(lines):
            line = lines[i]
            text = line.text.strip()
            if re.match(r"^(if|while|for|function|fn|try)\b", text):
                depth += 1
                current_body.append(line)
            elif text == "end":
                if depth == 0:
                    branches.append((current_cond, current_body, lines[start_idx]))
                    break
                depth -= 1
                current_body.append(line)
            elif depth == 0 and text.startswith("elseif "):
                branches.append((current_cond, current_body, lines[start_idx]))
                current_cond = text[7:].strip()
                current_body = []
            elif depth == 0 and text == "else":
                branches.append((current_cond, current_body, lines[start_idx]))
                current_cond = None
                current_body = []
            else:
                current_body.append(line)
            i += 1
        else:
            self.error("Missing 'end' for if block", lines[start_idx], NexCParseError)
        executed = False
        for cond, body, header_line in branches:
            if cond is None:
                self._execute_block(body, scope)
                executed = True
                break
            if self._truthy(safe_eval(cond, scope)):
                self._execute_block(body, scope)
                executed = True
                break
        return i

    def _execute_while(self, header: Line, body: List[Line], scope: Scope) -> None:
        cond = header.text.strip()[5:].strip()
        if cond.endswith("do"):
            cond = cond[:-2].strip()
        count = 0
        while self._truthy(safe_eval(cond, scope)):
            count += 1
            if count > 100000:
                self.error("while loop exceeded 100000 iterations", header)
            try:
                self._execute_block(body, scope)
            except _ContinueSignal:
                continue
            except _BreakSignal:
                break

    def _execute_for(self, header: Line, body: List[Line], scope: Scope) -> None:
        text = header.text.strip()[3:].strip()
        if text.endswith("do"):
            text = text[:-2].strip()
        numeric = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s+to\s+(.+?)(?:\s+step\s+(.+?))?$", text)
        iterable = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.+)$", text)
        if numeric:
            var = numeric.group(1)
            start = safe_eval(numeric.group(2), scope)
            end = safe_eval(numeric.group(3), scope)
            step = safe_eval(numeric.group(4), scope) if numeric.group(4) else 1
            if step == 0:
                self.error("for step cannot be 0", header)
            cur = start
            def cond(a, b, s):
                return a <= b if s > 0 else a >= b
            while cond(cur, end, step):
                scope.set_local(var, cur)
                try:
                    self._execute_block(body, scope)
                except _ContinueSignal:
                    pass
                except _BreakSignal:
                    break
                cur += step
            return
        if iterable:
            var = iterable.group(1)
            seq = safe_eval(iterable.group(2), scope)
            for item in seq:
                scope.set_local(var, item)
                try:
                    self._execute_block(body, scope)
                except _ContinueSignal:
                    continue
                except _BreakSignal:
                    break
            return
        self.error("Invalid for syntax", header, NexCParseError)

    def _execute_try(self, body: List[Line], scope: Scope, source_line: Line) -> None:
        catch_idx = None
        catch_name = "_error"
        depth = 0
        for idx, line in enumerate(body):
            text = line.text.strip()
            if re.match(r"^(if|while|for|function|fn|try)\b", text):
                depth += 1
            elif text == "end":
                depth -= 1
            elif depth == 0 and text.startswith("catch"):
                catch_idx = idx
                tail = text[5:].strip()
                if tail:
                    parts = tail.split()
                    if parts:
                        catch_name = parts[0]
                break
        try_lines = body if catch_idx is None else body[:catch_idx]
        catch_lines = None if catch_idx is None else body[catch_idx + 1:]
        try:
            self._execute_block(try_lines, scope)
        except (_ReturnSignal, _BreakSignal, _ContinueSignal):
            raise
        except Exception as exc:
            if catch_lines is None:
                if isinstance(exc, NexCError):
                    raise
                raise NexCRuntimeError(str(exc), source_line.number, self.source_name)
            scope.set_local(catch_name, str(exc))
            self._execute_block(catch_lines, scope)

    def _handle_import(self, line: Line, scope: Scope) -> None:
        text = line.text.strip()[6:].strip()
        if " as " in text:
            module_name, alias = [part.strip() for part in text.split(" as ", 1)]
        else:
            module_name, alias = text, None
        module = self.load_module(module_name, scope)
        scope.set_local(alias or module_name.split(".")[-1], module)

    def _handle_from_import(self, line: Line, scope: Scope) -> None:
        text = line.text.strip()[4:].strip()
        if " import " not in text:
            self.error("Invalid from-import syntax", line, NexCParseError)
        module_name, rest = text.split(" import ", 1)
        module = self.load_module(module_name.strip(), scope)
        for item in split_args(rest):
            if " as " in item:
                name, alias = [part.strip() for part in item.split(" as ", 1)]
            else:
                name, alias = item.strip(), item.strip()
            if not is_identifier(alias):
                self.error(f"Invalid import alias: {alias}", line, NexCParseError)
            if name not in module.values:
                self.error(f"Module '{module_name}' has no export '{name}'", line, NexCParseError)
            scope.set_local(alias, module.values[name])

    def load_module(self, module_name: str, caller_scope: Scope) -> ModuleNamespace:
        module_name = module_name.strip().strip('"').strip("'")
        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]
        if module_name in BUILTIN_MODULE_NAMES:
            module = self._load_builtin_module(module_name)
            self.loaded_modules[module_name] = module
            return module
        search_paths = self._module_search_paths(module_name)
        for path in search_paths:
            if path.exists() and path.is_file():
                code = path.read_text(encoding="utf-8")
                interp = Interpreter(code, source_name=str(path), base_dir=path.parent, argv=self.argv, module_mode=True)
                interp.loaded_modules = self.loaded_modules
                interp.import_stack = self.import_stack + [module_name]
                interp.scope.parent = caller_scope
                interp._install_builtins(interp.scope)
                interp.scope.set_local("args", self.argv)
                interp.scope.set_local("argv", self.argv)
                interp.scope.set_local("version", VERSION)
                interp.scope.set_local("true", True)
                interp.scope.set_local("false", False)
                interp.scope.set_local("nil", None)
                interp.scope.set_local("null", None)
                interp.run()
                exports = interp._module_exports()
                module = ModuleNamespace(module_name.split(".")[-1], exports)
                self.loaded_modules[module_name] = module
                return module
        raise NexCRuntimeError(f"Module not found: {module_name}", None, self.source_name)

    def _module_search_paths(self, module_name: str) -> List[Path]:
        rel = Path(*module_name.split("."))
        candidates = [
            self.base_dir / "modules" / rel.with_suffix(".nxc"),
            self.base_dir / "stdlib" / rel.with_suffix(".nxc"),
            self.base_dir / rel.with_suffix(".nxc"),
            Path.cwd() / "modules" / rel.with_suffix(".nxc"),
            Path.cwd() / "stdlib" / rel.with_suffix(".nxc"),
            Path.cwd() / rel.with_suffix(".nxc"),
        ]
        # remove duplicates while preserving order
        seen = set()
        out = []
        for p in candidates:
            sp = str(p.resolve()) if p.exists() else str(p)
            if sp not in seen:
                seen.add(sp)
                out.append(p)
        return out

    def _load_builtin_module(self, name: str) -> ModuleNamespace:
        if name == "math":
            values = {
                "sqrt": _math.sqrt,
                "pow": _math.pow,
                "abs": abs,
                "min": min,
                "max": max,
                "round": round,
                "floor": _math.floor,
                "ceil": _math.ceil,
                "pi": _math.pi,
                "e": _math.e,
            }
        elif name == "strings":
            values = {
                "lower": str.lower,
                "upper": str.upper,
                "trim": str.strip,
                "split": self._builtin_split,
                "join": self._builtin_join,
                "replace": str.replace,
            }
        elif name == "files":
            values = {
                "read": self._builtin_read,
                "write": self._builtin_write,
                "append": self._builtin_append,
                "exists": self._builtin_exists,
                "mkdir": self._builtin_mkdir,
                "delete": self._builtin_delete,
                "copy": self._builtin_copy,
                "move": self._builtin_move,
                "path": self._builtin_path,
                "file": self._builtin_file,
                "dir": self._builtin_dir,
            }
        elif name == "system":
            values = {
                "env": self._builtin_env,
                "args": self.argv,
                "argv": self.argv,
                "version": VERSION,
                "exit": self._builtin_exit,
            }
        elif name == "time":
            values = {
                "now": self._builtin_now,
                "sleep": self._builtin_sleep,
                "timestamp": lambda: _time.time(),
            }
        else:
            values = {}
        return ModuleNamespace(name, values)

    def _module_exports(self) -> Dict[str, Any]:
        exports = {}
        exports.update(self.scope.values)
        for name, fn in self.functions.items():
            exports[name] = self._make_function_export(name, fn)
        # remove internal entries and builtins
        skip = {
            "print", "input", "len", "size", "push", "pop", "shift", "unshift", "get",
            "keys", "values", "items", "sort", "reverse", "slice", "join", "split",
            "trim", "lower", "upper", "replace", "match", "search", "sleep", "now",
            "read", "write", "append", "exists", "mkdir", "delete", "copy", "move",
            "path", "file", "dir", "env", "exit", "range", "str", "int", "float",
            "bool", "abs", "min", "max", "round", "sum", "sorted", "args", "argv",
            "version", "true", "false", "nil", "null",
        }
        return {k: v for k, v in exports.items() if k not in skip and not k.startswith("_")}

    def _make_function_export(self, name: str, fn: FunctionDef):
        def wrapper(*args):
            return self._call_function(name, list(args), calling_scope=fn.closure)
        return wrapper

    def _handle_expression_statement(self, line: Line, scope: Scope) -> None:
        text = line.text.strip()
        call_match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)\s*\((.*)\)\s*$", text)
        if call_match:
            name = call_match.group(1)
            args_raw = call_match.group(2).strip()
            args = [safe_eval(arg, scope) for arg in split_args(args_raw)] if args_raw else []
            if "." in name:
                target, attr = name.split(".", 1)
                obj = scope.get(target)
                fn = getattr(obj, attr)
                return fn(*args)
            if name in self.functions:
                return self._call_function(name, args, calling_scope=scope)
            if scope.contains(name):
                fn = scope.get(name)
                if callable(fn):
                    return fn(*args)
        safe_eval(text, scope)

    def _truthy(self, value: Any) -> bool:
        return bool(value)

def repl() -> int:
    print(f"NexC v{VERSION} REPL. Type 'exit(0)' or Ctrl+C to quit.")
    interp = Interpreter("", source_name="<repl>")
    while True:
        try:
            line = input("nexc> ")
        except EOFError:
            return 0
        if not line.strip():
            continue
        try:
            lines = preprocess(line)
            interp._execute_block(lines, interp.scope)
        except SystemExit:
            return 0
        except NexCError as exc:
            print(f"[error] {exc}")
        except Exception as exc:
            print(f"[error] {exc}")

def create_project(name: str, target: Path) -> None:
    project = target / name
    project.mkdir(parents=True, exist_ok=True)
    (project / "examples").mkdir(exist_ok=True)
    (project / "modules").mkdir(exist_ok=True)
    (project / "README.md").write_text(f"# {name}\n\nCreated with NexC {VERSION}.\n", encoding="utf-8")
    (project / "main.nxc").write_text('print("hello from NexC")\n', encoding="utf-8")
    print(f"Created project at {project}")

def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    if len(argv) == 1:
        return repl()
    cmd = argv[1]
    if cmd in ("-h", "--help", "help"):
        print(f"NexC {VERSION}")
        print("Usage:")
        print("  nexc run <script.nxc> [args...]")
        print("  nexc <script.nxc> [args...]")
        print("  nexc repl")
        print("  nexc version")
        print("  nexc new <project-name>")
        return 0
    if cmd in ("version", "--version", "-v"):
        print(VERSION)
        return 0
    if cmd == "repl":
        return repl()
    if cmd == "new":
        if len(argv) < 3:
            print("Usage: nexc new <project-name>", file=sys.stderr)
            return 1
        create_project(argv[2], Path.cwd())
        return 0
    if cmd == "run":
        if len(argv) < 3:
            print("Usage: nexc run <script.nxc> [args...]", file=sys.stderr)
            return 1
        script = Path(argv[2])
        run_args = argv[3:]
    else:
        script = Path(cmd)
        run_args = argv[2:]
    if not script.exists():
        print(f"File not found: {script}", file=sys.stderr)
        return 1
    code = script.read_text(encoding="utf-8")
    interp = Interpreter(code, source_name=str(script), base_dir=script.parent, argv=run_args)
    try:
        return interp.run()
    except NexCError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    except SystemExit as exc:
        return int(exc.code or 0)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
