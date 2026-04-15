#!/usr/bin/env python3
"""
NexC v1.0 interpreter prototype.

This is a small Lua-like interpreter for the first usable NexC release.
It supports:
- let / assignment
- print()
- if / elseif / else / end
- while / end
- function / return / end
- arrays and maps
- a small standard library
- try / catch / end

The goal is to be simple, readable, and easy to extend.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


KEYWORDS = {
    "let", "var", "const", "mut", "print", "input", "if", "else", "elseif", "then",
    "end", "while", "for", "foreach", "loop", "break", "continue", "return",
    "function", "fn", "lambda", "class", "new", "this", "super", "true", "false",
    "nil", "null", "and", "or", "not", "in", "is", "as", "try", "catch", "finally",
    "throw", "assert", "import", "from", "export", "use", "include", "require",
    "package", "module", "namespace", "alias", "open", "close", "read", "write",
    "append", "delete", "exists", "copy", "move", "rename", "mkdir", "rmdir", "path",
    "dir", "file", "stream", "int", "float", "bool", "string", "char", "byte",
    "array", "list", "tuple", "map", "dict", "set", "object", "table", "vector",
    "matrix", "range", "enum", "struct", "union", "optional", "any", "void",
    "number", "text", "time", "date", "datetime", "duration", "uuid", "json", "xml",
    "yaml", "regex", "add", "sub", "mul", "div", "mod", "pow", "abs", "min", "max",
    "floor", "ceil", "round", "sqrt", "sin", "cos", "tan", "log", "exp", "rand",
    "seed", "len", "size", "push", "pop", "shift", "unshift", "get", "keys",
    "values", "items", "sort", "reverse", "slice", "join", "split", "trim", "lower",
    "upper", "replace", "match", "search", "await", "async", "spawn", "sleep",
    "delay", "signal", "event", "emit", "listen", "once", "on", "off", "schedule",
    "timer", "tick", "timeit", "parallel", "thread", "lock", "unlock", "atomic",
    "yield", "resume", "pause", "start", "stop", "restart", "console", "stdin",
    "stdout", "stderr", "warn", "error", "debug", "info", "trace", "prompt", "clear",
    "exit", "status", "env", "args", "argv", "os", "arch", "platform", "version",
    "clock", "timezone", "now", "today", "tomorrow", "yesterday", "http", "https",
    "getreq", "postreq", "putreq", "patchreq", "deletereq", "header", "cookie",
    "query", "body", "url", "host", "port", "connect", "disconnect", "socket",
    "server", "client", "listenport", "download", "upload", "ping", "dns", "ip",
    "tcp", "udp", "auth", "login", "logout", "token", "hash", "encrypt", "decrypt",
    "sign", "verify", "permissions", "role", "admin", "user", "guest", "owner",
    "secure", "unsafe", "sandbox", "permission", "grant", "revoke", "private",
    "public", "protected", "secret", "config", "setting", "theme", "color",
}

BUILTIN_FUNCS = {}
MAX_RECURSION = 2000


class NexCRuntimeError(RuntimeError):
    pass


class NexCParseError(RuntimeError):
    pass


@dataclasses.dataclass
class FunctionDef:
    params: List[str]
    body: List[str]


def strip_comment(line: str) -> str:
    # Remove -- comments outside quotes
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


def preprocess(code: str) -> List[str]:
    lines = []
    for raw in code.splitlines():
        line = strip_comment(raw).rstrip()
        if line.strip():
            lines.append(line)
    return lines


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def is_identifier(name: str) -> bool:
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is not None


def nxc_to_python_expr(expr: str) -> str:
    expr = expr.strip()
    expr = expr.replace("true", "True").replace("false", "False")
    expr = re.sub(r"\bnil\b|\bnull\b", "None", expr)
    expr = re.sub(r"\bthen\b", "", expr)
    expr = re.sub(r"\band\b", "and", expr)
    expr = re.sub(r"\bor\b", "or", expr)
    expr = re.sub(r"\bnot\b", "not", expr)
    # allow string concatenation and indexing etc. as Python-compatible expressions
    return expr


def safe_eval(expr: str, env: Dict[str, Any]) -> Any:
    expr = nxc_to_python_expr(expr)
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise NexCRuntimeError(f"Invalid expression: {expr}") from e

    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.IfExp, ast.Compare,
        ast.Call, ast.Name, ast.Load, ast.Constant, ast.List, ast.Tuple, ast.Dict,
        ast.Subscript, ast.Slice, ast.Index, ast.Attribute, ast.keyword, ast.JoinedStr,
        ast.FormattedValue, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
        ast.FloorDiv, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or, ast.Eq, ast.NotEq,
        ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    )
    for sub in ast.walk(node):
        if not isinstance(sub, allowed):
            raise NexCRuntimeError(f"Unsupported expression element: {type(sub).__name__}")
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            if sub.func.id not in env:
                raise NexCRuntimeError(f"Unknown function: {sub.func.id}")
        if isinstance(sub, ast.Name):
            if sub.id not in env:
                raise NexCRuntimeError(f"Unknown name: {sub.id}")

    compiled = compile(node, "<nexc>", "eval")
    return eval(compiled, {"__builtins__": {}}, env)


def split_args(argstr: str) -> List[str]:
    args = []
    current = []
    depth = 0
    in_single = False
    in_double = False
    for i, ch in enumerate(argstr):
        if ch == "\\":
            current.append(ch)
            if i + 1 < len(argstr):
                current.append(argstr[i + 1])
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch in "([{" and not in_single and not in_double:
            depth += 1
        elif ch in ")]}" and not in_single and not in_double:
            depth -= 1
        elif ch == "," and depth == 0 and not in_single and not in_double:
            args.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


class Interpreter:
    def __init__(self, code: str, argv: Optional[List[str]] = None):
        self.lines = preprocess(code)
        self.env: Dict[str, Any] = {}
        self.functions: Dict[str, FunctionDef] = {}
        self.argv = argv or []
        self.exit_code = 0
        self.env.update({
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
            "args": self.argv,
            "argv": self.argv,
            "exit": self._builtin_exit,
        })

    def _builtin_print(self, *args):
        print(*args)

    def _builtin_input(self, prompt=""):
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
        import time
        time.sleep(float(seconds))
        return None

    def _builtin_now(self):
        import datetime
        return datetime.datetime.now().isoformat()

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
            for child in p.iterdir():
                if child.is_file():
                    child.unlink()
            p.rmdir()
        elif p.exists():
            p.unlink()
        return True

    def _builtin_copy(self, src, dst):
        import shutil
        shutil.copy2(src, dst)
        return True

    def _builtin_move(self, src, dst):
        import shutil
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

    def run(self):
        self._execute_block(0, len(self.lines), local_env=self.env)
        return self.exit_code

    def _find_block_end(self, start_idx: int) -> int:
        depth = 0
        for i in range(start_idx, len(self.lines)):
            line = self.lines[i].strip()
            if re.match(r"^(if|while|function|try)\b", line):
                depth += 1
            elif line == "end":
                depth -= 1
                if depth == 0:
                    return i
        raise NexCParseError("Missing 'end' for block")

    def _find_if_chain(self, start_idx: int):
        """
        Returns:
            end_idx: index of matching end
            branches: list of (condition_or_None_for_else, body_start, body_end)
        """
        branches = []
        depth = 0
        i = start_idx + 1
        current_cond = self.lines[start_idx].strip()[3:].strip()
        current_body_start = start_idx + 1

        while i < len(self.lines):
            line = self.lines[i].strip()

            if re.match(r"^(if|while|function|try)", line):
                depth += 1
            elif line == "end":
                if depth == 0:
                    branches.append((current_cond, current_body_start, i))
                    return i, branches
                depth -= 1
            elif depth == 0 and line.startswith("elseif "):
                branches.append((current_cond, current_body_start, i))
                current_cond = line[7:].strip()
                current_body_start = i + 1
            elif depth == 0 and line == "else":
                branches.append((current_cond, current_body_start, i))
                current_cond = None
                current_body_start = i + 1

            i += 1

        raise NexCParseError("Missing 'end' for if")

    def _parse_function_def(self, header: str, body: List[str]):
        m = re.match(r"^(?:function|fn)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", header)
        if not m:
            raise NexCParseError(f"Invalid function declaration: {header}")
        name = m.group(1)
        params = [p.strip() for p in split_args(m.group(2)) if p.strip()]
        self.functions[name] = FunctionDef(params=params, body=body)

    def _call_user_function(self, name: str, args: List[Any]) -> Any:
        fn = self.functions.get(name)
        if not fn:
            raise NexCRuntimeError(f"Unknown function: {name}")
        if len(args) != len(fn.params):
            raise NexCRuntimeError(f"{name} expects {len(fn.params)} args, got {len(args)}")
        local = dict(self.env)
        for p, a in zip(fn.params, args):
            local[p] = a
        try:
            return self._execute_block_from_lines(fn.body, local_env=local)
        except _ReturnSignal as r:
            return r.value

    def _execute_block(self, start: int, end: int, local_env: Dict[str, Any]):
        i = start
        while i < end:
            line = self.lines[i].strip()

            if line.startswith("let "):
                self._handle_let(line, local_env)
                i += 1
                continue

            if line.startswith("function ") or line.startswith("fn "):
                block_end = self._find_block_end(i)
                body = self.lines[i + 1:block_end]
                self._parse_function_def(line, body)
                i = block_end + 1
                continue

            if line.startswith("if "):
                end_idx, branches = self._find_if_chain(i)
                for cond, body_start, body_end in branches:
                    if cond is None:
                        self._execute_block(body_start, body_end, local_env)
                        break
                    if self._truthy(safe_eval(cond, local_env)):
                        self._execute_block(body_start, body_end, local_env)
                        break
                i = end_idx + 1
                continue

            if line.startswith("while "):

                block_end = self._find_block_end(i)
                cond = line[6:].strip()
                body = self.lines[i + 1:block_end]
                for _ in range(MAX_RECURSION):
                    if not self._truthy(safe_eval(cond, local_env)):
                        break
                    self._execute_block_from_lines(body, local_env)
                else:
                    raise NexCRuntimeError("while loop exceeded maximum iterations")
                i = block_end + 1
                continue

            if line.startswith("try"):
                # try ... catch ... end
                block_end = self._find_block_end(i)
                try_body, catch_body = self._split_try_catch(i + 1, block_end)
                try:
                    self._execute_block_from_lines(try_body, local_env)
                except Exception as e:
                    if catch_body is None:
                        raise
                    local_env["_error"] = str(e)
                    self._execute_block_from_lines(catch_body, local_env)
                i = block_end + 1
                continue

            if line == "end" or line == "else" or line.startswith("elseif "):
                # handled by block parser
                return None

            if line.startswith("return"):
                expr = line[6:].strip()
                value = None if not expr else safe_eval(expr, local_env)
                raise _ReturnSignal(value)

            if "=" in line and not re.match(r"^(==|!=|>=|<=|=>|<=)\b", line):
                self._handle_assignment(line, local_env)
                i += 1
                continue

            self._handle_expression_statement(line, local_env)
            i += 1
        return None

    def _execute_block_from_lines(self, lines: List[str], local_env: Dict[str, Any]):
        saved = self.lines
        try:
            self.lines = lines
            return self._execute_block(0, len(lines), local_env)
        finally:
            self.lines = saved

    def _split_try_catch(self, start: int, block_end: int):
        depth = 0
        catch_idx = None
        for i in range(start, block_end):
            line = self.lines[i].strip()
            if re.match(r"^(if|while|function|try)\b", line):
                depth += 1
            elif line == "end":
                depth -= 1
            elif depth == 0 and line.startswith("catch"):
                catch_idx = i
                break
        if catch_idx is None:
            return self.lines[start:block_end], None
        return self.lines[start:catch_idx], self.lines[catch_idx + 1:block_end]

    def _handle_let(self, line: str, local_env: Dict[str, Any]):
        m = re.match(r"^let\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$", line)
        if not m:
            raise NexCParseError(f"Invalid let statement: {line}")
        name, expr = m.group(1), m.group(2)
        local_env[name] = safe_eval(expr, local_env)

    def _handle_assignment(self, line: str, local_env: Dict[str, Any]):
        if line.startswith("let "):
            return
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)(\s*\[[^\]]+\])?\s*=\s*(.+)$", line)
        if not m:
            raise NexCParseError(f"Invalid assignment: {line}")
        name, index_part, expr = m.group(1), m.group(2), m.group(3)
        value = safe_eval(expr, local_env)
        if index_part:
            idx_expr = index_part.strip()[1:-1]
            target = local_env.get(name)
            if target is None:
                raise NexCRuntimeError(f"Unknown variable: {name}")
            idx = safe_eval(idx_expr, local_env)
            target[idx] = value
        else:
            local_env[name] = value

    def _handle_expression_statement(self, line: str, local_env: Dict[str, Any]):
        # function call or standalone expression
        if "(" in line and line.endswith(")"):
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", line)
            if m:
                name = m.group(1)
                arglist = m.group(2).strip()
                args = []
                if arglist:
                    args = [safe_eval(a, local_env) for a in split_args(arglist)]
                if name in self.functions:
                    self._call_user_function(name, args)
                    return
                if name in local_env and callable(local_env[name]):
                    local_env[name](*args)
                    return
        # raw expression
        safe_eval(line, local_env)

    def _truthy(self, value: Any) -> bool:
        return bool(value)


class _ReturnSignal(Exception):
    def __init__(self, value):
        self.value = value


def repl():
    print("NexC v1.0 REPL. Type 'exit(0)' or Ctrl+C to quit.")
    interp = Interpreter("")
    while True:
        try:
            line = input("nexc> ")
        except EOFError:
            break
        if not line.strip():
            continue
        try:
            interp.lines = preprocess(line)
            interp._execute_block(0, len(interp.lines), interp.env)
        except SystemExit:
            break
        except Exception as e:
            print(f"[error] {e}")


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    if len(argv) < 2:
        repl()
        return 0
    if argv[1] in ("-h", "--help"):
        print("Usage: python nexc.py <script.nxc> [args...]")
        return 0
    script = Path(argv[1])
    if not script.exists():
        print(f"File not found: {script}", file=sys.stderr)
        return 1
    code = script.read_text(encoding="utf-8")
    interp = Interpreter(code, argv=argv[2:])
    try:
        return interp.run()
    except _ReturnSignal as r:
        return 0
    except SystemExit as e:
        return int(e.code or 0)
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
