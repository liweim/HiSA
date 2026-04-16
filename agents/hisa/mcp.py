#!/usr/bin/env python3
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple


class MCPRegistry:
    """Lightweight software-level MCP registry backed by copied domain schemas/handlers."""

    DOMAIN_ALIASES = {
        "chrome": "google_chrome",
        "google_chrome": "google_chrome",
        "vs_code": "code",
        "vscode": "code",
        "code": "code",
        "libreoffice_calc": "libreoffice_calc",
        "calc": "libreoffice_calc",
        "excel": "libreoffice_calc",
        "libreoffice_writer": "libreoffice_writer",
        "writer": "libreoffice_writer",
        "libreoffice_impress": "libreoffice_impress",
        "impress": "libreoffice_impress",
        "powerpoint": "libreoffice_impress",
        "vlc": "vlc",
    }

    HANDLER_CLASS = {
        "google_chrome": "BrowserTools",
        "code": "CodeTools",
        "libreoffice_calc": "CalcTools",
        "libreoffice_writer": "WriterTools",
        "libreoffice_impress": "ImpressTools",
        "vlc": "VLCTools",
    }

    CHROME_SETUP_URLS = {
        "open_profile_settings": "chrome://settings/people",
        "open_password_settings": "chrome://settings/autofill",
        "open_privacy_settings": "chrome://settings/privacy",
        "open_appearance_settings": "chrome://settings/appearance",
        "open_search_engine_settings": "chrome://settings/search",
        "open_extensions": "chrome://extensions",
        "open_bookmarks": "chrome://bookmarks",
    }

    CHROME_GUI_ACTIONS = {
        "bring_back_last_tab": "pyautogui.hotkey('ctrl', 'shift', 't')",
        "print": "pyautogui.hotkey('ctrl', 'p')",
        "delete_browsing_data": "pyautogui.hotkey('ctrl', 'shift', 'delete')",
        "bookmark_page": "pyautogui.hotkey('ctrl', 'd')",
    }

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = base_dir or os.path.dirname(__file__)
        self.schema_dir = os.path.join(self.base_dir, "mcp_schemas")
        self.handler_dir = os.path.join(self.base_dir, "mcp_handlers")
        self._schema_cache: Dict[str, List[Dict[str, Any]]] = {}

    def normalize_domain(self, domain: str) -> str:
        key = re.sub(r"[^a-z0-9_]+", "_", str(domain or "").strip().lower())
        key = re.sub(r"_+", "_", key).strip("_")
        return self.DOMAIN_ALIASES.get(key, key)

    def _schema_path(self, domain: str) -> str:
        return os.path.join(self.schema_dir, f"{domain}.json")

    def _handler_path(self, domain: str) -> str:
        return os.path.join(self.handler_dir, f"{domain}.py")

    def has_domain(self, domain: str) -> bool:
        normalized = self.normalize_domain(domain)
        return os.path.exists(self._schema_path(normalized))

    def get_domain_tools(self, domain: str) -> List[Dict[str, Any]]:
        normalized = self.normalize_domain(domain)
        if not normalized:
            return []
        if normalized not in self._schema_cache:
            path = self._schema_path(normalized)
            if not os.path.exists(path):
                self._schema_cache[normalized] = []
            else:
                with open(path, "r", encoding="utf-8") as f:
                    self._schema_cache[normalized] = json.load(f)
        return self._schema_cache.get(normalized, [])

    def render_prompt(self, domains: List[str]) -> str:
        sections: List[str] = []
        seen = set()
        for domain in domains:
            normalized = self.normalize_domain(domain)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            tools = self.get_domain_tools(normalized)
            if not tools:
                continue
            lines = [f"## mcp tools for `{normalized}`"]
            lines.append("Use `tool: \"mcp\"` when a software-level API can express the step more directly than GUI or bash.")
            lines.append("Choose `mcp` only for one concrete API call per step.")
            for item in tools:
                function = item.get("function", {})
                name = function.get("name", "")
                description = function.get("description", "")
                params = function.get("parameters", {}).get("properties", {}) or {}
                required = function.get("parameters", {}).get("required", []) or []
                if params:
                    param_parts = []
                    for param_name, meta in params.items():
                        marker = "required" if param_name in required else "optional"
                        param_parts.append(f"{param_name} ({meta.get('type', 'any')}, {marker})")
                    param_text = ", ".join(param_parts)
                else:
                    param_text = "no arguments"
                lines.append(f"- `{name}`: {description} Parameters: {param_text}.")
            sections.append("\n".join(lines))
        return "\n\n".join(section for section in sections if section)

    def invoke(
        self,
        domain: str,
        name: str,
        positional_arguments: Optional[List[Any]],
        arguments: Optional[Dict[str, Any]],
        env,
    ) -> Dict[str, Any]:
        positional_arguments = list(positional_arguments or [])
        arguments = dict(arguments or {})
        normalized = self.normalize_domain(domain)
        if not self.has_domain(normalized):
            raise ValueError(f"No mcp schema registered for domain={domain}")

        class_name, method_name = self._split_call_name(normalized, name)
        arguments = self._normalize_call_arguments(
            normalized,
            class_name,
            method_name,
            positional_arguments,
            arguments,
        )
        if normalized == "google_chrome":
            return self._invoke_chrome(method_name, arguments)
        return self._invoke_via_vm(normalized, class_name, method_name, arguments, env)

    def _split_call_name(self, normalized_domain: str, name: str) -> tuple[str, str]:
        raw_name = str(name or "").strip()
        if not raw_name:
            raise ValueError("mcp call name cannot be empty")
        if "." in raw_name:
            class_name, method_name = raw_name.split(".", 1)
            return class_name, method_name
        class_name = self.HANDLER_CLASS.get(normalized_domain)
        if not class_name:
            raise ValueError(f"Cannot infer handler class for domain={normalized_domain}")
        return class_name, raw_name

    def _invoke_chrome(self, method_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if arguments:
            raise ValueError(f"Chrome mcp tool `{method_name}` does not accept arguments")
        if method_name in self.CHROME_SETUP_URLS:
            url = self.CHROME_SETUP_URLS[method_name]
            return {
                "execution_mode": "setup",
                "payload": [{"type": "chrome_open_tabs", "parameters": {"urls_to_open": [url]}}],
                "observation": f"Opened Chrome page {url}.",
                "raw_result": {"action_type": "OPEN_CHROME_TAB", "parameters": {"urls_to_open": [url]}},
            }
        if method_name in self.CHROME_GUI_ACTIONS:
            code = self.CHROME_GUI_ACTIONS[method_name]
            return {
                "execution_mode": "gui",
                "payload": code,
                "observation": f"Executed Chrome shortcut `{method_name}`.",
                "raw_result": code,
            }
        raise ValueError(f"Unsupported Chrome mcp tool: {method_name}")

    def _invoke_via_vm(
        self,
        normalized_domain: str,
        class_name: str,
        method_name: str,
        arguments: Dict[str, Any],
        env,
    ) -> Dict[str, Any]:
        handler_path = self._handler_path(normalized_domain)
        if not os.path.exists(handler_path):
            raise ValueError(f"Missing mcp handler source for domain={normalized_domain}")

        with open(handler_path, "r", encoding="utf-8") as f:
            handler_source = self._strip_main_block(f.read())

        call_payload = {
            "class_name": class_name,
            "method_name": method_name,
            "arguments": arguments,
            "domain": normalized_domain,
        }
        wrapper = (
            "import json\n"
            "import socket\n"
            "import subprocess\n"
            "import time\n"
            "import traceback\n\n"
            f"_CALL = json.loads({json.dumps(json.dumps(call_payload))})\n"
            f"_HANDLER_SOURCE = {json.dumps(handler_source)}\n"
            "_class_name = _CALL['class_name']\n"
            "_method_name = _CALL['method_name']\n"
            "_arguments = _CALL.get('arguments', {}) or {}\n"
            "_domain = _CALL.get('domain', '')\n"
            "\n"
            "def _wait_for_port(host, port, timeout=10.0):\n"
            "    deadline = time.time() + max(0.5, float(timeout))\n"
            "    while time.time() < deadline:\n"
            "        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
            "        sock.settimeout(0.5)\n"
            "        try:\n"
            "            sock.connect((host, port))\n"
            "            return True\n"
            "        except Exception:\n"
            "            time.sleep(0.3)\n"
            "        finally:\n"
            "            try:\n"
            "                sock.close()\n"
            "            except Exception:\n"
            "                pass\n"
            "    return False\n"
            "\n"
            "def _ensure_libreoffice_listener():\n"
            "    if _wait_for_port('127.0.0.1', 2002, timeout=1.0):\n"
            "        return\n"
            "    subprocess.Popen(\n"
            "        ['soffice', '--accept=socket,host=localhost,port=2002;urp;StarOffice.Service'],\n"
            "        stdout=subprocess.DEVNULL,\n"
            "        stderr=subprocess.DEVNULL,\n"
            "    )\n"
            "    if not _wait_for_port('127.0.0.1', 2002, timeout=12.0):\n"
            "        raise RuntimeError('LibreOffice UNO listener on localhost:2002 is not available')\n"
            "\n"
            "try:\n"
            "    if _domain in {'libreoffice_calc', 'libreoffice_writer', 'libreoffice_impress'}:\n"
            "        _ensure_libreoffice_listener()\n"
            "    exec(_HANDLER_SOURCE, globals(), globals())\n"
            "    _cls = globals()[_class_name]\n"
            "    _func = getattr(_cls, _method_name)\n"
            "    _result = _func(**_arguments)\n"
            "    print(json.dumps({'ok': True, 'result': _result}, ensure_ascii=False, default=str))\n"
            "except Exception as exc:\n"
            "    print(json.dumps({'ok': False, 'error': str(exc), 'traceback': traceback.format_exc()}, ensure_ascii=False, default=str))\n"
        )

        output = env.controller.run_python_script(wrapper) or {}
        status = str(output.get("status", "") or "")
        raw_stdout = str(output.get("output", "") or output.get("message", "") or "").strip()
        if status == "error" and output.get("error"):
            raise RuntimeError(str(output.get("error")))
        parsed = self._parse_last_json_line(raw_stdout)
        if not parsed.get("ok"):
            error_text = parsed.get("error") or raw_stdout or "mcp VM execution failed"
            raise RuntimeError(error_text)
        return {
            "execution_mode": "vm",
            "payload": {
                "status": status or "success",
                "output": raw_stdout,
                "parsed": parsed,
            },
            "observation": f"{class_name}.{method_name} executed in VM.",
            "raw_result": parsed.get("result"),
        }

    def _normalize_call_arguments(
        self,
        normalized_domain: str,
        class_name: str,
        method_name: str,
        positional_arguments: List[Any],
        keyword_arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not positional_arguments:
            return keyword_arguments

        param_order, required_params = self._get_parameter_order(normalized_domain, class_name, method_name)
        if not param_order:
            raise ValueError(
                f"mcp input uses positional arguments for {class_name}.{method_name}, "
                "but no parameter schema is available to map them"
            )
        if len(positional_arguments) > len(param_order):
            raise ValueError(
                f"mcp input provides too many positional arguments for {class_name}.{method_name}: "
                f"got {len(positional_arguments)}, expected at most {len(param_order)}"
            )

        normalized_arguments = dict(keyword_arguments)
        for index, value in enumerate(positional_arguments):
            param_name = param_order[index]
            if param_name in normalized_arguments:
                raise ValueError(
                    f"mcp input assigns argument `{param_name}` both positionally and by keyword"
                )
            normalized_arguments[param_name] = value

        missing_required = [name for name in required_params if name not in normalized_arguments]
        if missing_required:
            raise ValueError(
                f"mcp input is missing required arguments for {class_name}.{method_name}: "
                + ", ".join(missing_required)
            )
        return normalized_arguments

    def _get_parameter_order(
        self,
        normalized_domain: str,
        class_name: str,
        method_name: str,
    ) -> Tuple[List[str], List[str]]:
        full_name = f"{class_name}.{method_name}"
        for item in self.get_domain_tools(normalized_domain):
            function = item.get("function", {}) or {}
            if function.get("name") != full_name:
                continue
            parameters = function.get("parameters", {}) or {}
            properties = parameters.get("properties", {}) or {}
            required = parameters.get("required", []) or []
            return list(properties.keys()), list(required)
        return [], []

    def _parse_last_json_line(self, stdout: str) -> Dict[str, Any]:
        lines = [line.strip() for line in str(stdout or "").splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                parsed = json.loads(line)
            except Exception:
                continue
            if isinstance(parsed, dict) and "ok" in parsed:
                return parsed
        return {"ok": False, "error": stdout or "No structured mcp output"}

    def _strip_main_block(self, source: str) -> str:
        marker = '\nif __name__ == "__main__":'
        if marker in source:
            return source.split(marker, 1)[0].rstrip() + "\n"
        return source
