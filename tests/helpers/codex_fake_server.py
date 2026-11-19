"""Offline App Server process double; scenarios are selected by input text."""

import json
import os
import sys
from pathlib import Path


def send(message):
    print(json.dumps(message), flush=True)


def notify(method, **params):
    send({"method": method, "params": params})


version = Path(__file__).with_suffix(".version").read_text().strip()
if "--version" in sys.argv:
    print(f"codex-cli {version}")
    sys.exit(0)

assert "OPENAI_API_KEY" not in os.environ
assert "CODEX_ACCESS_TOKEN" not in os.environ
assert "CORTEXMUX_TEST_ENDPOINT" not in os.environ
assert sys.argv[1:4] == ["app-server", "--listen", "stdio://"]
overrides = {
    value for option, value in zip(sys.argv[4::2], sys.argv[5::2], strict=True) if option == "-c"
}
assert {
    'approval_policy="never"',
    'approvals_reviewer="user"',
    'sandbox_mode="read-only"',
    'web_search="disabled"',
    'cli_auth_credentials_store="file"',
    'forced_login_method="chatgpt"',
    'model_provider="openai"',
    "mcp_servers={}",
    "features.shell_tool=false",
    "features.unified_exec=false",
    "features.plugins=false",
    "features.remote_plugin=false",
    "features.multi_agent=false",
} <= overrides
if version == "0.170.0":
    assert {
        "features.browser_use=false",
        "features.computer_use=false",
        "features.image_generation=false",
        "features.multi_agent_v2=false",
    } <= overrides
pending_login = (Path(os.environ["CODEX_HOME"]) / "fake-pending-login").exists()
connected = True
thread_count = 0
turn_count = 0
log = Path(os.environ["CODEX_HOME"]) / "fake-requests.jsonl"
for line in sys.stdin:
    message = json.loads(line)
    with log.open("a") as handle:
        handle.write(json.dumps(message) + "\n")
    method = message.get("method")
    if "id" not in message or method is None:
        continue
    request_id = message["id"]
    params = message.get("params", {})
    result = {}
    if method == "account/read":
        result = {
            "account": {
                "type": "chatgpt",
                "email": "test@example.invalid",
                "planType": "plus",
                "accessToken": "SECRET",
            }
            if connected
            else None
        }
        if version == "0.170.0":
            result["userId"] = "SECRET"
    elif method == "account/login/start":
        if not pending_login:
            notify("account/login/completed", loginId="login-1", success=True, error="SECRET")
        connected = True
        result = {
            "type": params["type"],
            "loginId": "login-1",
            "authUrl": "https://auth.openai.com/example",
            "verificationUrl": "https://auth.openai.com/codex/device",
            "userCode": "TEST",
        }
    elif method == "account/login/cancel":
        notify("account/login/completed", loginId=params["loginId"], success=False, error="SECRET")
    elif method == "account/logout":
        connected = False
    elif method == "account/rateLimits/read":
        result = {"rateLimits": {"primary": {"usedPercent": 25}, "accessToken": "SECRET"}}
    elif method == "model/list":
        result = {
            "data": [
                {"model": "test-model", "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}
            ]
        }
        catalog = Path(os.environ["CODEX_HOME"]) / "fake-model-catalog.json"
        if catalog.exists():
            rows = json.loads(catalog.read_text())
            if not params.get("includeHidden", False):
                rows = [row for row in rows if not row.get("hidden", False)]
            offset = int(params.get("cursor") or 0)
            result = {
                "data": rows[offset : offset + 1],
                "nextCursor": str(offset + 1) if offset + 1 < len(rows) else None,
            }
    elif method == "thread/start":
        assert params["approvalPolicy"] == "never"
        assert params["approvalsReviewer"] == "user"
        assert params["sandbox"] == "read-only"
        assert params["modelProvider"] == "openai"
        assert Path(params["cwd"]).resolve() == Path.cwd()
        assert Path(os.environ["CODEX_HOME"]) != Path.cwd()
        thread_count += 1
        result = {
            "thread": {"id": f"thread-{thread_count}"},
            "model": params["model"],
            "modelProvider": "openai",
            "sandbox": {"type": "readOnly", "networkAccess": False},
            "approvalPolicy": "never",
            "approvalsReviewer": "user",
            "instructionSources": [],
        }
        receipt = Path(os.environ["CODEX_HOME"]) / "fake-thread-receipt"
        if receipt.exists():
            match receipt.read_text().strip():
                case "substituted":
                    result["model"] = "fallback"
                case "missing":
                    del result["model"]
                case "network-enabled":
                    result["sandbox"]["networkAccess"] = True
    elif method == "turn/start":
        assert params["approvalPolicy"] == "never"
        assert params["approvalsReviewer"] == "user"
        assert params["sandboxPolicy"] == {"type": "readOnly", "networkAccess": False}
        turn_count += 1
        thread_id, turn_id = params["threadId"], f"turn-{turn_count}"
        prompt = params["input"][0]["text"]
        if prompt == "unknown-outcome":
            continue
        result = {"turn": {"id": turn_id, "status": "inProgress"}}
        notify("turn/started", threadId=thread_id, turn=result["turn"])
        if prompt == "hang":
            send({"id": request_id, "result": result})
            continue
        if prompt == "crash":
            sys.exit(1)
        if prompt == "malformed":
            print("NOT JSON SECRET", flush=True)
            continue
        if prompt == "permission":
            send(
                {
                    "id": "server-request",
                    "method": "item/commandExecution/requestApproval",
                    "params": {"threadId": thread_id, "turnId": turn_id, "command": "SECRET"},
                }
            )
        notify(
            "item/reasoning/textDelta",
            threadId=thread_id,
            turnId=turn_id,
            delta="PRIVATE REASONING",
        )
        notify(
            "item/completed",
            threadId="other",
            turnId="other",
            item={"type": "agentMessage", "id": "wrong", "text": "WRONG"},
        )
        if prompt in {"rerouted", "rerouted-missing", "other-turn-rerouted"}:
            rerouted = {
                "threadId": thread_id,
                "turnId": "other" if prompt == "other-turn-rerouted" else turn_id,
                "fromModel": params["model"],
                "toModel": "fallback",
                "reason": "highRiskCyberActivity",
            }
            if prompt == "rerouted-missing":
                del rerouted["toModel"]
            notify("model/rerouted", **rerouted)
        if version == "0.170.0" and "outputSchema" in params:
            partial = {
                "type": "agentMessage",
                "id": "partial",
                "phase": "partial_answer",
                "text": "Draft",
            }
            notify("item/started", threadId=thread_id, turnId=turn_id, item=partial)
            notify(
                "item/agentMessage/delta",
                threadId=thread_id,
                turnId=turn_id,
                itemId="partial",
                delta="Draft",
            )
            notify("item/completed", threadId=thread_id, turnId=turn_id, item=partial)
        text = '{"answer":42}' if "outputSchema" in params else "answer"
        if prompt == "bad-json":
            text = '{"answer":"wrong"}'
        if prompt == "empty":
            text = " "
        item = {"type": "agentMessage", "id": "answer", "phase": "final_answer", "text": text}
        notify("item/started", threadId=thread_id, turnId=turn_id, item=item)
        notify(
            "item/agentMessage/delta",
            threadId=thread_id,
            turnId=turn_id,
            itemId="answer",
            delta=text,
        )
        if prompt == "partial-hang":
            send({"id": request_id, "result": result})
            continue
        notify("item/completed", threadId=thread_id, turnId=turn_id, item=item)
        status = "interrupted" if prompt == "interrupted" else "completed"
        turn = {"id": turn_id, "status": status}
        if prompt in {"quota", "rate-limit", "model-not-found"}:
            turn = {
                "id": turn_id,
                "status": "failed",
                "error": {
                    "codexErrorInfo": {
                        "quota": "usageLimitExceeded",
                        "rate-limit": "rateLimitExceeded",
                        "model-not-found": "modelNotFound",
                    }[prompt],
                    "message": "SECRET",
                },
            }
        notify("turn/completed", threadId=thread_id, turn=turn)
    elif method == "turn/interrupt":
        notify(
            "turn/completed",
            threadId=params["threadId"],
            turn={"id": params["turnId"], "status": "interrupted"},
        )
    elif method == "test/error":
        send({"id": request_id, "error": {"code": -32602, "message": "SECRET"}})
        continue
    send({"id": request_id, "result": result})
