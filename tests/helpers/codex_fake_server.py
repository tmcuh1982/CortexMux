"""Offline App Server process double; scenarios are selected by input text."""

import json
import os
import sys
from pathlib import Path


def send(message):
    print(json.dumps(message), flush=True)


def notify(method, **params):
    send({"method": method, "params": params})


def usage_breakdown(input_tokens, output_tokens, cached=0, reasoning=0):
    return {
        "inputTokens": input_tokens,
        "outputTokens": output_tokens,
        "cachedInputTokens": cached,
        "reasoningOutputTokens": reasoning,
        "totalTokens": input_tokens + output_tokens,
    }


def notify_usage(thread_id, turn_id, total, last):
    notify(
        "thread/tokenUsage/updated",
        threadId=thread_id,
        turnId=turn_id,
        tokenUsage={"total": total, "last": last, "modelContextWindow": 128000},
    )


if "--version" in sys.argv:
    print("codex-cli 0.140.0")
    sys.exit(0)

assert "OPENAI_API_KEY" not in os.environ
assert "CODEX_ACCESS_TOKEN" not in os.environ
assert "--listen" in sys.argv
pending_login = (Path(os.environ["CODEX_HOME"]) / "fake-pending-login").exists()
connected = True
thread_count = 0
turn_count = 0
usage_snapshots = {}
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
        thread_count += 1
        result = {
            "thread": {"id": f"thread-{thread_count}"},
            "model": params["model"],
            "modelProvider": "openai",
            "sandbox": {"type": "readOnly"},
            "approvalPolicy": "never",
            "approvalsReviewer": "user",
            "instructionSources": [],
        }
    elif method == "turn/start":
        turn_count += 1
        thread_id, turn_id = params["threadId"], f"turn-{turn_count}"
        prompt = params["input"][0]["text"]
        if prompt == json.dumps([{"role": "user", "content": "usage"}]):
            prompt = "usage"
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
        if prompt in {"usage", "usage-follow-up", "usage-zero", "usage-replayed", "usage-partial"}:
            total, last, previous_turn = usage_snapshots.get(
                thread_id, (usage_breakdown(0, 0), None, None)
            )
            if last is not None:
                notify_usage(thread_id, previous_turn, total, last)
                notify_usage(thread_id, turn_id, total, last)
            if prompt == "usage-replayed":
                estimate = {**usage_breakdown(0, 0), "totalTokens": 1120}
                notify_usage(thread_id, turn_id, total, estimate)
            readings = {
                "usage": [usage_breakdown(120, 40, 20, 9)],
                "usage-follow-up": [usage_breakdown(150, 30, 25, 7), usage_breakdown(70, 11, 5, 3)],
                "usage-zero": [usage_breakdown(0, 0)],
                "usage-replayed": [],
                "usage-partial": [{"inputTokens": 0, "cachedInputTokens": 0}],
            }
            unrelated = usage_breakdown(999, 999)
            notify_usage("other", turn_id, unrelated, unrelated)
            notify_usage(thread_id, None, unrelated, unrelated)
            for last in readings[prompt]:
                total = {key: total[key] + value for key, value in last.items()}
                notify_usage(thread_id, turn_id, total, last)
                notify_usage(thread_id, turn_id, total, last)
                usage_snapshots[thread_id] = total, last, turn_id
        status = "interrupted" if prompt == "interrupted" else "completed"
        turn = {"id": turn_id, "status": status}
        if prompt == "quota":
            turn = {
                "id": turn_id,
                "status": "failed",
                "error": {"codexErrorInfo": "usageLimitExceeded", "message": "SECRET"},
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
