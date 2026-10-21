# Using the engine behind a LiteLLM proxy

This page describes how to put the engine behind a LiteLLM proxy that runs in a container on the same host, which is how a nightly batch pipeline called it in the records listed in [measurements](measurements.md). The engine speaks the OpenAI Chat Completions dialect ([server-api](server-api.md)), so LiteLLM treats it as a generic OpenAI-compatible backend. Nothing here installs anything; each host change is a step the owner of the host runs.

The LiteLLM side of this page (key names in its config file, how it reads its timeout) comes from LiteLLM's own documentation, not from this repository. Check those keys against the LiteLLM version you run. The engine side is tested in this repository.

## What you need first

- The engine installed and answering on `127.0.0.1:18434` ([install-spark](install-spark.md)).
- A proxy container on the same host as the engine. The engine never listens on a LAN address, so a proxy on another machine needs an ssh tunnel instead (see [server-api](server-api.md)).
- A decision on who may start the engine. The front starts it on demand, subject to the start policy in [memory-modes](memory-modes.md); on a shared host that policy can refuse a request.

## Reaching the front from a container

The front binds loopback only. A container cannot reach loopback of the host, so add a second address, the container network gateway, to the front socket:

1. Copy `deploy/slotstream-spark.socket.d/extra-bind.conf.example` to the user systemd directory as `slotstream-spark.socket.d/extra-bind.conf`, replace `<gateway-ip>` with the gateway address of the network the proxy container uses (`docker network inspect <network>` shows it under `IPAM.Gateway`), run `systemctl --user daemon-reload` and restart `slotstream-spark.socket`. The steps are in the RUNBOOK ("Containers on the same host").
2. Set `SLOTSTREAM_API_KEY` in the front's environment file. A request that arrives on the extra address then needs `Authorization: Bearer <key>`, and gets 401 without it. Loopback stays open. Without a key, every container on the host can call the front.
3. In the proxy's compose file, make the host name resolve to that gateway. With Docker this is `extra_hosts: ["host.docker.internal:host-gateway"]` on the proxy service; `host-gateway` resolves to the host's gateway address on the default bridge. Check that it equals the address you put in the drop-in, because a user-defined network has a different gateway.

Only the front port (18434) gets the extra address. The engine port (18435) is never exposed.

## The model id

LiteLLM needs the id the server reports, not an alias of your choosing. Ask the front:

```
curl -s -H "Authorization: Bearer $SLOTSTREAM_API_KEY" http://host.docker.internal:18434/v1/models
```

The first request on a stopped engine starts it, so this call may return 503 with `Retry-After` first; see "The front wait" below. Use the `id` of the answer as the model name in the proxy. The engine accepts the `model` field as sent, but the id from `/v1/models` is the served one.

## A proxy entry

In the proxy's model list, register the engine as an OpenAI-compatible model. A sketch, with placeholders in angle brackets:

```
model_list:
  - model_name: <name your clients use>
    litellm_params:
      model: openai/<id from /v1/models>
      api_base: http://host.docker.internal:18434/v1
      api_key: os.environ/SLOTSTREAM_API_KEY
      timeout: 300
```

`timeout` is the proxy's per-attempt limit in seconds. The pipeline in the records used 300 s per attempt. A warm call of about 16,000 prompt tokens finished in 64.778 s in the recorded profile ([workload-profile](../db/records/measurements/workload-profile.md)), so 300 s leaves room; it does not leave room for a long cold start plus a long call if the proxy counts the wait for the engine inside the same attempt.

## One request at a time

The engine runs one generation at a time. Up to 4 more requests wait in order, and a fifth waiting request gets 503 `queue_full` with `Retry-After: 1` ([server-api](server-api.md)). Keep the proxy's concurrency for this model at 1, or at most 5 in flight, so the queue never overflows. A batch client that sends calls one after another needs no setting.

Streaming is optional. The pipeline in the records sent non-streaming calls with no `temperature`, and every response in the profile finished with reason `stop`.

## `json_object` mode

The engine accepts `response_format` of type `text` or `json_object` and rejects other types with 400. It has no grammar-constrained decoding. For `json_object` it turns thinking off and prepends one instruction to the messages: "Answer with a single valid JSON object and nothing else, with no code fence." (`slotstream_spark/server/dialects/openai_chat.py`, `prepare`, and `slotstream_spark/server/common.py`).

What this means for a client:

- The output is JSON because the model was told to write JSON, not because the decoder can only emit JSON. Parse it and handle a failure.
- A call that runs into `max_tokens` can end in the middle of an object. Check `finish_reason` before parsing.
- A client's own `reasoning_effort` or thinking request is overridden for that call.
- The recorded profile checked status 200 and a finished response for its two `json_object` calls and did not parse their JSON, so this repository publishes no JSON validity rate. The only count is 3 of 3 calls with status 200 in the profile.

## The front wait

The front holds a request while the engine loads, up to `SLOTSTREAM_FRONT_WAIT_S` seconds (default 240), then relays it. After the wait it answers 503 with `Retry-After: 10`. Set the value to 0 to get the 503 at once. Two settings interact:

- The proxy's attempt timeout counts from the proxy's request. With the default 240 s wait and a proxy timeout of 300 s, a cold start fits in one attempt.
- A start the policy refuses is answered with 503 at once, regardless of the wait. Retrying that request changes nothing until the reason (a router with a model loaded, a window, free memory below the floor) goes away.

The cold start measured in the recorded profile was 20.137 s, in 1 attempt, with the page cache probably warm. A cold page cache after a reboot loads slower; that case was not measured.

## Controlling the engine around a window

For a nightly batch, the sequence that the recorded rehearsal followed was:

1. Before the window, stop whatever else holds the memory the engine needs. On the host of the rehearsal that was a set of resident services, stopped by a script of the host's owner.
2. Start the engine by sending one request through the front. A request for `/v1/models` is enough; a chat request with `max_tokens` 1 also warms the path. Wait until it answers 200. In the rehearsal the engine was ready 17 s after each start ([workload-rehearsal](../db/records/measurements/workload-rehearsal.md)).
3. Run the batch.
4. After the window, stop the engine. It exits by itself after 1800 s without a request (`SLOTSTREAM_IDLE_EXIT_S`). To stop it sooner, stop the service chain with `systemctl --user stop slotstream-spark.socket slotstream-spark-front.service slotstream-spark.service`.
5. Wait for the memory to return before restarting the other services. Read `MemAvailable` from `/proc/meminfo` and continue when it is back above the level your services need. In the rehearsal `MemAvailable` was 120.1 GB with the residents and the engine stopped and 87.1 GB with the residents up and the engine stopped.

The rehearsal ran one night. In it the engine and the resident services never ran together, so it shows a clean handover (no oom-kill, 14 of 14 containers up afterwards) and nothing about running both at once.

## Failure modes seen or documented

| Symptom | Cause | What to do |
|---|---|---|
| 503 with `Retry-After: 10` | the engine is loading and the front wait ran out, or it is down | retry after the delay; raise `SLOTSTREAM_FRONT_WAIT_S` or the proxy retry count |
| 503, immediate, no `Retry-After` wait helps | the start policy refused (router busy, outside the window, memory below the floor) | fix the reason; check `journalctl --user -u slotstream-spark-front.service` |
| 503 `queue_full` | more than 5 requests in flight | lower the proxy concurrency for this model |
| 503 `insufficient_memory` | the request memory budget is used up | send fewer large requests at once |
| 401 | the key is missing or wrong on the extra address | check `SLOTSTREAM_API_KEY` in both places |
| 400 `invalid_image` | an image was sent; the server is text only | remove the image |
| 400 on `response_format` | a type other than `text` or `json_object` | use one of those two |
| 413 `request_too_large` | the body is over 32 MiB | split the input |
| timeout at the proxy | the call plus the engine start passed the attempt limit | raise the proxy timeout or start the engine before the batch |

## What the records say about this setup

The comparison of October 17, 2026 ran the same stored day through gpt-oss-120b served by vLLM and through the engine, with 13 report sections. The engine finished with no failed reading segment, and the record calls the comparison inconclusive at that size ([workload-comparison](../db/records/measurements/workload-comparison.md)). Nothing on this page claims the engine is a better backend for that pipeline. Three more nights are expected to add data, and the figures here will be updated when they do.
