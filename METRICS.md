# Metrics

Custom Prometheus metrics for the studio, served at `/metrics` on the app port
(defined in `studio/metrics.py`). The chart's ServiceMonitor feeds OpenShift
user-workload monitoring; the pod's `prometheus.io/*` annotations are the
fallback for annotation-based scrapers. `LAYA_METRICS=0` in `studio.env`
disables updates and serves 404 from `/metrics`.

## Metric reference

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `laya_studio_http_requests_total` | counter | `method, endpoint, status` | every HTTP request answered |
| `laya_studio_http_request_duration_seconds` | histogram | `method, endpoint` | whole-request wall time |
| `laya_studio_inference_seconds` | histogram | `checkpoint` | forward-pass wall time behind the asyncio gate |
| `laya_studio_inference_input_tokens_total` | counter | `checkpoint` | input tokens fed to the checkpoints |
| `laya_studio_inference_output_tokens_total` | counter | `checkpoint` | output tokens (no generation: usually 0) |
| `laya_studio_decisions_total` | counter | `checkpoint` | decision results produced |
| `laya_studio_inference_in_flight` | gauge | — | forward passes currently running (the gate allows one) |

The `endpoint` label is the fixed API path (`/v1/systemone`, `/api/ready`, …),
`unmatched` for unknown `/api/` and `/v1/` paths, `static` for the studio UI —
never raw paths, so label cardinality stays bounded however much is scraped.
The `checkpoint` label is the checkpoint that actually answered
(`routing.model`), so Auto traffic is attributed to the checkpoint the Router
picked.

## Queries

### laya_studio_http_requests_total

```promql
# request rate, total
sum(rate(laya_studio_http_requests_total[5m]))

# by endpoint (systemone = your rg1 calls)
sum by (endpoint) (rate(laya_studio_http_requests_total[5m]))

# error rate (5xx)
sum(rate(laya_studio_http_requests_total{status=~"5.."}[5m]))
  /
sum(rate(laya_studio_http_requests_total[5m]))

# error rate when no 5xx series exists yet (no data → 0)
(sum(rate(laya_studio_http_requests_total{status=~"5.."}[5m])) or vector(0))
  /
sum(rate(laya_studio_http_requests_total[5m]))

# rejections (admission busy); /api/ready also 503s while checkpoints load —
# add endpoint!="/api/ready" to count only admission rejections
sum(rate(laya_studio_http_requests_total{status="503"}[5m]))

# contract rejections (unknown model, malformed questions, oversized bodies)
sum(rate(laya_studio_http_requests_total{status="422"}[5m]))
```

### laya_studio_http_request_duration_seconds (histogram)

```promql
# p50 / p95 / p99 — swap the quantile; per endpoint: sum by (le, endpoint)
histogram_quantile(0.95, sum by (le) (rate(laya_studio_http_request_duration_seconds_bucket[5m])))

# mean
sum(rate(laya_studio_http_request_duration_seconds_sum[5m]))
  /
sum(rate(laya_studio_http_request_duration_seconds_count[5m]))
```

### laya_studio_inference_seconds (per checkpoint)

```promql
# forward-pass p95 by checkpoint
histogram_quantile(0.95, sum by (le, checkpoint) (rate(laya_studio_inference_seconds_bucket[5m])))
```

### laya_studio_decisions_total

```promql
# decisions/sec total and per model
sum(rate(laya_studio_decisions_total[5m]))
sum by (checkpoint) (rate(laya_studio_decisions_total[5m]))
```

### laya_studio_inference_input_tokens_total / output_tokens_total

```promql
# token burn rate (what rg1's --budget is counting against)
sum(rate(laya_studio_inference_input_tokens_total[5m]))
sum by (checkpoint) (rate(laya_studio_inference_input_tokens_total[5m]))

# daily total
sum(increase(laya_studio_inference_input_tokens_total[24h]))

# output tokens: no generation, effectively always 0 — a change here is a signal
sum(rate(laya_studio_inference_output_tokens_total[5m]))
```

### laya_studio_inference_in_flight (gauge)

```promql
# gate allows one; >0 sustained means you're saturated
# with replicas > 1: max by (pod) (max_over_time(...[5m]))
max_over_time(laya_studio_inference_in_flight[5m])
```

## Platform (work regardless of the app module)

```promql
# CPU per pod
sum by (pod) (rate(container_cpu_usage_seconds_total{namespace="muh-namespace", pod=~"laya-.*"}[5m]))

# memory working set per pod
max by (pod) (container_memory_working_set_bytes{namespace="muh-namespace", pod=~"laya-.*"})

# restarts per hour
sum(increase(kube_pod_container_status_restarts_total{namespace="muh-namespace", pod=~"laya-.*"}[1h]))
```

Swap `muh-namespace` for the release namespace (`laya-studio` if installed
with the README command); `pod=~"laya-.*"` matches the deployment name, so it
holds as long as the release name starts with `laya-`. Add
`container!="", container!="POD"` to the `container_*` selectors if a pod's
pause/aggregate series ever double-counts.

## Caveats

- **A status series only exists once that status has been recorded at least
  once.** Before the first 5xx (or 503, or 422) the selector matches nothing
  and the query returns *no data*, not 0 — the `or vector(0)` variant above
  hardens the error rate; the same applies to the 503/422 rates.
- **`increase(...[24h])` is an estimate**: Prometheus extrapolates over the
  window edges, so treat it as a rough daily total, not an exact invoice.
- **Quantiles are bucket estimates**: precision is bounded by the histogram
  buckets (2.5 ms → 45 s), so p99 can only land on a bucket edge.
- **Batch pass duration** is observed once per distinct checkpoint a batch
  answered, so a cross-checkpoint batch counts once per checkpoint and
  `laya_studio_inference_seconds_count` can slightly exceed the number of
  forward passes. Tokens and decisions are exact per result.
