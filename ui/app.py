from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.getenv("PDM_API_BASE", "http://api:8000").rstrip("/")
API_KEY = os.getenv("PDM_API_KEY", "")
TIMEOUT = float(os.getenv("PDM_API_TIMEOUT_SECONDS", "5"))

SEVERITY_STYLE = {
    "NOMINAL": ("✅", "success"), "WATCH": ("👁", "info"),
    "DEGRADED": ("⚠️", "warning"), "CRITICAL": ("⛔", "error"),
    # Not a rung on the same ladder: the reading was not physically consistent,
    # so the machine was not assessed. Shown as a warning, never as "all clear".
    "DATA_QUALITY": ("🔧", "warning"),
}
TIERS = {"Low (L)": "L", "Medium (M)": "M", "High (H)": "H"}
MODE_LABEL = {"TWF": "Tool wear", "HDF": "Heat dissipation",
              "PWF": "Power envelope", "OSF": "Overstrain"}

st.set_page_config(page_title="PdM Operator Console", layout="wide",
                   initial_sidebar_state="collapsed")
st.title("CNC Machine Condition Monitor")
st.caption(
    "Checks a machine reading against its operating limits and reports near-term "
    "tool-wear risk. It reports the condition now — it does not forecast failures "
    "in advance. Advisory only: the shift supervisor makes the final call."
)

with st.expander("About this tool"):
    st.markdown("""
**What it does.** Compares each reading against the machine's specified limits for
heat dissipation, shaft power and mechanical overstrain, and estimates the chance
of tool-wear failure over a fixed horizon.

**What it does not do.** It does not predict failures ahead of time. The limit
checks report a breach at the moment it happens, so there is no lead time — keep
your normal inspection routine. A remaining-useful-life model was tested and
rejected: its accuracy did not survive validation on data it had not seen
(`docs/DECISIONS.md`, ADR-002).

**If it says CHECK INSTRUMENTATION**, the sensor readings contradict each other and
the machine was not assessed. Treat it as unchecked until the channels are verified.
""")

# --- connection status ------------------------------------------------------
with st.sidebar:
    st.subheader("Service")
    try:
        ready = requests.get(f"{API_BASE}/ready", timeout=TIMEOUT).json()
        st.success("Ready") if ready.get("ready") else st.error("Not ready")
        st.caption(ready.get("detail", ""))
        st.caption(f"Policy `{ready.get('policy_fingerprint', '?')}`")
    except requests.RequestException as exc:
        st.error("Cannot reach the API")
        st.caption(f"{API_BASE} — {type(exc).__name__}")

# --- inputs -----------------------------------------------------------------
with st.form("telemetry"):
    c1, c2, c3 = st.columns(3)
    with c1:
        machine_id = st.text_input("Machine ID", value="CNC-014")
        tier = TIERS[st.selectbox("Product quality tier", list(TIERS))]
    with c2:
        temp_air_k = st.number_input("Air temperature (K)", 290.0, 315.0, 298.2, 0.1)
        temp_process_k = st.number_input("Process temperature (K)", 295.0, 325.0, 308.7, 0.1)
    with c3:
        speed_rpm = st.number_input("Rotational speed (rpm)", 500.0, 4000.0, 1408.0, 10.0)
        torque_nm = st.number_input("Torque (Nm)", 0.0, 120.0, 46.3, 0.1)
    tool_wear_min = st.slider("Tool wear (minutes)", 0.0, 300.0, 115.0, 1.0)
    submitted = st.form_submit_button("Evaluate", type="primary", use_container_width=True)

if not submitted:
    st.stop()

payload = {"machine_id": machine_id, "product_type": tier, "temp_air_k": temp_air_k,
           "temp_process_k": temp_process_k, "speed_rpm": speed_rpm,
           "torque_nm": torque_nm, "tool_wear_min": tool_wear_min}
headers = {"x-api-key": API_KEY} if API_KEY else {}

try:
    with st.spinner("Evaluating…"):
        resp = requests.post(f"{API_BASE}/api/v1/score", json=payload,
                             headers=headers, timeout=TIMEOUT)
except requests.Timeout:
    st.error(f"The service did not respond within {TIMEOUT:.0f}s. Treat the machine as "
             "unevaluated and follow the manual check in the runbook.")
    st.stop()
except requests.RequestException as exc:
    st.error(f"Could not reach the service at {API_BASE} ({type(exc).__name__}).")
    st.stop()

if resp.status_code != 200:
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    st.error(f"**{body.get('code', resp.status_code)}** — {body.get('error', resp.text[:300])}")
    if body.get("request_id"):
        st.caption(f"Request id `{body['request_id']}` — quote this when reporting.")
    st.stop()

data = resp.json()
decision = data.get("decision", {})
severity = decision.get("severity", "UNKNOWN")
icon, style = SEVERITY_STYLE.get(severity, ("❔", "info"))

# --- verdict ----------------------------------------------------------------
st.divider()
getattr(st, style)(f"### {icon} {severity} — {decision.get('recommended_action', '')}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Urgency", decision.get("urgency", "—").replace("_", " ").title())
tl = data.get("tool_life", {})
k2.metric(f"Tool failure risk ({tl.get('horizon_minutes', 0):.0f} min)",
          f"{tl.get('p_failure_within_horizon', 0) * 100:.0f}%")
k3.metric("Expected tool life left", f"{tl.get('expected_remaining_minutes', 0):.0f} min")
delta = decision.get("expected_cost_delta", 0.0)
k4.metric("Expected cost vs. no action", f"{delta:,.0f}",
          delta=f"{-delta:,.0f} saved" if delta < 0 else None,
          help="Negative means acting on this alert is cheaper than ignoring it, "
               "under the configured cost model.")

if decision.get("triggered_by"):
    st.markdown("**Triggered by**")
    for t in decision["triggered_by"]:
        st.markdown(f"- {t}")

# --- instrumentation problems ----------------------------------------------
dq = data.get("data_quality", [])
if dq:
    st.subheader("Instrumentation")
    st.caption("Each channel is inside its own limits, but the combination is not "
               "physically possible. A drifting or failed transducer is a more likely "
               "explanation than a machine fault — check these before acting.")
    for issue in dq:
        text = f"**{issue.get('channel','?')}** — {issue.get('detail','')}"
        (st.error if issue.get("blocking") else st.warning)(text)

# --- per-mode evidence ------------------------------------------------------
st.subheader("Failure modes")
st.caption("Heat dissipation, power and overstrain are exact specification checks — "
           "the evidence below is the check itself, not an attribution estimate. "
           "Tool wear is a probability over the stated horizon.")

for f in data.get("findings", []):
    mode = f.get("mode", "?")
    label = MODE_LABEL.get(mode, mode)
    used = f.get("envelope_used")
    with st.container(border=True):
        left, right = st.columns([3, 1])
        with left:
            state = "BREACHED" if f.get("detected") else "within specification"
            st.markdown(f"**{label}** — {state}")
            st.caption(f.get("evidence", ""))
            st.caption(f"Source: {f.get('source', '?').replace('_', ' ')}")
        with right:
            if used is not None:
                st.progress(min(float(used), 1.0), text=f"{float(used):.0%} of envelope")

# --- margins ----------------------------------------------------------------
with st.expander("Distance to each specification boundary"):
    st.caption("Positive is headroom; negative means the reading is inside the "
               "failure region. `osf_capacity_used` is a fraction of the limit, "
               "so there 1.0 is the boundary.")
    st.json(data.get("margins", {}))

st.caption(f"Request `{data.get('request_id')}` · policy `{data.get('policy_fingerprint')}` "
           f"· {data.get('latency_ms', 0):.1f} ms")
