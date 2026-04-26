#!/usr/bin/env python3
"""
Real-time 5G latency violation predictor.

Collects live metrics via ping + iperf3, imputes missing features,
and runs the same ensemble inference pipeline as Cell 33 of the notebook.
"""

import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch

from models.definitions import LSTMClassifier, TCNClassifier

# ---------------------------------------------------------------------------
# Constants (must match notebook exactly)
# ---------------------------------------------------------------------------
SEQ_LEN = 20
N_BEST = 18
THRESHOLD = 0.582
SEED = 42
WINDOW_SHORT = 5
WINDOW_LONG = 20
LAG_STEPS = [1, 2, 5]

NODE_LABELS = {
    1: "WiFi_Router",
    2: "Phone_A",
    3: "Phone_B",
    4: "Camera_5G",
    5: "Laptop",
}
NODE_NAME_TO_ID = {v: k for k, v in NODE_LABELS.items()}

# Feature columns in the exact order produced by the notebook (Cell 10-11).
# selector.transform() expects this ordering.
FEATURE_COLS = [
    "hour_of_day",
    "day_of_week",
    "throughput",
    "packet_arrival_rate",
    "snr",
    "signal_strength",
    "network_congestion",
    "bandwidth_usage",
    "pdr",
    "end_to_end_latency",
    "throughput_roll_mean_short",
    "throughput_roll_std_short",
    "throughput_roll_mean_long",
    "network_congestion_roll_mean_short",
    "network_congestion_roll_std_short",
    "network_congestion_roll_mean_long",
    "end_to_end_latency_roll_mean_short",
    "end_to_end_latency_roll_std_short",
    "end_to_end_latency_roll_mean_long",
    "end_to_end_latency_lag1",
    "end_to_end_latency_lag2",
    "end_to_end_latency_lag5",
    "network_congestion_lag1",
    "network_congestion_lag2",
    "network_congestion_lag5",
    "throughput_lag1",
    "throughput_lag2",
    "throughput_lag5",
    "congestion_trend",
    "node_Camera_5G",
    "node_Laptop",
    "node_Phone_A",
    "node_Phone_B",
]

# Indices into FEATURE_COLS for quick lookup
_COL_IDX = {name: i for i, name in enumerate(FEATURE_COLS)}

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent


def load_models():
    """Load all saved models and preprocessing artifacts once at startup."""
    selector   = joblib.load(BASE_DIR / "selector.pkl")
    scaler     = joblib.load(BASE_DIR / "scaler.pkl")
    rf_model   = joblib.load(BASE_DIR / "rf_model.pkl")
    meta_model = joblib.load(BASE_DIR / "meta_model.pkl")

    # Determine input_dim from scaler output shape
    input_dim = scaler.n_features_in_  # equals N_BEST = 18

    lstm_model = LSTMClassifier(input_dim=input_dim)
    lstm_model.load_state_dict(
        torch.load(BASE_DIR / "lstm_model.pt", map_location="cpu")
    )
    lstm_model.eval()

    tcn_model = TCNClassifier(input_dim=input_dim)
    tcn_model.load_state_dict(
        torch.load(BASE_DIR / "tcn_model.pt", map_location="cpu")
    )
    tcn_model.eval()

    return lstm_model, tcn_model, rf_model, meta_model, selector, scaler


# ---------------------------------------------------------------------------
# Live metric collection
# ---------------------------------------------------------------------------

def run_ping(host: str, count: int = 10, interval: float = 0.2) -> dict:
    """Run ping and extract avg RTT (ms) and packet loss (%)."""
    result = {"end_to_end_latency": None, "packet_loss_pct": None}
    try:
        if sys.platform == "win32":
            cmd = ["ping", "-n", str(count), "-w", "1000", host]
        else:
            cmd = ["ping", "-c", str(count), "-i", str(interval), host]

        proc   = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        output = proc.stdout

        # Parse average RTT
        if sys.platform == "win32":
            m = re.search(r"Average\s*=\s*(\d+)\s*ms", output)
        else:
            m = re.search(r"rtt min/avg/max/mdev = [\d.]+/([\d.]+)/", output)
        if m:
            result["end_to_end_latency"] = float(m.group(1))

        # Parse packet loss
        m_loss = re.search(r"(\d+)%\s*(packet )?loss", output)
        if m_loss:
            result["packet_loss_pct"] = float(m_loss.group(1))

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    return result


def run_iperf3(server: str, duration: int = 3) -> dict:
    """Run iperf3 client and extract throughput (Mbps) and jitter (ms)."""
    result = {"throughput_mbps": None, "jitter_ms": None, "packet_loss_pct": None}
    if not server:
        return result
    try:
        import shutil
        iperf3_exe = shutil.which("iperf3") or r"C:\Users\Saravan Kumar\AppData\Local\Microsoft\WinGet\Packages\ar51an.iPerf3_Microsoft.Winget.Source_8wekyb3d8bbwe\iperf3.exe"
        cmd  = [iperf3_exe, "-c", server, "-t", str(duration), "-u", "-b", "200M", "-R", "-J"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=25)
        if proc.returncode != 0:
            return result

        import json as _json
        data = _json.loads(proc.stdout)
        end = data.get("end", {})

        # UDP reverse mode: use sum_received for receiver-side stats
        sum_received = end.get("sum_received", {})
        if sum_received:
            result["throughput_mbps"] = sum_received.get("bits_per_second", 0) / 1e6
            result["jitter_ms"]       = sum_received.get("jitter_ms", 0)
            result["packet_loss_pct"] = sum_received.get("lost_percent", 0)

        # Fallback to sum if sum_received not present (TCP mode)
        if not sum_received:
            sum_sent = end.get("sum_sent", {})
            if sum_sent:
                result["throughput_mbps"] = sum_sent.get("bits_per_second", 0) / 1e6

    except (subprocess.TimeoutExpired, FileNotFoundError, OSError, ValueError, KeyError):
        pass

    return result


def collect_metrics(ping_host: str, iperf_server: str) -> dict:
    """Collect ping and iperf3 metrics in parallel."""
    with ThreadPoolExecutor(max_workers=2) as executor:
        ping_future  = executor.submit(run_ping, ping_host)
        iperf_future = executor.submit(run_iperf3, iperf_server)
        ping_result  = ping_future.result()
        iperf_result = iperf_future.result()

    merged = {**ping_result, **iperf_result}
    # iperf3 packet_loss_pct is more accurate than ping's — keep iperf3 value
    return merged


# ---------------------------------------------------------------------------
# Feature imputation
# ---------------------------------------------------------------------------

class FeatureImputer:
    """Maintains a rolling buffer and imputes missing features for the model."""

    def __init__(self, buffer_size: int = 20):
        self.node_name = "unknown"
        self.buffer    = deque(maxlen=buffer_size)

        # Node dummy columns — all zeroes (no node selected)
        self.node_dummies = {
            "node_Camera_5G": 0.0,
            "node_Laptop":    0.0,
            "node_Phone_A":   0.0,
            "node_Phone_B":   0.0,
        }

    def impute(self, raw_metrics: dict) -> tuple:
        """
        Build a full feature vector from raw_metrics.
        Returns:
            feature_vector : np.ndarray of shape (len(FEATURE_COLS),)
            imputed_list   : list[str] of feature names that were imputed
        """
        vec     = np.zeros(len(FEATURE_COLS), dtype=np.float64)
        imputed = []

        # ── Direct measurements ───────────────────────────────────────────
        latency   = raw_metrics.get("end_to_end_latency")
        throughput = raw_metrics.get("throughput_mbps")
        loss_pct   = raw_metrics.get("packet_loss_pct")

        # Derive network_congestion from packet loss (0-1 scale)
        congestion = (loss_pct / 100.0) if loss_pct is not None else None

        # Latency — impute from buffer if ping failed
        if latency is None:
            latency = self._buffer_mean("end_to_end_latency")
            imputed.append("end_to_end_latency")
        vec[_COL_IDX["end_to_end_latency"]] = latency

        # Throughput — impute from buffer if iperf3 unreachable
        if throughput is None:
            throughput = self._buffer_mean("throughput")
            imputed.append("throughput")
        vec[_COL_IDX["throughput"]] = throughput

        # Congestion — impute from buffer if packet loss unavailable
        if congestion is None:
            congestion = self._buffer_mean("network_congestion")
            imputed.append("network_congestion")
        vec[_COL_IDX["network_congestion"]] = congestion

        # ── Time features ─────────────────────────────────────────────────
        now = datetime.now()
        vec[_COL_IDX["hour_of_day"]] = now.hour
        vec[_COL_IDX["day_of_week"]] = now.weekday()
        imputed.extend(["hour_of_day", "day_of_week"])

        # ── Features not measurable via ping/iperf3: impute from buffer ──
        for feat in ["packet_arrival_rate", "snr", "signal_strength",
                     "bandwidth_usage", "pdr"]:
            val = self._buffer_mean(feat)
            vec[_COL_IDX[feat]] = val
            imputed.append(feat)

        # ── Rolling statistics (computed from buffer) ─────────────────────
        for base_col in ["throughput", "network_congestion", "end_to_end_latency"]:
            rm_short = self._rolling_mean(base_col, WINDOW_SHORT)
            rs_short = self._rolling_std(base_col, WINDOW_SHORT)
            rm_long  = self._rolling_mean(base_col, WINDOW_LONG)

            vec[_COL_IDX[f"{base_col}_roll_mean_short"]] = rm_short
            vec[_COL_IDX[f"{base_col}_roll_std_short"]]  = rs_short
            vec[_COL_IDX[f"{base_col}_roll_mean_long"]]  = rm_long
            imputed.extend([
                f"{base_col}_roll_mean_short",
                f"{base_col}_roll_std_short",
                f"{base_col}_roll_mean_long",
            ])

        # ── Lag features ──────────────────────────────────────────────────
        for base_col in ["end_to_end_latency", "network_congestion", "throughput"]:
            for lag in LAG_STEPS:
                lag_feat = f"{base_col}_lag{lag}"
                vec[_COL_IDX[lag_feat]] = self._lag_value(base_col, lag)
                imputed.append(lag_feat)

        # ── Congestion trend ──────────────────────────────────────────────
        vec[_COL_IDX["congestion_trend"]] = self._congestion_trend()
        imputed.append("congestion_trend")

        # ── Node dummy encoding ───────────────────────────────────────────
        for col, val in self.node_dummies.items():
            vec[_COL_IDX[col]] = val

        # Save snapshot to buffer for future imputation cycles
        snapshot = {
            "end_to_end_latency":  latency,
            "throughput":          throughput,
            "network_congestion":  congestion,
            "packet_arrival_rate": vec[_COL_IDX["packet_arrival_rate"]],
            "snr":                 vec[_COL_IDX["snr"]],
            "signal_strength":     vec[_COL_IDX["signal_strength"]],
            "bandwidth_usage":     vec[_COL_IDX["bandwidth_usage"]],
            "pdr":                 vec[_COL_IDX["pdr"]],
        }
        self.buffer.append(snapshot)

        return vec, imputed

    # ── Buffer helpers ────────────────────────────────────────────────────

    def _buffer_mean(self, key: str) -> float:
        vals = [s[key] for s in self.buffer if key in s and s[key] is not None]
        return float(np.mean(vals)) if vals else 0.0

    def _rolling_mean(self, key: str, window: int) -> float:
        vals = [s.get(key, 0.0) for s in self.buffer]
        if not vals:
            return 0.0
        return float(np.mean(vals[-window:]))

    def _rolling_std(self, key: str, window: int) -> float:
        vals = [s.get(key, 0.0) for s in self.buffer]
        if not vals:
            return 0.0
        window_vals = vals[-window:]
        return float(np.std(window_vals, ddof=0)) if len(window_vals) > 1 else 0.0

    def _lag_value(self, key: str, lag: int) -> float:
        if len(self.buffer) >= lag:
            return self.buffer[-lag].get(key, 0.0)
        elif len(self.buffer) > 0:
            return self.buffer[0].get(key, 0.0)
        return 0.0

    def _congestion_trend(self) -> float:
        if len(self.buffer) >= 10:
            current = self.buffer[-1].get("network_congestion", 0.0)
            past    = self.buffer[-10].get("network_congestion", 0.0)
            return current - past
        return 0.0


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

LOG_COLUMNS = [
    "timestamp", "latency_ms", "throughput_mbps", "congestion",
    "p_lstm", "p_tcn", "p_rf", "p_ensemble", "alert", "imputed_features",
]


def init_csv(path: str):
    """Write header row only if the file does not already exist."""
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(LOG_COLUMNS)


def append_csv(path: str, row: dict):
    """Append one prediction row to the CSV log."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Interactive configuration prompt
# ---------------------------------------------------------------------------

def prompt_config() -> argparse.Namespace:
    """Interactively ask the user for all runtime configuration at startup."""
    print()
    print("=" * 78)
    print("   5G CAMPUS NETWORK — REAL-TIME LATENCY VIOLATION PREDICTOR")
    print("=" * 78)
    print()

    # Ping host
    ping_host = input(
        "  Enter ping target IP address [default: 192.168.10.12]: "
    ).strip()
    if not ping_host:
        ping_host = "192.168.10.12"

    # iperf3 server
    iperf_server = input(
        "  Enter MEC server IP address (iperf3 -s must be running there) "
        "[default: 192.168.10.12, leave blank to skip iperf3]: "
    ).strip()
    if not iperf_server:
        iperf_server = "192.168.10.12"

    # Poll interval
    interval_raw = input(
        "  Prediction frequency in seconds [default: 5]: "
    ).strip()
    try:
        interval = int(interval_raw)
    except ValueError:
        interval = 5

    # Log file
    log = input(
        "  CSV log file path [default: predictions_log.csv]: "
    ).strip()
    if not log:
        log = "predictions_log.csv"

    # Summary
    print()
    print("  ── Configuration Summary " + "─" * 53)
    print(f"  Ping host      : {ping_host}")
    print(f"  iperf3 server  : {iperf_server if iperf_server else '(skipped)'}")
    print(f"  Interval       : {interval}s")
    print(f"  Log file       : {log}")
    print("  " + "─" * 75)
    input("  Press ENTER to start  |  Ctrl+C to cancel: ")
    print()

    return argparse.Namespace(
        ping_host=ping_host,
        iperf_server=iperf_server,
        interval=interval,
        log=log,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    args = prompt_config()

    # Load all model artifacts once at startup
    print("  Loading models...")
    lstm_model, tcn_model, rf_model, meta_model, selector, scaler = load_models()
    print("  Models loaded successfully.\n")

    # Initialise pipeline components
    imputer    = FeatureImputer(buffer_size=WINDOW_LONG)
    seq_buffer = deque(maxlen=SEQ_LEN)
    init_csv(args.log)

    total_predictions = 0
    total_alerts      = 0
    running           = True

    def shutdown(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT,  shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    print("=" * 78)
    print("  LIVE PREDICTION FEED  —  press Ctrl+C to stop")
    print("=" * 78)
    print(
        f"  {'TIMESTAMP':22}  {'LATENCY':>10}  {'THROUGHPUT':>12}  "
        f"{'CONGESTION':>12}  {'P(VIOL)':>9}  STATUS"
    )
    print("  " + "-" * 74)

    while running:
        cycle_start = time.time()
        ts          = datetime.now()
        ts_str      = ts.strftime("%Y-%m-%d %H:%M:%S")

        # ── Step 1: Collect live metrics ─────────────────────────────────
        raw = collect_metrics(args.ping_host, args.iperf_server)

        # ── Step 2: Impute full feature vector ────────────────────────────
        feat_vec, imputed_names = imputer.impute(raw)

        # ── Step 3: Selector → Scaler → Sequence buffer ───────────────────
        feat_raw = feat_vec.reshape(1, -1)
        feat_sel = selector.transform(feat_raw)[0]
        feat_sc  = scaler.transform([feat_sel])[0]
        seq_buffer.append(feat_sc)

        latency_val    = feat_vec[_COL_IDX["end_to_end_latency"]]
        throughput_val = feat_vec[_COL_IDX["throughput"]]
        congestion_val = feat_vec[_COL_IDX["network_congestion"]]

        # Warm-up: wait until sequence buffer has enough history
        if len(seq_buffer) < SEQ_LEN:
            print(
                f"  [{ts_str}]  Warming up buffer... "
                f"({len(seq_buffer)}/{SEQ_LEN} samples collected)"
            )
            end_sleep = time.time() + max(0, args.interval - (time.time() - cycle_start))
            while running and time.time() < end_sleep:
                time.sleep(0.5)
            continue

        # ── Step 4: Inference (identical to Cell 33 pipeline) ─────────────
        seq_array  = np.array(list(seq_buffer))
        seq_tensor = torch.tensor(np.array([seq_array]), dtype=torch.float32)

        with torch.no_grad():
            lstm_p = torch.sigmoid(lstm_model(seq_tensor)).item()
            tcn_p  = torch.sigmoid(tcn_model(seq_tensor)).item()

        rf_p   = rf_model.predict_proba([feat_sel])[0][1]
        p_viol = meta_model.predict_proba([[lstm_p, tcn_p, rf_p]])[0][1]

        alert = p_viol >= THRESHOLD
        total_predictions += 1
        if alert:
            total_alerts += 1

        # ── Step 5: Terminal output ───────────────────────────────────────
        alert_str = "*** ALERT ***" if alert else "OK"
        print(
            f"  [{ts_str}]  "
            f"{latency_val:>8.1f} ms  "
            f"{throughput_val:>10.1f} Mbps  "
            f"{congestion_val:>10.3f}      "
            f"{p_viol:>7.3f}   {alert_str}"
        )

        # ── Step 6: Append to CSV log ─────────────────────────────────────
        append_csv(args.log, {
            "timestamp":        ts_str,
            "latency_ms":       round(latency_val,    2),
            "throughput_mbps":  round(throughput_val, 2),
            "congestion":       round(congestion_val, 4),
            "p_lstm":           round(lstm_p,         4),
            "p_tcn":            round(tcn_p,          4),
            "p_rf":             round(rf_p,           4),
            "p_ensemble":       round(p_viol,         4),
            "alert":            alert,
            "imputed_features": ";".join(imputed_names),
        })

        # ── Sleep for remainder of the polling interval ───────────────────
        elapsed   = time.time() - cycle_start
        end_sleep = time.time() + max(0, args.interval - elapsed)
        while running and time.time() < end_sleep:
            time.sleep(0.5)

    # ── Graceful shutdown summary ─────────────────────────────────────────
    print()
    print("=" * 78)
    print("   SHUTDOWN SUMMARY")
    print("=" * 78)
    print(f"  Total predictions : {total_predictions}")
    print(f"  Total alerts      : {total_alerts}")
    rate = (total_alerts / total_predictions * 100) if total_predictions > 0 else 0.0
    print(f"  Alert rate        : {rate:.1f}%")
    print(f"  Log saved to      : {args.log}")
    print("=" * 78)


if __name__ == "__main__":
    main()