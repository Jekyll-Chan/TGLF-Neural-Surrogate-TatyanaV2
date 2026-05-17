
"""
This is the tool I made to benchmark TatyanaV2 against TGLF prototype
run_tglf_batch.py
Batch TGLF runner + Tatyana V2 inference for a .tsv file
Its structure should be as following:
test_id kymin   trpeps  shat    q0      omt_i   omt_e   omn     gamma_i omega_i gamma_e omega_e source  gamma   omega   mode    is_unstable(1 or 0)
Electrons-first species ordering as: index 1=e, index 2=i
And please change the path on line 23 & 24 before your tryout!!!
(使用前，请记得修改代码中的调用路径)
"""

import os, sys, shutil, subprocess, shlex
import numpy as np
import pandas as pd
from pathlib import Path, PurePosixPath
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ==============================================================
# CONFIG
# Do remember to change the path below accordingly!!
# ==============================================================
TGLF_BIN    = "/home/jekyllchan/gacode/tglf/bin/tglf"   # ← adjust if different!
TATYANA_DIR = "/home/jekyllchan/TatyanaV2"              # ← and here!

INPUT_TSV   = "df_clean_reconstructed.tsv"
OUTPUT_TSV  = "tglf_tatyana_comparison.tsv"
RUNS_DIR    = Path("tglf_batch_runs")

N_WORKERS   = max(1, cpu_count() - 2)
TIMEOUT_S   = 120
KEEP_RUNS   = False
MAX_SAMPLES = None  # Full dataset
SKIP_TGLF   = False  # Set True only if the WSL TGLF binary is unavailable or TGLF runs should be skipped

INPUT_COLS  = ["kymin", "trpeps", "shat", "q0", "omt_i", "omt_e", "omn"]

# some TGLF fixed settings
FIXED = {
    "GEOMETRY_FLAG": 0,
    "SAT_RULE":      0,
    "UNITS":         "'GYRO'",
    "NKY":           1,
    "NS":            2,
    # electrons (species 1)
    "ZS_1": -1,  "MASS_1": 2.7240e-4,  "AS_1": 1.0,
    "VPAR_1": 0.0,  "VPAR_SHEAR_1": 0.0,
    # deuterium ions (species 2)
    "ZS_2":  1,  "MASS_2": 1.0,        "AS_2": 1.0,
    "VPAR_2": 0.0,  "VPAR_SHEAR_2": 0.0,
}


# ---------------------------------------------------------------
# Tatyana V2 inference (the batch runs once before TGLF loop)
# ---------------------------------------------------------------
def run_tatyana_inference(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (gamma_pred, omega_pred) arrays of shape (N,).
    Direct inference with correct ResBlock architecture.
    """
    import torch
    import torch.nn as nn
    import joblib
    
    INPUT_FEATURES = ["kymin", "trpeps", "shat", "q0", "omt_i", "omt_e", "omn"]
    
    # Must match the original ResBlock exactly!!
    class ResBlock(nn.Module):
        def __init__(self, dim, dropout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim), nn.LayerNorm(dim),
            )
            self.act = nn.SiLU()

        def forward(self, x):
            return self.act(x + self.net(x))

    class TatyanaMLP(nn.Module):
        def __init__(self, n_in=7, n_out=2, hidden=256, depth=6, dropout=0.10):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(n_in, hidden), nn.SiLU())
            self.blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(depth)])
            self.head = nn.Linear(hidden, n_out)

        def forward(self, x):
            return self.head(self.blocks(self.embed(x)))

    try:
        print(f"[Tatyana] Loading model from {TATYANA_DIR}...")
        
        # Load scalers
        scaler_file = os.path.join(TATYANA_DIR, "tatyana_v2_scalers.pkl")
        scalers = joblib.load(scaler_file)
        sx, sy = scalers["sx"], scalers["sy"]
        
        # Load model
        device = "cpu"
        model = TatyanaMLP(7, 2, 256, 6, 0.10)
        ckpt_file = os.path.join(TATYANA_DIR, "tatyana_v2.pt")
        model.load_state_dict(torch.load(ckpt_file, map_location=device, weights_only=False))
        model.eval()
        model = model.to(device)
        
        # Prepare data
        X = df[INPUT_FEATURES].values.astype(np.float32)
        
        # Run inference in batches
        batch_size = 256
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                Xs = torch.from_numpy(sx.transform(batch).astype(np.float32)).to(device)
                preds = model(Xs).cpu().numpy()
                all_preds.append(preds)
        
        preds = np.vstack(all_preds) if all_preds else np.zeros((len(X), 2))
        preds_physical = sy.inverse_transform(preds)
        
        print(f"[Tatyana] Inference OK ({len(preds_physical)} samples)")
        return preds_physical[:, 0], preds_physical[:, 1]
        
    except Exception as e:
        print(f"[Tatyana] Inference failed: {e}")
        return np.full(len(df), np.nan), np.full(len(df), np.nan)


# ---------------------------------------------------------------
# TGLF single-run helpers
# ---------------------------------------------------------------
def write_input_tglf(row, run_dir: Path) -> None:
    params = {
        **FIXED,
        "KY":      row["kymin"],
        "RMIN_SA": row["trpeps"],
        "SHAT_SA": row["shat"],   # use SHAT_SA for s-alpha
        "Q_SA":    row["q0"],
        "RLTS_1":  row["omt_e"],  # electrons first here
        "RLNS_1":  row["omn"],
        "RLTS_2":  row["omt_i"],
        "RLNS_2":  row["omn"],
        "TAUS_1":  1.0,
        "TAUS_2":  1.0,
        "AS_1":    1.0,
        "AS_2":    1.0,
    }
    lines = ["# auto-generated by run_tglf_batch.py"]
    for k, v in params.items():
        lines.append(f"{k}={v}")
    (run_dir / "input.tglf").write_text("\n".join(lines) + "\n")


def parse_eigenvalue(run_dir: Path):
    """
    Parse out.tglf.eigenvalue_spectrum.
    Returns the first numeric row as (gamma, omega) or (nan, nan).
    """
    ef = run_dir / "out.tglf.eigenvalue_spectrum"
    if not ef.exists():
        return np.nan, np.nan
    try:
        for line in ef.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    gamma = float(parts[0])
                    omega = float(parts[1])
                    return gamma, omega
                except ValueError:
                    continue
    except Exception:
        pass
    return np.nan, np.nan


def resolve_run_dir_wsl(run_dir: Path) -> str:
    run_dir = run_dir.resolve()
    run_dir_wsl = str(run_dir).replace('\\', '/')
    if 'wsl.localhost/Ubuntu' in run_dir_wsl:
        run_dir_wsl = run_dir_wsl.split('wsl.localhost/Ubuntu', 1)[1]
        if not run_dir_wsl.startswith('/'):
            run_dir_wsl = '/' + run_dir_wsl
    return run_dir_wsl


def run_one(args):
    seq_idx, row, g_tatyana, w_tatyana = args
    run_dir = RUNS_DIR / f"run_{seq_idx:07d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    result = dict(
        test_id       = row["test_id"],
        kymin         = row["kymin"],
        trpeps        = row["trpeps"],
        shat          = row["shat"],
        q0            = row["q0"],
        omt_i         = row["omt_i"],
        omt_e         = row["omt_e"],
        omn           = row["omn"],
        mode_ref      = row["mode"],
        gamma_ref     = row["gamma"],
        omega_ref     = row["omega"],
        gamma_tglf    = np.nan,
        omega_tglf    = np.nan,
        gamma_tatyana = g_tatyana,
        omega_tatyana = w_tatyana,
        tglf_status   = "ok",
    )

    try:
        write_input_tglf(row, run_dir)
        
        run_dir_wsl = resolve_run_dir_wsl(run_dir)
        gacode_root = PurePosixPath(TGLF_BIN).parents[2]
        shared_bin = gacode_root / "shared" / "bin"
        tglf_bin_dir = PurePosixPath(TGLF_BIN).parent

        if os.name == 'nt':
            shell_cmd = (
                f"export GACODE_ROOT={shlex.quote(str(gacode_root))} && "
                f"export PATH=\"$PATH:{shlex.quote(str(tglf_bin_dir))}:{shlex.quote(str(shared_bin))}\" && "
                f"cd {shlex.quote(run_dir_wsl)} && {shlex.quote(TGLF_BIN)} -e ."
            )
            proc = subprocess.run(
                ["wsl.exe", "bash", "-lc", shell_cmd],
                capture_output=True,
                text=False,
                timeout=TIMEOUT_S,
            )
        else:
            env = os.environ.copy()
            env["GACODE_ROOT"] = str(gacode_root)
            env["PATH"] = ":".join(filter(None, [env.get("PATH", ""), str(tglf_bin_dir), str(shared_bin)]))
            proc = subprocess.run(
                [TGLF_BIN, "-e", "."],
                cwd=run_dir,
                env=env,
                capture_output=True,
                text=False,
                timeout=TIMEOUT_S,
            )
        if proc.returncode != 0:
            result["tglf_status"] = f"tglf_err:{proc.returncode}"
        else:
            gamma, omega = parse_eigenvalue(run_dir)
            result["gamma_tglf"] = gamma
            result["omega_tglf"] = omega
            if np.isnan(gamma):
                result["tglf_status"] = "parse_fail"
    except subprocess.TimeoutExpired:
        result["tglf_status"] = "timeout"
    except FileNotFoundError:
        result["tglf_status"] = "tglf_not_found"
    except Exception as e:
        result["tglf_status"] = f"exception:{str(e)[:50]}"
    finally:
        if not KEEP_RUNS:
            shutil.rmtree(run_dir, ignore_errors=True)

    return result


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    df = pd.read_csv(INPUT_TSV, sep=r"\s+", engine="python")
    
    # Limit dataset if MAX_SAMPLES is set  
    if MAX_SAMPLES is not None and len(df) > MAX_SAMPLES:
        df = df.head(MAX_SAMPLES)
    
    print(f"Loaded {len(df)} rows from {INPUT_TSV}")
    print(f"Workers: {N_WORKERS}  |  Timeout: {TIMEOUT_S}s/run")
    if SKIP_TGLF:
        print(f"[INFO] TGLF binary not found at {TGLF_BIN} — skipping TGLF runs")
    
    # Tatyana V2 — fast batch inference up front
    print("\n[Tatyana] Running batch inference...")
    g_tat, w_tat = run_tatyana_inference(df)

    # TGLF — parallel per-row runs (if available) 好像不能平行跑
    if not SKIP_TGLF:
        RUNS_DIR.mkdir(exist_ok=True)
        rows  = df.to_dict(orient="records")
        args  = [(i, row, float(g_tat[i]), float(w_tat[i])) for i, row in enumerate(rows)]

        print(f"\n[TGLF] Launching {len(args)} runs with {N_WORKERS} workers...")
        with Pool(N_WORKERS) as pool:
            results = list(tqdm(
                pool.imap_unordered(run_one, args, chunksize=8),
                total=len(args),
                desc="TGLF",
            ))
    else:
        # Generate results if with no TGLF
        # 嗯，这是一部分TGLF跑失败的备份出口。。。
        results = []
        for i, (idx, row) in enumerate(df.iterrows()):
            result = dict(
                test_id       = row["test_id"],
                kymin         = row["kymin"],
                trpeps        = row["trpeps"],
                shat          = row["shat"],
                q0            = row["q0"],
                omt_i         = row["omt_i"],
                omt_e         = row["omt_e"],
                omn           = row["omn"],
                mode_ref      = row["mode"],
                gamma_ref     = row["gamma"],
                omega_ref     = row["omega"],
                gamma_tglf    = np.nan,
                omega_tglf    = np.nan,
                gamma_tatyana = g_tat[i],
                omega_tatyana = w_tat[i],
                tglf_status   = "skipped",
            )
            results.append(result)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_TSV, sep="\t", index=False)

    # 检查一下TGLF原型有没有被跳过
    if not SKIP_TGLF:
        ok   = out["tglf_status"] == "ok"
        n_ok = ok.sum()
        print(f"\n{'='*60}")
        print(f"TGLF: {n_ok}/{len(out)} OK  |  {len(out)-n_ok} failed")
        if (len(out) - n_ok) > 0:
            print(out.loc[~ok, "tglf_status"].value_counts().to_string())
    else:
        print(f"\n{'='*60}")
        print(f"[INFO] TGLF runs skipped — Tatyana-only results")

    clip = 1e-8
    if len(out) > 0:
        sub    = out.copy()
        # Convert to numeric types to handle string columns
        sub["gamma_ref"] = pd.to_numeric(sub["gamma_ref"], errors='coerce')
        sub["omega_ref"] = pd.to_numeric(sub["omega_ref"], errors='coerce')
        sub["gamma_tatyana"] = pd.to_numeric(sub["gamma_tatyana"], errors='coerce')
        sub["omega_tatyana"] = pd.to_numeric(sub["omega_tatyana"], errors='coerce')
        # Filter out NaN rows 空白的都叉出去
        sub = sub.dropna(subset=['gamma_ref', 'omega_ref', 'gamma_tatyana', 'omega_tatyana'])
        if len(sub) > 0:
            ref_g  = sub["gamma_ref"].abs().clip(lower=clip)
            ref_w  = sub["omega_ref"].abs().clip(lower=clip)

            if not SKIP_TGLF:
                sub["gamma_tglf"] = pd.to_numeric(sub["gamma_tglf"], errors='coerce')
                sub["omega_tglf"] = pd.to_numeric(sub["omega_tglf"], errors='coerce')
                tglf_g = (sub["gamma_tglf"]    - sub["gamma_ref"]).abs() / ref_g
                tglf_w = (sub["omega_tglf"]    - sub["omega_ref"]).abs() / ref_w
            
            tat_g  = (sub["gamma_tatyana"] - sub["gamma_ref"]).abs() / ref_g
            tat_w  = (sub["omega_tatyana"] - sub["omega_ref"]).abs() / ref_w

            fmt = "{:<38} {:>10} {:>12}"
            if not SKIP_TGLF:
                print(f"\n{fmt.format('Metric', 'TGLF local', 'Tatyana V2')}")
                print("-" * 62)
                print(fmt.format("γ  median rel err vs NSCC ref",
                                 f"{tglf_g.median()*100:.3f}%", f"{tat_g.median()*100:.3f}%"))
                print(fmt.format("ω  median rel err vs NSCC ref",
                                 f"{tglf_w.median()*100:.3f}%", f"{tat_w.median()*100:.3f}%"))
                print(fmt.format("γ  mean   rel err vs NSCC ref",
                                 f"{tglf_g.mean()*100:.3f}%",   f"{tat_g.mean()*100:.3f}%"))
                print(fmt.format("ω  mean   rel err vs NSCC ref",
                                 f"{tglf_w.mean()*100:.3f}%",   f"{tat_w.mean()*100:.3f}%"))
                print(f"\n(TGLF local ~0% vs NSCC = setup matches; Tatyana ~0.5% expected)")
            else:
                print(f"\n{'Metric':<38} {'Tatyana V2':>12}")
                print("-" * 53)
                print(f"{'γ  median rel err vs NSCC ref':<38} {tat_g.median()*100:>11.3f}%")
                print(f"{'ω  median rel err vs NSCC ref':<38} {tat_w.median()*100:>11.3f}%")
                print(f"{'γ  mean   rel err vs NSCC ref':<38} {tat_g.mean()*100:>11.3f}%")
                print(f"{'ω  mean   rel err vs NSCC ref':<38} {tat_w.mean()*100:>11.3f}%")

    print(f"\nResults -> {OUTPUT_TSV}")

    # Comparison plots 结算绘图部分 🫡
    make_comparison_plots(out)


# ---------------------------------------------------------------
# Comparison plots: 🍔 TGLF VS TatyanaV2 🍲
# ---------------------------------------------------------------
def make_comparison_plots(out: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOT_DIR = Path(TATYANA_DIR)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for col in ["gamma_ref", "omega_ref", "gamma_tglf", "omega_tglf",
                "gamma_tatyana", "omega_tatyana"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    clip = 1e-8
    has_tglf = (not SKIP_TGLF) and out["gamma_tglf"].notna().any()

    # helper
    def parity_ax(ax, ref, pred, label, color):
        mask = ref.notna() & pred.notna()
        x, y = ref[mask].values, pred[mask].values
        ax.scatter(x, y, s=6, alpha=0.4, color=color, rasterized=True)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", lw=0.8, label="y=x")
        ax.set_xlabel("NSCC reference")
        ax.set_ylabel(label)
        rel_err = np.abs(y - x) / np.abs(x).clip(clip)
        ax.set_title(f"{label}  median Δ={np.median(rel_err)*100:.2f}%")
        ax.legend(fontsize=7)

    def hist_ax(ax, ref, pred, label, color, bins=60):
        mask = ref.notna() & pred.notna()
        x, y = ref[mask].values, pred[mask].values
        rel_err = np.abs(y - x) / np.abs(x).clip(clip) * 100
        ax.hist(rel_err, bins=bins, color=color, alpha=0.7, edgecolor="none")
        ax.axvline(np.median(rel_err), color="k", lw=1, linestyle="--",
                   label=f"median {np.median(rel_err):.2f}%")
        ax.set_xlabel("Relative error (%)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}  rel-err distribution")
        ax.legend(fontsize=7)

    # Figure 1: Parity plots 
    n_cols = 2 if has_tglf else 1
    fig, axes = plt.subplots(2, n_cols * 2, figsize=(6 * n_cols * 2, 10))
    fig.suptitle("TGLF vs Tatyana V2 — Parity & Error Distributions", fontsize=13)

    col = 0
    if has_tglf:
        parity_ax(axes[0, col],   out["gamma_ref"], out["gamma_tglf"],    "TGLF γ",    "#2166ac")
        parity_ax(axes[1, col],   out["omega_ref"], out["omega_tglf"],    "TGLF ω",    "#2166ac")
        hist_ax  (axes[0, col+1], out["gamma_ref"], out["gamma_tglf"],    "TGLF γ",    "#2166ac")
        hist_ax  (axes[1, col+1], out["omega_ref"], out["omega_tglf"],    "TGLF ω",    "#2166ac")
        col += 2

    parity_ax(axes[0, col],   out["gamma_ref"], out["gamma_tatyana"], "Tatyana γ", "#d6604d")
    parity_ax(axes[1, col],   out["omega_ref"], out["omega_tatyana"], "Tatyana ω", "#d6604d")
    hist_ax  (axes[0, col+1], out["gamma_ref"], out["gamma_tatyana"], "Tatyana γ", "#d6604d")
    hist_ax  (axes[1, col+1], out["omega_ref"], out["omega_tatyana"], "Tatyana ω", "#d6604d")

    fig.tight_layout()
    p1 = PLOT_DIR / "comparison_parity.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved {p1}")

    # Figure 2: Scatter TGLF vs Tatyana 只有两个都跑出来才可以画
    if has_tglf:
        mask = out["gamma_tglf"].notna() & out["gamma_tatyana"].notna()
        sub  = out[mask]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("TGLF local  vs  Tatyana V2", fontsize=13)
        for ax, gcol, wcol, lbl in [
            (axes[0], "gamma_tglf", "gamma_tatyana", "γ"),
            (axes[1], "omega_tglf", "omega_tatyana", "ω"),
        ]:
            ax.scatter(sub[gcol], sub[wcol], s=6, alpha=0.4, rasterized=True)
            lims = [min(sub[gcol].min(), sub[wcol].min()),
                    max(sub[gcol].max(), sub[wcol].max())]
            ax.plot(lims, lims, "k--", lw=0.8)
            ax.set_xlabel(f"TGLF {lbl}")
            ax.set_ylabel(f"Tatyana {lbl}")
            ax.set_title(f"TGLF vs Tatyana — {lbl}")
        fig.tight_layout()
        p2 = PLOT_DIR / "comparison_tglf_vs_tatyana.png"
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        print(f"[Plot] Saved {p2}")

    # Figure 3: rel-error vs each input feature 
    features = ["kymin", "trpeps", "shat", "q0", "omt_i", "omt_e", "omn"]
    targets  = [("gamma", "#d6604d"), ("omega", "#4dac26")]
    sources  = []
    if has_tglf:
        sources.append(("tglf",    "#2166ac", "TGLF"))
    sources.append(("tatyana", "#d6604d", "Tatyana V2"))

    for src_key, src_color, src_label in sources:
        fig, axes = plt.subplots(2, len(features), figsize=(4 * len(features), 8))
        fig.suptitle(f"{src_label} — relative error vs input features", fontsize=13)
        for row_i, (qty, color) in enumerate(targets):
            ref_col  = f"{qty}_ref"
            pred_col = f"{qty}_{src_key}"
            mask = out[ref_col].notna() & out[pred_col].notna()
            sub  = out[mask]
            rel_err = (sub[pred_col] - sub[ref_col]).abs() / sub[ref_col].abs().clip(clip) * 100
            for col_j, feat in enumerate(features):
                ax = axes[row_i, col_j]
                ax.scatter(sub[feat], rel_err, s=4, alpha=0.3, color=color, rasterized=True)
                ax.set_xlabel(feat, fontsize=8)
                ax.set_ylabel("rel err (%)" if col_j == 0 else "", fontsize=8)
                ax.set_title(f"{qty} vs {feat}", fontsize=8)
                ax.tick_params(labelsize=7)
        fig.tight_layout()
        p3 = PLOT_DIR / f"comparison_relerr_vs_features_{src_key}.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        print(f"[Plot] Saved {p3}")

    print(f"[Plot] All plots written to {PLOT_DIR}/")


if __name__ == "__main__":
    main()

# 其实让TGLF跑起来才是难点吧😅😅😅