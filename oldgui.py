#!/usr/bin/env python3
"""
Tkinter GUI – always generates a sample that truly belongs to the clicked segment.
Recency / Frequency values are produced using quintile cut-offs learned from
your CSV; therefore the calculated segment never drifts.
"""
import os, datetime as dt, random, tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

DATA_PATH = "csv/file.csv"
FONT_H    = ("Segoe UI", 11, "bold")
PAD       = {"padx": 6, "pady": 3}

# ── 1. Read quintile thresholds ────────────────────────────────────────────────
def quintiles():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found.")
    df = pd.read_csv(DATA_PATH)

    date_col = "Transaction_Date" if "Transaction_Date" in df.columns else "Date"
    df["Transaction_Date"] = pd.to_datetime(df[date_col])

    if "Total_Spend" not in df.columns:
        df["Total_Spend"] = df.get("Online_Spend", 0) + df.get("Offline_Spend", 0)

    snap = df["Transaction_Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        Recency   = ("Transaction_Date", lambda x: (snap - x.max()).days),
        Frequency = ("Transaction_ID",  "nunique"),
        Monetary  = ("Total_Spend",     "sum"),
    )
    return {
        "rec" : rfm["Recency"].quantile([.2,.4,.6,.8]).tolist(),
        "freq": rfm["Frequency"].quantile([.2,.4,.6,.8]).tolist(),
        "mon" : rfm["Monetary"].quantile([.2,.4,.6,.8]).tolist(),
    }
Q = quintiles()

# Helper: pick a random value that yields the desired score
def random_value_for_score(score:int, cuts:list[float],
                           reverse=False, integer=True):
    """
    Generate a random value that falls into the quintile band of the given score.
    reverse=True  →  smaller value = better score (Recency).
    """
    idx = {True:  {5:0, 4:1, 3:2, 2:3, 1:4},
           False: {1:0, 2:1, 3:2, 4:3, 5:4}}[reverse][score]

    low  = -float("inf") if idx == 0 else cuts[idx-1]
    high =  float("inf") if idx == 4 else cuts[idx]

    if low  == -float("inf"): low  = 0
    if high ==  float("inf"): high = low * 1.5 + 10

    if integer:
        low_i  = max(int(low)  + 1, 1)   # ensure ≥1
        high_i = max(int(high), low_i)
        return random.randint(low_i, high_i)
    else:
        return round(random.uniform(low, high), 2)

def score(v, cuts, reverse=False):
    idx = sum(v > c for c in cuts) + 1
    return 6 - idx if reverse else idx

# ── 2. Segment rules ───────────────────────────────────────────────────────────
SEG_MAP = {
    "High-Value Loyalists":  lambda r: r["R"]==5 and r["F"]==5,
    "Loyal":                 lambda r: r["F"]==5 and r["R"]>=4,
    "Potential Loyalists":   lambda r: r["R"]==5 and r["F"]==4,
    "Recent Customers":      lambda r: r["R"]==5 and r["F"]<=3,
    "Promising":             lambda r: r["R"]==4 and r["F"]>=3,
    "Need Attention":        lambda r: r["R"]==3 and r["F"]==3,
    "At Risk":               lambda r: r["R"]<=2 and r["F"]>=3,
    "Hibernating":           lambda r: r["R"]<=2 and r["F"]<=2,
}
def get_segment(rfm_dict):
    for name, rule in SEG_MAP.items():
        if rule(rfm_dict):
            return name
    return "Others"

COLORS = {
    "High-Value Loyalists":"#1b9e77", "Loyal":"#66a61e", "Potential Loyalists":"#b2df8a",
    "Recent Customers":"#7570b3",     "Promising":"#80b1d3", "Need Attention":"#e6ab02",
    "At Risk":"#d95f02",             "Hibernating":"#8dd3c7", "Others":"#999999"
}

# ── 3. Produce a sample that ALWAYS matches the chosen segment ────────────────
def strict_random_example(seg:str):
    """Generate a sample that satisfies the given segment (usually first try)."""
    pool = {
        "High-Value Loyalists":  [(5,5)],
        "Loyal":                 [(5,5),(4,5)],
        "Potential Loyalists":   [(5,4)],
        "Recent Customers":      [(5,i) for i in (1,2,3)],
        "Promising":             [(4,i) for i in (3,4,5)],
        "Need Attention":        [(3,3)],
        "At Risk":               [(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)],
        "Hibernating":           [(1,1),(1,2),(2,1),(2,2)],
        "Others":                [(r,f) for r in range(1,6) for f in range(1,6)
                                  if get_segment({"R":r,"F":f}) == "Others"],
    }[seg]

    for _ in range(50):                 # safety loop
        R, F = random.choice(pool)
        rec  = random_value_for_score(R, Q["rec"],  reverse=True, integer=True)
        freq = random_value_for_score(F, Q["freq"], reverse=False, integer=True)
        mon  = random_value_for_score(random.randint(1,5), Q["mon"],
                                      reverse=False, integer=False)

        calc_seg = get_segment({"R":score(rec, Q["rec"], True),
                                 "F":score(freq, Q["freq"])})
        if calc_seg == seg:
            return rec, freq, mon
    raise RuntimeError(f"Could not find a valid sample for segment '{seg}'.")

# ── 4. Tkinter GUI ─────────────────────────────────────────────────────────────
class RFM_GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RFM Segment Explorer")
        self.geometry("700x450")
        ttk.Style().theme_use("clam")

        # Left-hand segment list
        guide = ttk.Frame(self, padding=10); guide.pack(side="left", fill="y")
        ttk.Label(guide, text="SEGMENT GUIDE", font=FONT_H)\
            .pack(pady=(0,8))
        for seg, color in COLORS.items():
            lbl = tk.Label(guide, text=f"• {seg}", bg=color, fg="white",
                           anchor="w", padx=6, cursor="hand2")
            lbl.segment = seg
            lbl.bind("<Button-1>", self.on_click_segment)
            lbl.pack(fill="x", pady=1)

        # Input form
        form = ttk.Frame(self, padding=18); form.pack(expand=True, fill="both")
        form.columnconfigure(1, weight=1)

        today = dt.date.today()
        self.last_var = tk.StringVar(value=(today - dt.timedelta(days=7)).isoformat())
        self.snap_var = tk.StringVar(value=today.isoformat())
        self.freq_var = tk.IntVar(value=1)
        self.mon_var  = tk.DoubleVar(value=100.0)

        ttk.Label(form, text="Last Purchase Date").grid(row=0, column=0, **PAD)
        ttk.Entry(form, textvariable=self.last_var, width=14)\
            .grid(row=0, column=1, **PAD, sticky="we")

        ttk.Label(form, text="Snapshot Date (Today)").grid(row=1, column=0, **PAD)
        ttk.Entry(form, textvariable=self.snap_var, width=14)\
            .grid(row=1, column=1, **PAD, sticky="we")

        ttk.Label(form, text="Transaction Count").grid(row=2, column=0, **PAD)
        ttk.Spinbox(form, from_=1, to=9999, textvariable=self.freq_var, width=10)\
            .grid(row=2, column=1, **PAD, sticky="w")

        ttk.Label(form, text="Total Spend").grid(row=3, column=0, **PAD)
        ttk.Entry(form, textvariable=self.mon_var, width=12)\
            .grid(row=3, column=1, **PAD, sticky="w")

        ttk.Separator(form).grid(row=4, column=0, columnspan=2, sticky="ew", pady=8)

        ttk.Button(form, text="➜ Calculate Segment",
                   command=self.calc, width=22)\
            .grid(row=5, column=0, columnspan=2, pady=4)

        self.out = ttk.Label(form, text="", font=("Segoe UI", 12, "bold"),
                             foreground="#005b96", justify="center")
        self.out.grid(row=6, column=0, columnspan=2, pady=(14,4))

        ttk.Button(form, text="Next Sample", style="Outline.TButton",
                   command=self.fill_example)\
            .grid(row=7, column=0, columnspan=2, pady=(6,0))

    # Click on a segment label
    def on_click_segment(self, event):
        seg = event.widget.segment
        try:
            rec, freq, mon = strict_random_example(seg)
        except RuntimeError as e:
            messagebox.showerror("Sample Error", str(e))
            return

        d_snap = dt.date.today()
        d_last = d_snap - dt.timedelta(days=rec)
        self.last_var.set(d_last.isoformat())
        self.snap_var.set(d_snap.isoformat())
        self.freq_var.set(freq)
        self.mon_var.set(mon)
        self.calc()

    # Manual calculation
    def calc(self):
        try:
            d_last = dt.date.fromisoformat(self.last_var.get().strip())
            d_snap = dt.date.fromisoformat(self.snap_var.get().strip())
            if d_last > d_snap:
                raise ValueError("Last date cannot be after snapshot date.")
            rec  = (d_snap - d_last).days
            freq = int(self.freq_var.get()); mon = float(self.mon_var.get())
            if freq <= 0 or mon < 0:
                raise ValueError("Frequency must be >0 and spend ≥0.")
        except Exception as e:
            messagebox.showerror("Input Error", str(e)); return

        r = score(rec,  Q["rec"], reverse=True)
        f = score(freq, Q["freq"])
        m = score(mon,  Q["mon"])
        seg = get_segment({"R":r, "F":f})

        self.out.config(
            text=(f"Recency = {rec} days  →  R_Score = {r}\n"
                  f"F_Score = {f},  M_Score = {m}\n\nSEGMENT: {seg}"),
            background=COLORS.get(seg, "#d9d9d9"))

    # “Next Sample” cycles through segments
    _eg_idx = -1
    def fill_example(self):
        segs = list(SEG_MAP) + ["Others"]
        self._eg_idx = (self._eg_idx + 1) % len(segs)
        seg = segs[self._eg_idx]
        # Mock an event to reuse on_click_segment
        self.on_click_segment(type("E", (), {"widget": type("W", (), {"segment": seg})}))

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        RFM_GUI().mainloop()
    except FileNotFoundError as e:
        messagebox.showerror("Missing File", str(e))
