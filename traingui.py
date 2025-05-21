import os, datetime as dt, random, tkinter as tk, joblib, pandas as pd
from tkinter import ttk, messagebox

# ───────── Paths & Fonts ──────────────────────────────────────────────────────
DATA_PATH, MODEL_PATH = "csv/file.csv", "model.pkl"
FONT_TITLE = ("Segoe UI", 16, "bold")
FONT_HEAD  = ("Segoe UI", 11, "bold")
PAD = {"padx": 6, "pady": 4}

if not os.path.exists(MODEL_PATH):
    messagebox.showerror("Missing Model", "Run train_rfm_model.py first.")
    raise SystemExit

bundle = joblib.load(MODEL_PATH)
model, le = bundle["pipe"], bundle["label_enc"]
SEGMENTS  = list(le.classes_)

# ───────── Segment colours ────────────────────────────────────────────────────
COLORS = {
    "High-Value Loyalists":"#1b9e77", "Loyal":"#66a61e", "Potential Loyalists":"#b2df8a",
    "Recent Customers":"#7570b3",     "Promising":"#80b1d3", "Need Attention":"#e6ab02",
    "At Risk":"#d95f02",             "Hibernating":"#8dd3c7", "Others":"#999999"
}

# ───────── Optional sample generator (unchanged) ─────────────────────────────
def quintiles():
    df = pd.read_csv(DATA_PATH)
    dc = "Transaction_Date" if "Transaction_Date" in df.columns else "Date"
    df[dc] = pd.to_datetime(df[dc])
    if "Total_Spend" not in df.columns:
        df["Total_Spend"] = df.get("Online_Spend",0)+df.get("Offline_Spend",0)
    snap = df[dc].max()+pd.Timedelta(days=1)
    grp = df.groupby("CustomerID").agg(
        Recency=("Transaction_Date",lambda x:(snap-x.max()).days),
        Frequency=("Transaction_ID","nunique"),
        Monetary=("Total_Spend","sum"))
    return {
        "rec":  grp["Recency"].quantile([.2,.4,.6,.8]).tolist(),
        "freq": grp["Frequency"].quantile([.2,.4,.6,.8]).tolist(),
        "mon":  grp["Monetary"].quantile([.2,.4,.6,.8]).tolist(),
    }
Q = quintiles()

def rnd_val(score,cuts,rev=False,int_=True):
    idx = {True:{5:0,4:1,3:2,2:3,1:4},
           False:{1:0,2:1,3:2,4:3,5:4}}[rev][score]
    lo = 0 if idx==0 else cuts[idx-1]
    hi = (lo*1.5+10) if idx==4 else cuts[idx]
    if int_:
        return random.randint(max(int(lo)+1,1), max(int(hi),int(lo)+1))
    return round(random.uniform(lo,hi),2)

SCORE_POOL = {
 "High-Value Loyalists":[(5,5)], "Loyal":[(5,5),(4,5)], "Potential Loyalists":[(5,4)],
 "Recent Customers":[(5,1),(5,2),(5,3)], "Promising":[(4,3),(4,4),(4,5)],
 "Need Attention":[(3,3)],
 "At Risk":[(1,3),(1,4),(1,5),(2,3),(2,4),(2,5)],
 "Hibernating":[(1,1),(1,2),(2,1),(2,2)]
}
all_pairs = [(r,f) for r in range(1,6) for f in range(1,6)]
SCORE_POOL["Others"] = [p for p in all_pairs if p not in sum(SCORE_POOL.values(),[])]

def random_sample(seg):
    R,F = random.choice(SCORE_POOL[seg])
    rec  = rnd_val(R,Q["rec"],True,True)
    freq = rnd_val(F,Q["freq"],False,True)
    mon  = rnd_val(random.randint(1,5),Q["mon"],False,False)
    return rec,freq,mon
# ───────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RFM Segment Predictor")
        self.geometry("780x500")
        self.minsize(760, 480)
        ttk.Style().theme_use("clam")

        # Global style tweaks
        style = ttk.Style()
        style.configure("TFrame", background="#f6f7f9")
        style.configure("TLabel", background="#f6f7f9")
        style.configure("TButton", font=("Segoe UI",10))
        style.configure("Big.TButton", font=("Segoe UI",11,"bold"))
        style.map("TButton",
                  foreground=[("active","#005b96")],
                  background=[("active","#e1e7f0")])

        # Title bar
        ttk.Label(self, text="Customer Segmentation — RFM Predictor",
                  font=FONT_TITLE, padding=(12,10,0,6)).pack(anchor="w")

        # Main container
        main = ttk.Frame(self, padding=6); main.pack(expand=True, fill="both")

        # ── Left: Segment guide
        guide = ttk.Frame(main, padding=8, style="Card.TFrame")
        guide.pack(side="left", fill="y")
        ttk.Label(guide, text="SEGMENT GUIDE", font=FONT_HEAD
                 ).pack(anchor="w", pady=(0,6))
        for seg in SEGMENTS:
            lab = tk.Label(guide, text=seg, bg=COLORS.get(seg,"#bbb"),
                           fg="white", padx=8, pady=4, cursor="hand2")
            lab.segment = seg
            lab.bind("<Button-1>", self.on_click_segment)
            lab.bind("<Enter>",  lambda e,l=lab: l.config(relief="raised"))
            lab.bind("<Leave>",  lambda e,l=lab: l.config(relief="flat"))
            lab.pack(fill="x", pady=1)

        # ── Right: Input form
        form = ttk.Frame(main, padding=12)
        form.pack(expand=True, fill="both")
        form.columnconfigure(1, weight=1)

        today = dt.date.today()
        self.last_var = tk.StringVar(value=(today-dt.timedelta(days=7)).isoformat())
        self.snap_var = tk.StringVar(value=today.isoformat())
        self.freq_var = tk.IntVar(value=1)
        self.mon_var  = tk.DoubleVar(value=100.0)

        self._entry(form, "Last Purchase Date",   self.last_var, 0)
        self._entry(form, "Snapshot Date (Today)",self.snap_var, 1)
        self._spin (form, "Transaction Count",    self.freq_var, 2)
        self._entry(form, "Total Spend",          self.mon_var,  3)

        ttk.Separator(form).grid(row=4, column=0, columnspan=2, sticky="ew", pady=6)
        ttk.Button(form, text="➜ Predict Segment", width=24,
                   command=self.predict, style="Big.TButton").grid(
                   row=5, column=0, columnspan=2, pady=(0,6))

        self.out = ttk.Label(form, text="", font=("Segoe UI",12,"bold"),
                             foreground="#005b96", anchor="center",
                             padding=6, background="#ffffff", relief="groove")
        self.out.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(4,4))

        ttk.Button(form, text="Random Sample", command=self.next_sample
                  ).grid(row=7,column=0,columnspan=2)

        # hover cursor for entries
        for child in form.winfo_children():
            if isinstance(child, ttk.Entry): child.config(cursor="xterm")

    # quick helpers
    def _entry(self, parent, text, var, row):
        ttk.Label(parent, text=text).grid(row=row,column=0,**PAD,sticky="w")
        ttk.Entry(parent, textvariable=var, width=18
                 ).grid(row=row,column=1,**PAD,sticky="we")
    def _spin(self,  parent, text, var, row):
        ttk.Label(parent, text=text).grid(row=row,column=0,**PAD,sticky="w")
        ttk.Spinbox(parent, from_=1, to=9999, textvariable=var, width=10
                   ).grid(row=row,column=1,**PAD,sticky="w")

    # ── event: click segment label
    def on_click_segment(self, event):
        seg = event.widget.segment
        rec,freq,mon = random_sample(seg)
        d_snap = dt.date.today()
        self.last_var.set((d_snap-dt.timedelta(days=rec)).isoformat())
        self.snap_var.set(d_snap.isoformat())
        self.freq_var.set(freq); self.mon_var.set(mon)
        self.predict()

    # ── parse features
    def _features(self):
        d_last = dt.date.fromisoformat(self.last_var.get().strip())
        d_snap = dt.date.fromisoformat(self.snap_var.get().strip())
        if d_last>d_snap:
            raise ValueError("Last date cannot be after snapshot date.")
        rec  = (d_snap-d_last).days
        freq = int(self.freq_var.get()); mon=float(self.mon_var.get())
        if freq<=0 or mon<0: raise ValueError("Frequency >0 and spend ≥0.")
        return rec,freq,mon

    # ── main prediction
    def predict(self):
        try: rec,freq,mon = self._features()
        except Exception as e:
            messagebox.showerror("Input Error", str(e)); return

        n_feat = model.named_steps["scaler"].mean_.shape[0]
        if n_feat == 6:
            avg   = mon/freq if freq else 0
            tenure= rec
            ratio = 0.0
            X = [[rec,freq,mon,avg,tenure,ratio]]
        else:
            X = [[rec,freq,mon]]

        seg = le.inverse_transform([model.predict(X)[0]])[0]
        self.out.config(text=(f"Recency = {rec} days   |   Freq = {freq}   |   "
                              f"Monetary = {mon:,.0f}\n"
                              f"➜   Predicted Segment:  {seg}"),
                        background=COLORS.get(seg,"#d9d9d9"))

    # cycle samples
    _i = -1
    def next_sample(self):
        self._i = (self._i+1)%len(SEGMENTS)
        self.on_click_segment(type("E",(object,),{"widget":type("W",(object,),{"segment":SEGMENTS[self._i]})})())

# ────────────────
if __name__=="__main__":
    App().mainloop()
