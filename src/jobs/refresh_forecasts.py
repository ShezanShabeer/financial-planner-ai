import subprocess, sys, os

os.environ.setdefault("FORECAST_DATA_ROOT", "src/stock_predictions_ens")

TICKERS = [
    "AJMANBANK.AE","ALFIRDOUS.AE","EMAAR.AE","ALSALAMSUDAN.AE","DIB.AE",
    "DEWA.AE","AMLAK.AE","EMIRATESNBD.AE","AIRARABIA.AE", "ARAMEX.AE", "CBD.AE", "SUKOON.AE",
    "WATANIA.AE", "ALRAMZ.AE", "DEYAAR.AE", "DFM.AE"
]

CMD = [
  sys.executable, "src/forecast_lstm.py",
  "--period","max",
  "--lookbacks","24,30,36",
  "--seeds","11,22,33,44,55",
  "--hvec","12",
  "--horizons","12,24",
  "--rmax", "0.12",
  "--use_focal", "1",
  "--focal_gamma", "2.0",
  "--exog","^GSPC",
  "--min_fold_windows","10",
  "--val_tail_windows","12",
  "--debug","1"
]

for t in TICKERS:
    print("→", t)
    subprocess.run(CMD + ["--ticker", t], check=False)