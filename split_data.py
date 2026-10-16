"""Create a chronological holdout based on the date of the target Canvass inspection."""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
INPUT_PATH = OUTPUT_DIR / "restaurant_target_prepared.csv"
if not INPUT_PATH.exists(): INPUT_PATH = BASE_DIR / INPUT_PATH.name
TRAIN_PATH = OUTPUT_DIR / "restaurant_train.csv"
TEST_PATH = OUTPUT_DIR / "restaurant_test.csv"
TARGET = "Target_Next_Routine_Failure"
TEST_START = pd.Timestamp("2022-08-04")

if not INPUT_PATH.exists():
    raise FileNotFoundError(f"Missing {INPUT_PATH}. Run target.py first.")

df = pd.read_csv(
    INPUT_PATH,
    parse_dates=["Inspection Date", "Next_Routine_Inspection_Date"],
    low_memory=False,
)
required = [TARGET, "Next_Routine_Inspection_Date", "Restaurant_ID"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

train_df = df[df["Next_Routine_Inspection_Date"] < TEST_START].copy()
test_df = df[df["Next_Routine_Inspection_Date"] >= TEST_START].copy()
if train_df.empty or test_df.empty:
    raise ValueError("Train or test split is empty; inspect target date range and cutoff.")
if train_df[TARGET].isna().any() or test_df[TARGET].isna().any():
    raise ValueError("Target has missing values.")
if set(train_df[TARGET].unique()) - {0, 1} or set(test_df[TARGET].unique()) - {0, 1}:
    raise ValueError("Target must contain only 0 and 1.")

assert train_df["Next_Routine_Inspection_Date"].max() < TEST_START
assert test_df["Next_Routine_Inspection_Date"].min() >= TEST_START

print("Train rows:", f"{len(train_df):,}")
print("Test rows:", f"{len(test_df):,}")
print("Train target dates:", train_df["Next_Routine_Inspection_Date"].min(), "to", train_df["Next_Routine_Inspection_Date"].max())
print("Test target dates:", test_df["Next_Routine_Inspection_Date"].min(), "to", test_df["Next_Routine_Inspection_Date"].max())
print("Train failure rate:", round(train_df[TARGET].mean(), 4))
print("Test failure rate:", round(test_df[TARGET].mean(), 4))
print("Restaurants train/test:", train_df["Restaurant_ID"].nunique(), test_df["Restaurant_ID"].nunique())

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)
print("Saved:", TRAIN_PATH)
print("Saved:", TEST_PATH)
