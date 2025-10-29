# ================================================================
# 2 Choco Crunch Analytics — End-to-End Pipeline
# Fetch → Clean → Feature Engineer → EDA → 27 Analytics CSVs
# ================================================================

import os, time, json, random, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- setup directories ---
os.makedirs("data", exist_ok=True)
os.makedirs("data/clean", exist_ok=True)
os.makedirs("out", exist_ok=True)
os.makedirs("eda", exist_ok=True)

RAW_JSONL = "data/raw_chocolate.jsonl"
PRODUCT_CSV = "data/product_info.csv"
NUTRIENT_CSV = "data/nutrient_info.csv"
DERIVED_CSV  = "data/derived_metrics.csv"
ENGINEERED_PARQUET = "data/clean/choco_engineered.parquet"
ENGINEERED_SNAPSHOT = "out/full_engineered_snapshot.csv"

MAX_RECORDS = 12000
PAGE_SIZE = 100
TIMEOUT = 50

# ------------------------------------------------
# Helpers
# ------------------------------------------------
def to_float(x):
    try: return float(str(x).split()[0])
    except: return np.nan

def safe_num(s): return pd.to_numeric(s, errors="coerce")

def normalize_brand(b):
    if not b: return "Unknown"
    first = str(b).split(",")[0].strip()
    return first.title() if first else "Unknown"

# ------------------------------------------------
# Step 1 — Fetch ≈ 12 000 records (v2 API + fallback)
# ------------------------------------------------
def _get_products(params, page):
    for attempt in range(4):
        try:
            r = requests.get("https://world.openfoodfacts.org/api/v2/search",
                             params={**params, "page": page, "page_size": PAGE_SIZE},
                             timeout=TIMEOUT)
            r.raise_for_status()
            return r.json().get("products", []) or []
        except Exception as e:
            print(f"⚠️ page {page} attempt {attempt+1}: {e}")
            time.sleep(1 + attempt)
    return []

def step_fetch_combined(max_records=MAX_RECORDS):
    print("🔹 Fetching chocolate data (try categories then search_terms)...")
    all_prods, total = [], 0
    sources = [
        {"categories":"chocolates"},
        {"search_terms":"chocolate"},
    ]
    for source in sources:
        page = 1
        while total < max_records:
            prods = _get_products(source, page)
            if not prods: break
            all_prods.extend(prods)
            total = len(all_prods)
            print(f"   → {source} page {page}: {len(prods)} rows (total {total})")
            page += 1
            if total >= max_records: break
    all_prods = all_prods[:max_records]
    with open(RAW_JSONL, "w", encoding="utf-8") as f:
        for p in all_prods:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"✅ Fetched {len(all_prods)} products → {RAW_JSONL}")
    return len(all_prods)

# ------------------------------------------------
# Step 2 — Cleaning and Transformation
# ------------------------------------------------
def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s=line.strip()
            if s: 
                try: yield json.loads(s)
                except: continue

def step_clean_transform():
    if not os.path.exists(RAW_JSONL):
        raise FileNotFoundError("Missing raw JSONL.")
    prows, nrows, drows = [], [], []
    for p in read_jsonl(RAW_JSONL):
        code=p.get("code"); name=p.get("product_name") or "Unknown"
        brand=normalize_brand(p.get("brands"))
        nutr=p.get("nutriments") or {}
        prows.append({"product_code":code,"product_name":name,"brand":brand})
        nrows.append({
            "product_code":code,
            "energy-kcal_value":to_float(nutr.get("energy-kcal_100g")),
            "energy-kj_value":to_float(nutr.get("energy-kj_100g")),
            "carbohydrates_value":to_float(nutr.get("carbohydrates_100g")),
            "sugars_value":to_float(nutr.get("sugars_100g")),
            "fat_value":to_float(nutr.get("fat_100g")),
            "saturated-fat_value":to_float(nutr.get("saturated-fat_100g")),
            "proteins_value":to_float(nutr.get("proteins_100g")),
            "fiber_value":to_float(nutr.get("fiber_100g")),
            "salt_value":to_float(nutr.get("salt_100g")),
            "sodium_value":to_float(nutr.get("sodium_100g")),
            "fruits-vegetables-nuts-estimate-from-ingredients_100g":
                to_float(nutr.get("fruits-vegetables-nuts-estimate-from-ingredients_100g")),
            "nutrition-score-fr":to_float(nutr.get("nutrition-score-fr_100g") or nutr.get("nutrition-score-fr")),
            "nova-group":to_float(nutr.get("nova-group_100g") or nutr.get("nova-group")),
        })
        drows.append({"product_code":code})
    dfp=pd.DataFrame(prows).drop_duplicates("product_code")
    dfn=pd.DataFrame(nrows).drop_duplicates("product_code")
    dfd=pd.DataFrame(drows).drop_duplicates("product_code")
    dfp.to_csv(PRODUCT_CSV,index=False)
    dfn.to_csv(NUTRIENT_CSV,index=False)
    dfd.to_csv(DERIVED_CSV,index=False)
    print("✅ Cleaned data saved.")

# ------------------------------------------------
# Step 3 — Feature Engineering and Derived Metrics
# ------------------------------------------------
def step_feature_engineer():
    dfp=pd.read_csv(PRODUCT_CSV)
    dfn=pd.read_csv(NUTRIENT_CSV)
    df=dfp.merge(dfn,on="product_code",how="left")

    df["sugars_value"]=safe_num(df["sugars_value"])
    df["carbohydrates_value"]=safe_num(df["carbohydrates_value"])
    df["energy-kcal_value"]=safe_num(df["energy-kcal_value"])
    df["nova-group"]=safe_num(df["nova-group"])

    df["sugar_to_carb_ratio"]=np.where(
        (df["carbohydrates_value"]>0)&df["sugars_value"].notna(),
        np.clip(df["sugars_value"]/df["carbohydrates_value"],0,1),np.nan)

    def cat_cal(k):
        if pd.isna(k):return"Unknown"
        if k<=150:return"Low"
        if k<=300:return"Moderate"
        return"High"
    df["calorie_category"]=df["energy-kcal_value"].apply(cat_cal)

    def cat_sug(s):
        if pd.isna(s):return"Unknown"
        if s<5:return"Low"
        if s<=15:return"Moderate"
        return"High"
    df["sugar_category"]=df["sugars_value"].apply(cat_sug)

    df["is_ultra_processed"]=df["nova-group"].apply(lambda n:"Yes" if n==4 else("No" if pd.notna(n) else"Unknown"))
    df.to_parquet(ENGINEERED_PARQUET,index=False)
    df.to_csv(ENGINEERED_SNAPSHOT,index=False)
    print("✅ Feature-engineered snapshot saved.")
    return df

# ------------------------------------------------
# Step 4 — Exploratory Data Analysis (EDA)
# ------------------------------------------------
def step_eda(df):
    print("📊 Running EDA plots...")
    sns.histplot(df["energy-kcal_value"].dropna(),bins=30)
    plt.title("Energy (kcal per 100g)"); plt.savefig("eda/energy_hist.png"); plt.clf()
    sns.histplot(df["sugars_value"].dropna(),bins=30)
    plt.title("Sugar (g per 100g)"); plt.savefig("eda/sugar_hist.png"); plt.clf()
    sns.boxplot(x="calorie_category",y="sugars_value",data=df)
    plt.title("Sugar by Calorie Category"); plt.savefig("eda/box_sugar_calorie.png"); plt.clf()
    sns.scatterplot(x="energy-kcal_value",y="sugars_value",data=df)
    plt.title("Energy vs Sugar"); plt.savefig("eda/scatter_energy_sugar.png"); plt.clf()
    sns.heatmap(df[["energy-kcal_value","sugars_value","carbohydrates_value","fat_value","proteins_value"]].corr(),
                annot=True,cmap="coolwarm")
    plt.title("Correlation Heatmap"); plt.savefig("eda/heatmap_corr.png"); plt.clf()
    print("✅ EDA plots saved to 'eda/'.")

# ------------------------------------------------
# Step 5 — 27 Analytics CSV Queries
# ------------------------------------------------
def step_analytics(df):
    df["brand"]=df["brand"].fillna("Unknown")
    df["sugartocarbratio"]=df["sugar_to_carb_ratio"]
    def w(name,frame): frame.to_csv(f"out/{name}.csv",index=False)

    w("q01_products_per_brand",df.groupby("brand").size().reset_index(name="total"))
    w("q02_unique_products_per_brand",df.groupby("brand")["product_code"].nunique().reset_index(name="unique_products"))
    w("q03_top5_brands_by_count",df["brand"].value_counts().head(5).reset_index())
    w("q04_missing_product_name",df[df["product_name"].isna()])
    w("q05_unique_brands_count",pd.DataFrame([{"unique_brands":df['brand'].nunique()}]))
    w("q06_codes_starting_3",df[df["product_code"].astype(str).str.startswith("3")])
    w("q07_top10_energy_kcal",df.nlargest(10,"energy-kcal_value")[["product_code","brand","energy-kcal_value"]])
    w("q08_avg_sugar_by_nova",df.groupby("nova-group")["sugars_value"].mean().reset_index(name="avg_sugar"))
    w("q09_fat_gt20",df[df["fat_value"]>20])
    w("q10_avg_carbs_per_product",df.groupby("product_code")["carbohydrates_value"].mean().reset_index())
    w("q11_sodium_gt1g",df[df["sodium_value"]>1])
    w("q12_fvn_nonzero",df[df["fruits-vegetables-nuts-estimate-from-ingredients_100g"].fillna(0)>0])
    w("q13_energy_gt500",df[df["energy-kcal_value"]>500])
    w("q14_count_per_calorie_category",df["calorie_category"].value_counts().reset_index())
    w("q15_high_sugar",df[df["sugar_category"]=="High"])
    w("q16_avg_ratio_high_calorie",pd.DataFrame([{"avg_ratio":df[df["calorie_category"]=="High"]["sugar_to_carb_ratio"].mean()}]))
    w("q17_high_cal_high_sugar",df[(df["calorie_category"]=="High")&(df["sugar_category"]=="High")])
    w("q18_ultra_processed",df[df["is_ultra_processed"]=="Yes"])
    w("q19_ratio_gt0_7",df[df["sugar_to_carb_ratio"]>0.7])
    w("q20_avg_ratio_by_calorie_category",df.groupby("calorie_category")["sugar_to_carb_ratio"].mean().reset_index())
    w("q21_top5_brands_high_cal",df[df["calorie_category"]=="High"]["brand"].value_counts().head(5).reset_index())
    w("q22_avg_kcal_by_calorie_category",df.groupby("calorie_category")["energy-kcal_value"].mean().reset_index())
    w("q23_ultra_count_per_brand",df[df["is_ultra_processed"]=="Yes"]["brand"].value_counts().reset_index())
    w("q24_high_sugar_high_cal_brand",df[(df["sugar_category"]=="High")&(df["calorie_category"]=="High")][["brand","sugars_value","energy-kcal_value"]])
    w("q25_avg_sugar_ultra_by_brand",df[df["is_ultra_processed"]=="Yes"].groupby("brand")["sugars_value"].mean().reset_index())
    w("q26_avg_fvn_by_calorie_category",df.groupby("calorie_category")["fruits-vegetables-nuts-estimate-from-ingredients_100g"].mean().reset_index())
    w("q27_top5_by_ratio",df.nlargest(5,"sugar_to_carb_ratio")[["product_code","brand","product_name","sugar_to_carb_ratio","calorie_category","sugar_category"]])
    print("✅ 27 analytics CSVs saved to 'out/'.")

# ------------------------------------------------
# Orchestration
# ------------------------------------------------
if __name__=="__main__":
    print("🚀 Starting Choco Crunch Analytics Pipeline...\n")
    fetched=step_fetch_combined(MAX_RECORDS)
    step_clean_transform()
    df=step_feature_engineer()
    step_eda(df)
    step_analytics(df)
    print(f"\n🎉 Completed ({fetched} records) → check 'data/', 'out/', and 'eda/' folders.")
