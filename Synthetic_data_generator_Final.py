import numpy as np
import pandas as pd
from datetime import datetime

#===============================#
#Synthetic SEM Dataset Generator#
#===============================#


#-------------- Settings/Config --------------
SEED = 42
np.random.seed(SEED)

START_DATE = "2023-01-01"
END_DATE   = "2024-12-31"

#Dimensions
PROGRAM = ["SEM"]  # focusing only on SEM (allowed by brief)
GEOS    = ["US", "UK", "IN", "DE", "AU"]
DEVICES = ["Desktop", "Mobile", "Tablet"]
SEGMENTS = ["New", "Returning", "HighValue"]
CAMPAIGNS = ["SEM_A", "SEM_B", "SEM_C", "SEM_D", "SEM_E", "SEM_F"]

#Base campaign mix (used to sample rows per day)
CAMPAIGN_WEIGHTS = {
    "SEM_A": 0.22, "SEM_B": 0.18, "SEM_C": 0.15,
    "SEM_D": 0.20, "SEM_E": 0.15, "SEM_F": 0.10
}

#Base CTR/CVR/CPC by device/geo/segment to introduce structure
BASE_CTR_BY_DEVICE = {"Desktop": 0.035, "Mobile": 0.040, "Tablet": 0.030}
BASE_CVR_BY_SEGMENT = {"New": 0.020, "Returning": 0.035, "HighValue": 0.055}
BASE_CPC_BY_GEO = {"US": 0.90, "UK": 0.85, "IN": 0.25, "DE": 0.70, "AU": 0.60}

#AOV by geo & segment (HighValue higher AOV)
BASE_AOV = {
    "US": {"New": 85, "Returning": 110, "HighValue": 180},
    "UK": {"New": 80, "Returning": 105, "HighValue": 170},
    "IN": {"New": 30, "Returning": 45,  "HighValue": 70},
    "DE": {"New": 75, "Returning": 100, "HighValue": 160},
    "AU": {"New": 78, "Returning": 102, "HighValue": 165}
}

#Daily row budget 
#Keep small for speed; increase if needed
ROWS_PER_DAY = 180  

#Campaign pauses (start, end inclusive)
PAUSES = [
    # (campaign_id, start_date, end_date)
    ("SEM_C", "2023-03-10", "2023-03-24"),
    ("SEM_E", "2024-09-05", "2024-09-14")
]

# Performance shifts
SHIFTS = {
    # From 2023-07-01: mobile CTR improves (LP revamp)
    "mobile_ctr_boost_start": "2023-07-01",
    "mobile_ctr_multiplier": 1.12,

    # 2024-03: UK site issues -> lower conversion
    "uk_cvr_dip_start": "2024-03-01",
    "uk_cvr_dip_end": "2024-03-31",
    "uk_cvr_multiplier": 0.78,

    # 2024-Q4: CPC inflation (auction)
    "cpc_inflation_start": "2024-10-01",
    "cpc_inflation_multiplier": 1.15,
}

#--------------Seasonality helpers --------------
def annual_seasonality(day: pd.Timestamp) -> float:
    """Nov-Dec uplift, Jan dip, summer mild bump."""
    m = day.month
    factor = 1.0
    if m in (11, 12):
        factor = 1.25  # peak
    elif m in (4, 5):
        factor = 1.08  # mild bump
    elif m in (1,):
        factor = 0.90  # post-holiday dip
    return factor

def weekly_seasonality(day: pd.Timestamp) -> float:
    """Weekend demand lift, slight Mon dip."""
    dow = day.weekday()  # Mon=0
    if dow in (5, 6):
        return 1.12  # Sat/Sun
    if dow == 0:
        return 0.95   # Mon
    return 1.0

#--------------Pause/ shift helpers--------------
def is_paused(campaign_id: str, day: pd.Timestamp) -> bool:
    for cid, s, e in PAUSES:
        if cid == campaign_id and pd.Timestamp(s) <= day <= pd.Timestamp(e):
            return True
    return False

def apply_shifts(day: pd.Timestamp, geo: str, device: str, ctr: float, cvr: float, cpc: float):
    # Mobile CTR boost from date
    if day >= pd.Timestamp(SHIFTS["mobile_ctr_boost_start"]) and device == "Mobile":
        ctr *= SHIFTS["mobile_ctr_multiplier"]

    # UK CVR dip in March 2024
    if pd.Timestamp(SHIFTS["uk_cvr_dip_start"]) <= day <= pd.Timestamp(SHIFTS["uk_cvr_dip_end"]) and geo == "UK":
        cvr *= SHIFTS["uk_cvr_multiplier"]

    # CPC inflation in 2024-Q4
    if day >= pd.Timestamp(SHIFTS["cpc_inflation_start"]):
        cpc *= SHIFTS["cpc_inflation_multiplier"]

    return ctr, cvr, cpc

#--------------Main Code generation part--------------
def generate_sem_dataset(save_csv=False, out_path= "sem_synthetic.csv"):
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    #Pre-assemble the pool of splits we’ll sample each day
    splits = []
    for camp, geo, dev, seg in [(c,g,d,s) for c in CAMPAIGNS for g in GEOS for d in DEVICES for s in SEGMENTS]:
        splits.append((camp, geo, dev, seg))
    splits = np.array(splits, dtype=object)

    #Build campaign weight vector aligned to 'splits'
    camp_w = np.array([CAMPAIGN_WEIGHTS[s[0]] for s in splits], dtype=float)
    camp_w = camp_w / camp_w.sum()

    rows = []
    for day in dates:
        #Seasonality scaling applied to volume (impressions baseline)
        vol_scale = annual_seasonality(day) * weekly_seasonality(day)

        #Sample which splits to simulate today (without replacement)
        idx = np.random.choice(len(splits), size=min(ROWS_PER_DAY, len(splits)), replace=False, p=camp_w)
        today_splits = splits[idx]

        for (campaign_id, geo, device, segment) in today_splits:
            #If paused, set volume ~0 (but keep row for analysis continuity)
            paused = is_paused(campaign_id, day)

            #---Baselines ---
            #Daily base impressions by geo (larger for US/IN), then scale
            geo_base = {"US": 12000, "UK": 7000, "IN": 15000, "DE": 6000, "AU": 5000}[geo]
            device_adj = {"Desktop": 0.9, "Mobile": 1.1, "Tablet": 0.5}[device]
            seg_adj = {"New": 1.0, "Returning": 0.8, "HighValue": 0.4}[segment]

            base_impr = geo_base * device_adj * seg_adj * vol_scale

            #Add some multiplicative noise on volume
            base_impr *= np.random.lognormal(mean=0, sigma=0.15)

            #If paused: crush impressions/spend
            if paused:
                base_impr *= 0.02

            #--- Rates & costs---
            ctr = BASE_CTR_BY_DEVICE[device] * np.random.normal(1.0, 0.08)
            cvr = BASE_CVR_BY_SEGMENT[segment] * np.random.normal(1.0, 0.10)
            cpc = BASE_CPC_BY_GEO[geo] * np.random.normal(1.0, 0.07)

            #Apply known shifts (device/geo/time based)
            ctr, cvr, cpc = apply_shifts(day, geo, device, ctr, cvr, cpc)

            #Clip to plausible bounds
            ctr = float(np.clip(ctr, 0.005, 0.10))
            cvr = float(np.clip(cvr, 0.005, 0.12))
            cpc = float(np.clip(cpc, 0.10, 4.00))

            #---Funnel calcualtions ---
            impressions = int(max(0, base_impr))
            clicks = int(np.random.binomial(impressions, ctr)) if impressions > 0 else 0

            #Spend from clicks * CPC with some noise
            spend = clicks * cpc * np.random.normal(1.0, 0.05)

            #Shoppers(Shopped or converted by buying) from clicks * cvr (binomial to keep integers)
            shoppers = int(np.random.binomial(clicks, cvr)) if clicks > 0 else 0

            #AOV by geo/segment with noise and mild annual inflation in 2024
            aov = BASE_AOV[geo][segment] * (1.00 if day.year == 2023 else 1.04)
            aov *= np.random.normal(1.0, 0.07)
            aov = float(np.clip(aov, 10, 5000))

            revenue = shoppers * aov

            #Bounce rate inversely related to cvr (but noisy)
            bounce_rate = float(np.clip(0.6 - 2.5 * cvr + np.random.normal(0, 0.05), 0.05, 0.95))

            #Items sold: shoppers times basket size (Poisson with device factor)
            basket_lambda = {"Desktop": 1.9, "Mobile": 1.7, "Tablet": 1.5}[device]
            items_sold = int(np.random.poisson(lam=basket_lambda * max(shoppers, 0)))

            #Derived reporting rates
            ctr_out = (clicks / impressions) if impressions > 0 else 0.0
            cvr_out = (shoppers / clicks) if clicks > 0 else 0.0
            cpc_out = (spend / clicks) if clicks > 0 else 0.0
            aov_out = (revenue / shoppers) if shoppers > 0 else 0.0

            rows.append({
                "date": day.date().isoformat(),
                "program": "SEM",
                "geo": geo,
                "device_type": device,
                "customer_segment": segment,
                "campaign_id": campaign_id,
                "impressions": impressions,
                "clicks": clicks,
                "spend": round(float(spend), 2),
                "ctr": round(ctr_out, 4),
                "cpc": round(cpc_out, 4),
                "conversion_rate": round(cvr_out, 4),
                "shoppers": shoppers,
                "items_sold": items_sold,
                "avg_order_value": round(aov_out, 2),
                "revenue": round(float(revenue), 2),
                "bounce_rate": round(bounce_rate, 3),
                "is_paused": int(paused)
            })

    df = pd.DataFrame(rows)

    #Saving file
    if save_csv:
        df.to_csv('/Users/dattatreyabiswas/Document/Python_work/Projects/Bicycle_ai/data/ecommerce_transactions1.csv', index=False)

    return df





#----------Example usage ----------
df = generate_sem_dataset(save_csv=True)  # writes "sem_synthetic.csv"
df.head()
