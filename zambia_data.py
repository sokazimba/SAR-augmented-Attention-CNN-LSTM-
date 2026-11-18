"""
zambia_data.py
Comprehensive database and geographic registry of Zambia's 116 districts across 10 provinces.
Includes:
- District centroids (Latitude, Longitude)
- Agro-Ecological Zones (AEZ I, IIa, IIb, III)
- Historical baseline maize yields (t/ha) based on Crop Forecast Survey (CFS) data
- Harvested area weights for accurate national yield aggregation: Y_national = sum(A_i * Y_i) / sum(A_i)
"""

import pandas as pd
import numpy as np

# Complete catalog of Zambian districts grouped by province
DISTRICTS_DATA = [
    # CENTRAL PROVINCE (11 districts) - Prime commercial & smallholder maize belt
    {"district": "Chibombo", "province": "Central", "lat": -14.65, "lon": 28.08, "aez": "AEZ IIa", "area_ha": 65000, "base_yield": 2.25},
    {"district": "Chisamba", "province": "Central", "lat": -14.98, "lon": 28.25, "aez": "AEZ IIa", "area_ha": 58000, "base_yield": 2.45},
    {"district": "Chitambo", "province": "Central", "lat": -12.92, "lon": 30.65, "aez": "AEZ III", "area_ha": 22000, "base_yield": 1.75},
    {"district": "Itezhi-Tezhi", "province": "Central", "lat": -15.75, "lon": 26.02, "aez": "AEZ IIa", "area_ha": 34000, "base_yield": 1.85},
    {"district": "Kabwe", "province": "Central", "lat": -14.44, "lon": 28.45, "aez": "AEZ IIa", "area_ha": 18000, "base_yield": 2.10},
    {"district": "Kapiri Mposhi", "province": "Central", "lat": -13.97, "lon": 28.67, "aez": "AEZ IIa", "area_ha": 72000, "base_yield": 2.30},
    {"district": "Luano", "province": "Central", "lat": -14.65, "lon": 29.80, "aez": "AEZ I", "area_ha": 25000, "base_yield": 1.45},
    {"district": "Mkushi", "province": "Central", "lat": -13.62, "lon": 29.43, "aez": "AEZ IIa", "area_ha": 85000, "base_yield": 2.65},
    {"district": "Mumbwa", "province": "Central", "lat": -14.98, "lon": 27.06, "aez": "AEZ IIa", "area_ha": 62000, "base_yield": 1.95},
    {"district": "Ngabwe", "province": "Central", "lat": -14.15, "lon": 27.50, "aez": "AEZ IIa", "area_ha": 19000, "base_yield": 1.80},
    {"district": "Serenje", "province": "Central", "lat": -13.23, "lon": 30.23, "aez": "AEZ III", "area_ha": 42000, "base_yield": 1.90},

    # COPPERBELT PROVINCE (10 districts) - High rainfall belt, peri-urban and commercial farming
    {"district": "Chililabombwe", "province": "Copperbelt", "lat": -12.37, "lon": 27.83, "aez": "AEZ III", "area_ha": 14000, "base_yield": 2.05},
    {"district": "Chingola", "province": "Copperbelt", "lat": -12.53, "lon": 27.85, "aez": "AEZ III", "area_ha": 16000, "base_yield": 2.10},
    {"district": "Kalulushi", "province": "Copperbelt", "lat": -12.83, "lon": 28.08, "aez": "AEZ III", "area_ha": 18000, "base_yield": 2.00},
    {"district": "Kitwe", "province": "Copperbelt", "lat": -12.80, "lon": 28.21, "aez": "AEZ III", "area_ha": 12000, "base_yield": 1.95},
    {"district": "Luanshya", "province": "Copperbelt", "lat": -13.13, "lon": 28.40, "aez": "AEZ III", "area_ha": 15000, "base_yield": 2.05},
    {"district": "Lufwanyama", "province": "Copperbelt", "lat": -12.87, "lon": 27.57, "aez": "AEZ III", "area_ha": 32000, "base_yield": 2.15},
    {"district": "Masaiti", "province": "Copperbelt", "lat": -13.17, "lon": 28.60, "aez": "AEZ III", "area_ha": 44000, "base_yield": 2.25},
    {"district": "Mpongwe", "province": "Copperbelt", "lat": -13.52, "lon": 28.15, "aez": "AEZ III", "area_ha": 68000, "base_yield": 2.70},
    {"district": "Mufulira", "province": "Copperbelt", "lat": -12.55, "lon": 28.24, "aez": "AEZ III", "area_ha": 13000, "base_yield": 1.95},
    {"district": "Ndola", "province": "Copperbelt", "lat": -12.97, "lon": 28.63, "aez": "AEZ III", "area_ha": 14000, "base_yield": 2.00},

    # EASTERN PROVINCE (15 districts) - Plateau maize, high smallholder density
    {"district": "Chadiza", "province": "Eastern", "lat": -14.07, "lon": 32.43, "aez": "AEZ IIa", "area_ha": 32000, "base_yield": 1.85},
    {"district": "Chama", "province": "Eastern", "lat": -11.21, "lon": 33.15, "aez": "AEZ III", "area_ha": 26000, "base_yield": 1.65},
    {"district": "Chasefu", "province": "Eastern", "lat": -12.10, "lon": 33.30, "aez": "AEZ IIa", "area_ha": 28000, "base_yield": 1.90},
    {"district": "Chipangali", "province": "Eastern", "lat": -13.35, "lon": 32.55, "aez": "AEZ IIa", "area_ha": 35000, "base_yield": 2.00},
    {"district": "Chipata", "province": "Eastern", "lat": -13.63, "lon": 32.65, "aez": "AEZ IIa", "area_ha": 54000, "base_yield": 2.15},
    {"district": "Kasenengwa", "province": "Eastern", "lat": -13.72, "lon": 32.18, "aez": "AEZ IIa", "area_ha": 36000, "base_yield": 1.95},
    {"district": "Katete", "province": "Eastern", "lat": -14.05, "lon": 32.05, "aez": "AEZ IIa", "area_ha": 48000, "base_yield": 2.05},
    {"district": "Lumezi", "province": "Eastern", "lat": -12.45, "lon": 33.18, "aez": "AEZ IIa", "area_ha": 31000, "base_yield": 1.80},
    {"district": "Lundazi", "province": "Eastern", "lat": -12.29, "lon": 33.18, "aez": "AEZ IIa", "area_ha": 45000, "base_yield": 1.95},
    {"district": "Lusangazi", "province": "Eastern", "lat": -13.85, "lon": 31.35, "aez": "AEZ I", "area_ha": 22000, "base_yield": 1.50},
    {"district": "Mambwe", "province": "Eastern", "lat": -13.35, "lon": 31.93, "aez": "AEZ I", "area_ha": 24000, "base_yield": 1.55},
    {"district": "Nyimba", "province": "Eastern", "lat": -14.55, "lon": 30.82, "aez": "AEZ I", "area_ha": 33000, "base_yield": 1.60},
    {"district": "Petauke", "province": "Eastern", "lat": -14.24, "lon": 31.33, "aez": "AEZ IIa", "area_ha": 62000, "base_yield": 2.10},
    {"district": "Sinda", "province": "Eastern", "lat": -14.17, "lon": 31.83, "aez": "AEZ IIa", "area_ha": 38000, "base_yield": 1.90},
    {"district": "Vubwi", "province": "Eastern", "lat": -14.12, "lon": 32.78, "aez": "AEZ IIa", "area_ha": 18000, "base_yield": 1.80},

    # LUAPULA PROVINCE (12 districts) - High rainfall AEZ III, cassava/maize belt
    {"district": "Chembe", "province": "Luapula", "lat": -11.95, "lon": 28.75, "aez": "AEZ III", "area_ha": 16000, "base_yield": 1.70},
    {"district": "Chiengi", "province": "Luapula", "lat": -8.65, "lon": 29.17, "aez": "AEZ III", "area_ha": 14000, "base_yield": 1.55},
    {"district": "Chifunabuli", "province": "Luapula", "lat": -11.60, "lon": 29.80, "aez": "AEZ III", "area_ha": 17000, "base_yield": 1.65},
    {"district": "Chipili", "province": "Luapula", "lat": -10.58, "lon": 29.13, "aez": "AEZ III", "area_ha": 19000, "base_yield": 1.75},
    {"district": "Kawambwa", "province": "Luapula", "lat": -9.79, "lon": 29.08, "aez": "AEZ III", "area_ha": 22000, "base_yield": 1.80},
    {"district": "Lunga", "province": "Luapula", "lat": -11.85, "lon": 29.95, "aez": "AEZ III", "area_ha": 9000, "base_yield": 1.40},
    {"district": "Mansa", "province": "Luapula", "lat": -11.20, "lon": 28.89, "aez": "AEZ III", "area_ha": 34000, "base_yield": 1.85},
    {"district": "Milenge", "province": "Luapula", "lat": -12.15, "lon": 29.35, "aez": "AEZ III", "area_ha": 18000, "base_yield": 1.65},
    {"district": "Mwansabombwe", "province": "Luapula", "lat": -9.83, "lon": 28.75, "aez": "AEZ III", "area_ha": 15000, "base_yield": 1.70},
    {"district": "Mwense", "province": "Luapula", "lat": -10.38, "lon": 28.70, "aez": "AEZ III", "area_ha": 20000, "base_yield": 1.75},
    {"district": "Nchelenge", "province": "Luapula", "lat": -9.35, "lon": 28.73, "aez": "AEZ III", "area_ha": 19000, "base_yield": 1.60},
    {"district": "Samfya", "province": "Luapula", "lat": -11.36, "lon": 29.55, "aez": "AEZ III", "area_ha": 21000, "base_yield": 1.65},

    # LUSAKA PROVINCE (7 districts)
    {"district": "Chilanga", "province": "Lusaka", "lat": -15.57, "lon": 28.27, "aez": "AEZ IIa", "area_ha": 28000, "base_yield": 2.20},
    {"district": "Chirundu", "province": "Lusaka", "lat": -16.03, "lon": 28.85, "aez": "AEZ I", "area_ha": 15000, "base_yield": 1.35},
    {"district": "Chongwe", "province": "Lusaka", "lat": -15.33, "lon": 28.68, "aez": "AEZ IIa", "area_ha": 48000, "base_yield": 2.15},
    {"district": "Luangwa", "province": "Lusaka", "lat": -15.62, "lon": 30.42, "aez": "AEZ I", "area_ha": 16000, "base_yield": 1.40},
    {"district": "Lusaka", "province": "Lusaka", "lat": -15.42, "lon": 28.28, "aez": "AEZ IIa", "area_ha": 12000, "base_yield": 2.10},
    {"district": "Rufunsa", "province": "Lusaka", "lat": -15.12, "lon": 29.42, "aez": "AEZ I", "area_ha": 21000, "base_yield": 1.45},
    {"district": "Shibuyunji", "province": "Lusaka", "lat": -15.22, "lon": 27.75, "aez": "AEZ IIa", "area_ha": 26000, "base_yield": 1.95},

    # MUCHINGA PROVINCE (8 districts)
    {"district": "Chinsali", "province": "Muchinga", "lat": -10.55, "lon": 32.08, "aez": "AEZ III", "area_ha": 31000, "base_yield": 1.85},
    {"district": "Isoka", "province": "Muchinga", "lat": -10.13, "lon": 32.63, "aez": "AEZ III", "area_ha": 28000, "base_yield": 1.90},
    {"district": "Kanchibiya", "province": "Muchinga", "lat": -11.55, "lon": 31.25, "aez": "AEZ III", "area_ha": 22000, "base_yield": 1.75},
    {"district": "Lavushimanda", "province": "Muchinga", "lat": -12.22, "lon": 31.05, "aez": "AEZ III", "area_ha": 20000, "base_yield": 1.70},
    {"district": "Mafinga", "province": "Muchinga", "lat": -9.95, "lon": 33.30, "aez": "AEZ III", "area_ha": 24000, "base_yield": 1.80},
    {"district": "Mpika", "province": "Muchinga", "lat": -11.83, "lon": 31.45, "aez": "AEZ III", "area_ha": 36000, "base_yield": 1.90},
    {"district": "Nakonde", "province": "Muchinga", "lat": -9.33, "lon": 32.75, "aez": "AEZ III", "area_ha": 29000, "base_yield": 1.95},
    {"district": "Shiwang'andu", "province": "Muchinga", "lat": -11.20, "lon": 31.75, "aez": "AEZ III", "area_ha": 23000, "base_yield": 1.80},

    # NORTHERN PROVINCE (12 districts) - High rainfall northern plateau
    {"district": "Chilubi", "province": "Northern", "lat": -11.15, "lon": 30.22, "aez": "AEZ III", "area_ha": 18000, "base_yield": 1.60},
    {"district": "Kaputa", "province": "Northern", "lat": -8.47, "lon": 29.67, "aez": "AEZ III", "area_ha": 16000, "base_yield": 1.55},
    {"district": "Kasama", "province": "Northern", "lat": -10.21, "lon": 31.18, "aez": "AEZ III", "area_ha": 46000, "base_yield": 2.05},
    {"district": "Lunte", "province": "Northern", "lat": -9.75, "lon": 30.55, "aez": "AEZ III", "area_ha": 21000, "base_yield": 1.75},
    {"district": "Lupososhi", "province": "Northern", "lat": -10.45, "lon": 30.15, "aez": "AEZ III", "area_ha": 22000, "base_yield": 1.70},
    {"district": "Luwingu", "province": "Northern", "lat": -10.26, "lon": 29.93, "aez": "AEZ III", "area_ha": 27000, "base_yield": 1.75},
    {"district": "Mbala", "province": "Northern", "lat": -8.84, "lon": 31.37, "aez": "AEZ III", "area_ha": 35000, "base_yield": 2.00},
    {"district": "Mporokoso", "province": "Northern", "lat": -9.38, "lon": 30.13, "aez": "AEZ III", "area_ha": 25000, "base_yield": 1.80},
    {"district": "Mpulungu", "province": "Northern", "lat": -8.76, "lon": 31.11, "aez": "AEZ III", "area_ha": 17000, "base_yield": 1.65},
    {"district": "Mungwi", "province": "Northern", "lat": -10.17, "lon": 31.37, "aez": "AEZ III", "area_ha": 33000, "base_yield": 1.90},
    {"district": "Nsama", "province": "Northern", "lat": -8.85, "lon": 30.05, "aez": "AEZ III", "area_ha": 15000, "base_yield": 1.60},
    {"district": "Senga Hill", "province": "Northern", "lat": -9.35, "lon": 31.40, "aez": "AEZ III", "area_ha": 29000, "base_yield": 1.85},

    # NORTH-WESTERN PROVINCE (11 districts) - High rainfall forest and copper belt
    {"district": "Chavuma", "province": "North-Western", "lat": -13.10, "lon": 22.68, "aez": "AEZ III", "area_ha": 15000, "base_yield": 1.60},
    {"district": "Ikelenge", "province": "North-Western", "lat": -11.23, "lon": 24.28, "aez": "AEZ III", "area_ha": 14000, "base_yield": 1.65},
    {"district": "Kabompo", "province": "North-Western", "lat": -13.60, "lon": 24.20, "aez": "AEZ III", "area_ha": 21000, "base_yield": 1.70},
    {"district": "Kalumbila", "province": "North-Western", "lat": -12.22, "lon": 25.32, "aez": "AEZ III", "area_ha": 24000, "base_yield": 1.85},
    {"district": "Kasempa", "province": "North-Western", "lat": -13.46, "lon": 25.83, "aez": "AEZ III", "area_ha": 26000, "base_yield": 1.80},
    {"district": "Manyinga", "province": "North-Western", "lat": -13.25, "lon": 24.45, "aez": "AEZ III", "area_ha": 18000, "base_yield": 1.70},
    {"district": "Mufumbwe", "province": "North-Western", "lat": -14.00, "lon": 24.80, "aez": "AEZ III", "area_ha": 19000, "base_yield": 1.75},
    {"district": "Mushindamo", "province": "North-Western", "lat": -12.35, "lon": 26.85, "aez": "AEZ III", "area_ha": 20000, "base_yield": 1.80},
    {"district": "Mwinilunga", "province": "North-Western", "lat": -11.73, "lon": 24.43, "aez": "AEZ III", "area_ha": 22000, "base_yield": 1.70},
    {"district": "Solwezi", "province": "North-Western", "lat": -12.18, "lon": 26.40, "aez": "AEZ III", "area_ha": 38000, "base_yield": 1.95},
    {"district": "Zambezi", "province": "North-Western", "lat": -13.54, "lon": 23.11, "aez": "AEZ III", "area_ha": 17000, "base_yield": 1.55},

    # SOUTHERN PROVINCE (15 districts) - Traditional maize basket, high vulnerability to heat & drought
    {"district": "Chikankata", "province": "Southern", "lat": -15.95, "lon": 28.30, "aez": "AEZ IIa", "area_ha": 38000, "base_yield": 2.10},
    {"district": "Choma", "province": "Southern", "lat": -16.81, "lon": 26.99, "aez": "AEZ IIa", "area_ha": 76000, "base_yield": 2.15},
    {"district": "Gwembe", "province": "Southern", "lat": -16.48, "lon": 27.61, "aez": "AEZ I", "area_ha": 24000, "base_yield": 1.30},
    {"district": "Kalomo", "province": "Southern", "lat": -17.03, "lon": 26.49, "aez": "AEZ IIa", "area_ha": 82000, "base_yield": 2.10},
    {"district": "Kazungula", "province": "Southern", "lat": -17.79, "lon": 25.86, "aez": "AEZ I", "area_ha": 41000, "base_yield": 1.50},
    {"district": "Livingstone", "province": "Southern", "lat": -17.85, "lon": 25.86, "aez": "AEZ I", "area_ha": 11000, "base_yield": 1.40},
    {"district": "Mazabuka", "province": "Southern", "lat": -15.86, "lon": 27.75, "aez": "AEZ IIa", "area_ha": 78000, "base_yield": 2.50},
    {"district": "Monze", "province": "Southern", "lat": -16.28, "lon": 27.48, "aez": "AEZ IIa", "area_ha": 69000, "base_yield": 2.20},
    {"district": "Munyumbwe", "province": "Southern", "lat": -16.65, "lon": 27.80, "aez": "AEZ I", "area_ha": 18000, "base_yield": 1.30},
    {"district": "Namwala", "province": "Southern", "lat": -15.75, "lon": 26.44, "aez": "AEZ IIa", "area_ha": 45000, "base_yield": 1.85},
    {"district": "Pemba", "province": "Southern", "lat": -16.53, "lon": 27.36, "aez": "AEZ IIa", "area_ha": 39000, "base_yield": 2.10},
    {"district": "Siavonga", "province": "Southern", "lat": -16.53, "lon": 28.71, "aez": "AEZ I", "area_ha": 19000, "base_yield": 1.25},
    {"district": "Sinazongwe", "province": "Southern", "lat": -17.26, "lon": 27.46, "aez": "AEZ I", "area_ha": 28000, "base_yield": 1.20},
    {"district": "Zimba", "province": "Southern", "lat": -17.32, "lon": 26.20, "aez": "AEZ IIa", "area_ha": 34000, "base_yield": 1.80},

    # WESTERN PROVINCE (16 districts) - Kalahari sands AEZ IIb, livestock & floodplains
    {"district": "Kalabo", "province": "Western", "lat": -14.99, "lon": 22.68, "aez": "AEZ IIb", "area_ha": 21000, "base_yield": 1.35},
    {"district": "Kaoma", "province": "Western", "lat": -14.78, "lon": 24.80, "aez": "AEZ IIa", "area_ha": 49000, "base_yield": 1.90},
    {"district": "Limulunga", "province": "Western", "lat": -15.10, "lon": 23.14, "aez": "AEZ IIb", "area_ha": 14000, "base_yield": 1.30},
    {"district": "Luampa", "province": "Western", "lat": -15.02, "lon": 24.50, "aez": "AEZ IIb", "area_ha": 23000, "base_yield": 1.55},
    {"district": "Lukulu", "province": "Western", "lat": -14.40, "lon": 23.25, "aez": "AEZ IIb", "area_ha": 25000, "base_yield": 1.40},
    {"district": "Mitete", "province": "Western", "lat": -14.18, "lon": 22.95, "aez": "AEZ IIb", "area_ha": 18000, "base_yield": 1.35},
    {"district": "Mongu", "province": "Western", "lat": -15.28, "lon": 23.13, "aez": "AEZ IIb", "area_ha": 26000, "base_yield": 1.45},
    {"district": "Mulobezi", "province": "Western", "lat": -16.78, "lon": 25.17, "aez": "AEZ IIb", "area_ha": 17000, "base_yield": 1.40},
    {"district": "Mwandi", "province": "Western", "lat": -17.55, "lon": 24.75, "aez": "AEZ I", "area_ha": 16000, "base_yield": 1.30},
    {"district": "Nalolo", "province": "Western", "lat": -15.65, "lon": 23.18, "aez": "AEZ IIb", "area_ha": 20000, "base_yield": 1.35},
    {"district": "Nkeyema", "province": "Western", "lat": -14.90, "lon": 25.30, "aez": "AEZ IIa", "area_ha": 35000, "base_yield": 1.85},
    {"district": "Senanga", "province": "Western", "lat": -16.12, "lon": 23.27, "aez": "AEZ IIb", "area_ha": 24000, "base_yield": 1.30},
    {"district": "Sesheke", "province": "Western", "lat": -17.47, "lon": 24.30, "aez": "AEZ I", "area_ha": 22000, "base_yield": 1.35},
    {"district": "Shang'ombo", "province": "Western", "lat": -16.32, "lon": 22.10, "aez": "AEZ IIb", "area_ha": 19000, "base_yield": 1.25},
    {"district": "Sikongo", "province": "Western", "lat": -15.00, "lon": 22.25, "aez": "AEZ IIb", "area_ha": 16000, "base_yield": 1.30},
    {"district": "Sioma", "province": "Western", "lat": -16.42, "lon": 23.35, "aez": "AEZ I", "area_ha": 21000, "base_yield": 1.25}
]

df_districts = pd.DataFrame(DISTRICTS_DATA)

def get_all_districts():
    """Returns the complete list of all 116 Zambian districts as dictionaries."""
    return df_districts.to_dict(orient="records")

def get_district_by_name(name: str):
    """Fetches metadata for a single district."""
    match = df_districts[df_districts["district"].str.lower() == name.lower().strip()]
    if not match.empty:
        return match.iloc[0].to_dict()
    return None

def get_provinces_summary():
    """Aggregates districts by province."""
    grouped = df_districts.groupby("province").agg(
        num_districts=("district", "count"),
        total_area_ha=("area_ha", "sum"),
        mean_base_yield=("base_yield", "mean")
    ).reset_index()
    return grouped.to_dict(orient="records")

# Historical CFS National Survey Timeline (1997-2026, 2027 is forecast)
HISTORICAL_NATIONAL_CFS = {
    1997: 1.60, 1998: 2.24, 1999: 2.02, 2000: 1.84, 2001: 2.38,
    2002: 2.03, 2003: 2.38, 2004: 2.12, 2005: 1.45, 2006: 1.89,
    2007: 2.22, 2008: 2.08, 2009: 2.05, 2010: 2.33, 2011: 2.47,
    2012: 2.25, 2013: 1.86, 2014: 2.02, 2015: 1.70, 2016: 2.21,
    2017: 1.98, 2018: 2.21, 2019: 1.89, 2020: 1.80, 2021: 1.93,
    2022: 1.56, 2023: 2.24, 2024: 1.39, 2025: 1.76, 2026: 2.16
    # 2027 is forward-looking forecast
}
