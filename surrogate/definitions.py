"""
Study definitions: inputs, case studies, output categories and evaluation criteria.
"""

REGION = "RE1"
INPUT_COLUMNS = [
    "demand_multiplier", "discount_rate", "fuel_cost_multiplier", "capital_cost_multiplier",
    "capacity_factor_multiplier", "iar_multiplier", "oar_multiplier",
    "water_accumulated_multiplier", "crop_accumulated_multiplier",
]
# Sampled range of each input (Latin hypercube sampling, the same for both countries).
INPUT_RANGES = {
    "demand_multiplier": (0.5, 1.5), "discount_rate": (0.07, 0.15), "fuel_cost_multiplier": (0.6, 2.0),
    "capital_cost_multiplier": (0.75, 1.3), "capacity_factor_multiplier": (0.75, 1.25),
    "iar_multiplier": (0.7, 1.3), "oar_multiplier": (0.7, 1.3),
    "water_accumulated_multiplier": (0.5, 1.5), "crop_accumulated_multiplier": (0.5, 1.5),
}
# How each input changes the model data (see apply_inputs):
#   demand          SpecifiedAnnualDemand x multiplier
#   discount rate   DiscountRate = value
#   fuel cost       VariableCost x multiplier for the fuel technologies and all land technologies (LND*)
#   capital cost    CapitalCost x multiplier
#   capacity factor CapacityFactor x multiplier for hydropower, solar and wind
#   IAR             InputActivityRatio x multiplier for the agricultural water commodities (AGRWAT*)
#   OAR             OutputActivityRatio x multiplier for the crop commodities (CRP*) in the modelled years
#   water / crop    AccumulatedAnnualDemand x multiplier for the water and crop commodities of the country
COUNTRIES = {
    "Zambia": {
        "years": list(range(2025, 2051)),
        "solve_time_min": 0.69,          # average CLEWs solution time per scenario, one core
        "fuel_techs": ["IMPCOA", "IMPGSL", "IMPHFO", "IMPNGS", "IMPURN", "MINCOA", "MINNGS", "MINURN",
                       "PWRCOA001", "PWRNGS001", "PWRNGS002", "PWROHC001", "PWROHC002", "PWROHC003",
                       "PWROHC004", "PWRNUC", "PWRBIO001"],
        "water_demand": ["PUBWATC01", "PUBWATC02", "PUBWATC03", "INDWATC01", "INDWATC02", "INDWATC03"],
        "crop_demand": ["CRPMAI", "CRPSOY", "CRPWHE"],
    },
    "Ghana": {
        "years": list(range(2022, 2071)),
        "solve_time_min": 1.29,
        "fuel_techs": ["IMPCOA", "IMPHFO", "IMPNGS", "IMPURN", "MINCOA", "MINNGS", "MINURN", "IMPLPG",
                       "IMPLFO", "PWRCOA001", "PWRNGS001", "PWRNGS002", "PWROHC001", "PWROHC002",
                       "PWROHC003", "PWRNUC", "PWRBIO001"],
        "water_demand": ["PUBWAT"],
        "crop_demand": ["CRPMAI", "CRPCAS", "CRPYAM", "CRPRIC", "CRPAGR"],
    },
}
CAPACITY_FACTOR_PREFIXES = ("PWRHYD", "PWRSOL", "PWRWND")

# Result files kept for every solved scenario; all other solver output is deleted.
RESULT_FILES = [
    "AnnualTechnologyEmission.csv", "UseByTechnologyAnnual.csv", "ProductionByTechnologyAnnual.csv",
    "AnnualFixedOperatingCost.csv", "AnnualVariableOperatingCost.csv", "TotalCapacityAnnual.csv",
    "TotalDiscountedCost.csv", "CapitalInvestment.csv", "DiscountedSalvageValue.csv",
    "DiscountedTechnologyEmissionsPenalty.csv",
]
# Total discounted cost (TDC) predicted by the surrogates: all technologies except forest and grassland.
TDC_EXCLUDE_PREFIXES = ("LNDFOR", "LNDGRS")

# Files of the OSeMOSYS-Solver-script (S. Kumar, MIT licence) used to convert and solve the models.
SOLVER_SCRIPT_URL = "https://raw.githubusercontent.com/ShravanKumar23/OSeMOSYS-Solver-script/main/"
SOLVER_SCRIPT_FILES = ["model.v.5.2.txt", "config_com.yaml", "preprocess_data.py"]

# ---- output categories of the Zambia surrogate: result file, filter on one column, and the
# technology groups (group name -> OSeMOSYS technologies that are summed)
LAND_CROPS = ["MAI", "CAS", "SOY", "OTC", "WHE"]
LAND_TYPES = ["HI", "HR", "LI", "LR"]            # high / low input, irrigated / rainfed
LAND_CLUSTERS = ["C01", "C02", "C03"]
LAND_GROUPS = {f"{crop}_{cl}": [f"LND{crop}{t}{cl}" for t in LAND_TYPES] for crop in LAND_CROPS for cl in LAND_CLUSTERS}
ZAMBIA_OUTPUTS = {
    "CAPACITY": {"label": "Installed capacity", "file": "TotalCapacityAnnual.csv", "filter": None,
                 "groups": {"HYD": ["PWRHYD001", "PWRHYD004"], "SOL": ["PWRSOL001", "PWRSOL002"],
                            "OHC": ["PWROHC001", "PWROHC002", "PWROHC003"], "TRN": ["PWRTRN"], "DIST": ["PWRDIST"],
                            "WND": ["PWRWND001"], "COA": ["PWRCOA001"], "BIO": ["PWRBIO001"], "GEO": ["PWRGEO"],
                            "TRNIMP": ["PWRTRNIMP"]}},
    "ELC001": {"label": "Electricity generation", "file": "ProductionByTechnologyAnnual.csv",
               "filter": ("COMMODITY", ["ELC001"]),
               "groups": {"HYD": ["PWRHYD001", "PWRHYD004"], "SOL": ["PWRSOL001", "PWRSOL002"],
                          "OHC": ["PWROHC001", "PWROHC002", "PWROHC003"], "WND": ["PWRWND001"], "GEO": ["PWRGEO"],
                          "COA": ["PWRCOA001"], "BIO": ["PWRBIO001"], "TRNIMP": ["PWRTRNIMP"]}},
    "EMICO2": {"label": "CO$_2$ emissions", "file": "AnnualTechnologyEmission.csv", "filter": ("EMISSION", ["EMICO2"]),
               "groups": {"COA": ["PWRCOA001"], "AGRDSL": ["DEMAGRDSL"]}},
    "LANDUSE": {"label": "Land use", "file": "UseByTechnologyAnnual.csv",
                "filter": ("COMMODITY", [f"LND{c}" for c in LAND_CLUSTERS]), "groups": LAND_GROUPS},
    "CROPPROD": {"label": "Crop production", "file": "ProductionByTechnologyAnnual.csv",     # imports not included
                 "filter": ("COMMODITY", [f"CRP{c}" for c in LAND_CROPS]), "groups": LAND_GROUPS},
    "AGRWATER": {"label": "Agricultural water use", "file": "UseByTechnologyAnnual.csv",    # rainfed types add 0
                 "filter": ("COMMODITY", [f"AGRWAT{c}" for c in LAND_CLUSTERS]), "groups": LAND_GROUPS},
    "FIXEDCOST": {"label": "Fixed operating cost", "file": "AnnualFixedOperatingCost.csv", "filter": None,
                  "groups": LAND_GROUPS},
    "VARCOST": {"label": "Variable operating cost", "file": "AnnualVariableOperatingCost.csv", "filter": None,
                "groups": LAND_GROUPS},
    "TDC": {"label": "Total discounted cost", "file": None, "filter": None, "groups": {"TOTAL": []}},
}
LATEX_LABEL = {k: v["label"].replace("CO$_2$", "CO\\textsubscript{2}") for k, v in ZAMBIA_OUTPUTS.items()}

# ---- output categories of the transfer: the same group NAMES in both countries (the technologies
# behind them may differ). Ghana has no land clusters, so the Zambia land outputs are summed over
# its clusters, and only the crops of both models (maize, cassava) are kept. CO2 emissions are not
# transferred, as the technologies that emit CO2 differ between the two models.
TRANSFER_CROPS = ["MAI", "CAS"]


def _transfer_land(country):
    if country == "Zambia":
        return {c: [f"LND{c}{t}{k}" for t in LAND_TYPES for k in LAND_CLUSTERS] for c in TRANSFER_CROPS}
    return {c: [f"LND{c}{t}" for t in LAND_TYPES] for c in TRANSFER_CROPS}


_POWER = {
    "Zambia": {"BIO": ["PWRBIO001"], "HYD": ["PWRHYD001", "PWRHYD004"], "WND": ["PWRWND001"],
               "OHC": ["PWROHC001", "PWROHC002", "PWROHC003"], "TRNIMP": ["PWRTRNIMP"],
               "SOL": ["PWRSOL001", "PWRSOL002"], "COA": ["PWRCOA001"]},
    "Ghana": {"BIO": ["PWRBIO001"], "HYD": ["PWRHYD001", "PWRHYD003", "PWRHYD004"], "WND": ["PWRWND001", "PWRWND001S"],
              "OHC": ["PWROHC001", "PWROHC002", "PWROHC003"], "TRNIMP": ["PWRTRNIMP"],
              "SOL": ["PWRSOL001", "PWRCSP001"], "COA": ["PWRCOA001"]},
}
_GRID = {"DIST": ["PWRDIST"], "TRN": ["PWRTRN"]}
TRANSFER_OUTPUTS = {
    "CAPACITY": {"file": "TotalCapacityAnnual.csv", "filter": {"Zambia": None, "Ghana": None},
                 "groups": {c: {**_POWER[c], **_GRID} for c in COUNTRIES}},
    "ELC001": {"file": "ProductionByTechnologyAnnual.csv", "filter": {c: ("COMMODITY", ["ELC001"]) for c in COUNTRIES},
               "groups": {c: dict(_POWER[c]) for c in COUNTRIES}},
    "LANDUSE": {"file": "UseByTechnologyAnnual.csv",
                "filter": {"Zambia": ("COMMODITY", [f"LND{k}" for k in LAND_CLUSTERS]), "Ghana": ("COMMODITY", ["LND"])},
                "groups": {c: _transfer_land(c) for c in COUNTRIES}},
    "CROPPROD": {"file": "ProductionByTechnologyAnnual.csv",
                 "filter": {c: ("COMMODITY", [f"CRP{x}" for x in TRANSFER_CROPS]) for c in COUNTRIES},
                 "groups": {c: _transfer_land(c) for c in COUNTRIES}},
    "AGRWATER": {"file": "UseByTechnologyAnnual.csv",
                 "filter": {"Zambia": ("COMMODITY", [f"AGRWAT{k}" for k in LAND_CLUSTERS]),
                            "Ghana": ("COMMODITY", ["AGRWAT"])},
                 "groups": {c: _transfer_land(c) for c in COUNTRIES}},
    "FIXEDCOST": {"file": "AnnualFixedOperatingCost.csv", "filter": {"Zambia": None, "Ghana": None},
                  "groups": {c: _transfer_land(c) for c in COUNTRIES}},
    "VARCOST": {"file": "AnnualVariableOperatingCost.csv", "filter": {"Zambia": None, "Ghana": None},
                "groups": {c: _transfer_land(c) for c in COUNTRIES}},
    "TDC": {"file": None, "filter": {"Zambia": None, "Ghana": None},
            "groups": {"Zambia": {"TOTAL": []}, "Ghana": {"TOTAL": []}}},
}

# ---- evaluation
ACTIVE_THRESHOLD = 1e-9          # a value above this counts as non-zero
EPS = 1e-12
ALPHA = 0.10                     # 90% prediction intervals
CALIBRATION_RUNS = 250           # calibration scenarios used for the intervals (both countries)
CLEAR_SHARE = 0.01               # pair accuracy: pairs must differ by > 1% of the 5-95% range
# Example thresholds of the two levels of use (Section 2.5); they can be changed for other purposes.
SCREENING_CRITERIA = {"pair_accuracy_clear_mean": (">=", 0.90)}
DETAILED_CRITERIA = {"r2_mean": (">=", 0.95), "wmape_mean": ("<=", 0.05), "half_width": ("<=", 0.10)}
