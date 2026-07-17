"""Generate expanded fuzzy rules (200+) for the Fuzzy Logic Agent."""

import yaml
from itertools import product

RULES = []

def rule(name, desc, priority, antecedents, consequents):
    RULES.append({
        "name": name,
        "description": desc,
        "priority": priority,
        "antecedents": antecedents,
        "consequents": consequents,
    })

def ant(var, fset):
    return {"var": var, "set": fset}

def cons_nutrient(nutrient, action, amount=0, unit=""):
    return {"type": "nutrient", "nutrient": nutrient, "action": action, "amount": amount, "unit": unit}

def cons_adj(var, fset):
    return {"type": "adjustment", "var": var, "set": fset}

def cons_risk(level):
    return {"type": "risk", "level": level}

def cons_conf(level):
    return {"type": "confidence", "level": level}

def n_adj(s): return cons_adj("Nitrogen_Adjustment", s)
def p_adj(s): return cons_adj("Potash_Adjustment", s)
def i_adj(s): return cons_adj("Irrigation_Adjustment", s)
def d_adj(s): return cons_adj("Dose_Adjustment", s)

# ── SINGLE-NUTRIENT DEFICIENCY RULES (N/P/K × low/medium × pH/rain/oc/temp combos) ──

n_statuses = [("low", "Nitrogen"), ("medium", "Nitrogen"), ("high", "Nitrogen")]
p_statuses = [("low", "Phosphorus"), ("medium", "Phosphorus"), ("high", "Phosphorus")]
k_statuses = [("low", "Potassium"), ("medium", "Potassium"), ("high", "Potassium")]

pH_sets = [("acidic", "Soil_pH"), ("neutral", "Soil_pH"), ("alkaline", "Soil_pH")]
rain_sets = [("low", "Rainfall"), ("medium", "Rainfall"), ("high", "Rainfall")]
temp_sets = [("cool", "Temperature_Max"), ("moderate", "Temperature_Max"), ("hot", "Temperature_Max")]
oc_sets = [("low", "Organic_Carbon"), ("medium", "Organic_Carbon"), ("high", "Organic_Carbon")]
growth_sets = [("seedling", "Growth_Stage"), ("vegetative", "Growth_Stage"), ("flowering", "Growth_Stage"), ("maturity", "Growth_Stage")]
zinc_sets = [("low", "Zinc"), ("medium", "Zinc"), ("high", "Zinc")]
yield_sets = [("low", "Yield_Prediction"), ("medium", "Yield_Prediction"), ("high", "Yield_Prediction")]

# 1. Single N deficiency + one environmental factor
priority = 10
for n_stat, n_name in [("low", "Nitrogen")]:
    for env_fset, env_var in pH_sets + rain_sets + oc_sets + temp_sets:
        rule_name = f"LowN_{env_var}_{env_fset}"
        cons = [
            cons_nutrient("Nitrogen", "increase", 20, "%"),
            cons_nutrient("Phosphorus", "maintain"),
            cons_nutrient("Potassium", "maintain"),
            n_adj("high"), d_adj("high"), i_adj("medium"),
            cons_risk("medium"), cons_conf("high"),
        ]
        if env_fset == "low" and env_var == "Rainfall":
            cons = [
                cons_nutrient("Nitrogen", "increase", 25, "%"),
                n_adj("high"), d_adj("high"), i_adj("high"),
                cons_risk("high"), cons_conf("medium"),
            ]
        elif env_fset == "high" and env_var == "Rainfall":
            cons = [
                cons_nutrient("Nitrogen", "increase", 20, "%"),
                n_adj("high"), d_adj("high"), i_adj("low"),
                cons_risk("medium"), cons_conf("high"),
            ]
        elif env_fset == "hot" and env_var == "Temperature_Max":
            cons = [
                cons_nutrient("Nitrogen", "increase", 15, "%"),
                n_adj("high"), d_adj("medium"), i_adj("high"),
                cons_risk("high"), cons_conf("medium"),
            ]
        rule(rule_name, f"Nitrogen low + {env_var}={env_fset}", priority, [ant("Nitrogen", n_stat), ant(env_var, env_fset)], cons)
priority -= 1

# 2. Single P deficiency + one environmental factor
for env_fset, env_var in pH_sets + rain_sets + oc_sets + temp_sets:
    rule_name = f"LowP_{env_var}_{env_fset}"
    cons = [
        cons_nutrient("Phosphorus", "increase", 15, "%"),
        cons_nutrient("Nitrogen", "maintain"),
        cons_nutrient("Potassium", "maintain"),
        n_adj("medium"), d_adj("high"), i_adj("medium"),
        cons_risk("medium"), cons_conf("high"),
    ]
    if env_fset == "acidic":
        cons = [
            cons_nutrient("Phosphorus", "increase", 20, "%"),
            n_adj("medium"), d_adj("high"), i_adj("medium"),
            cons_risk("high"), cons_conf("medium"),
        ]
    elif env_fset == "alkaline":
        cons = [
            cons_nutrient("Phosphorus", "apply_soil"),
            cons_nutrient("Zinc", "apply_foliar_spray"),
            n_adj("medium"), d_adj("medium"), i_adj("medium"),
            cons_risk("medium"), cons_conf("medium"),
        ]
    rule(rule_name, f"Phosphorus low + {env_var}={env_fset}", priority, [ant("Phosphorus", "low"), ant(env_var, env_fset)], cons)

# 3. Single K deficiency + one environmental factor
for env_fset, env_var in pH_sets + rain_sets + oc_sets + temp_sets:
    rule_name = f"LowK_{env_var}_{env_fset}"
    cons = [
        cons_nutrient("Potassium", "increase", 20, "%"),
        cons_nutrient("Nitrogen", "maintain"),
        p_adj("high"), d_adj("high"), i_adj("medium"),
        cons_risk("medium"), cons_conf("high"),
    ]
    if env_fset == "low" and env_var == "Rainfall":
        cons = [
            cons_nutrient("Potassium", "increase", 25, "%"),
            p_adj("high"), d_adj("high"), i_adj("high"),
            cons_risk("high"), cons_conf("medium"),
        ]
    rule(rule_name, f"Potassium low + {env_var}={env_fset}", priority, [ant("Potassium", "low"), ant(env_var, env_fset)], cons)

# ── TWO-NUTRIENT DEFICIENCY RULES ──
priority = 9
np_combos = [("low", "low"), ("low", "medium"), ("medium", "low")]
for n_stat, p_stat in np_combos:
    for env_fset, env_var in rain_sets[:2] + pH_sets[:2]:
        rule_name = f"N{n_stat}_P{p_stat}_{env_var}_{env_fset}"
        n_amount = 25 if n_stat == "low" else 0
        p_amount = 20 if p_stat == "low" else 0
        cons = [
            cons_nutrient("Nitrogen", "increase" if n_stat == "low" else "maintain", n_amount, "%"),
            cons_nutrient("Phosphorus", "increase" if p_stat == "low" else "maintain", p_amount, "%"),
            cons_nutrient("Potassium", "maintain"),
            n_adj("high" if n_stat == "low" else "medium"),
            p_adj("medium"), d_adj("high"), i_adj("medium"),
            cons_risk("high"), cons_conf("medium"),
        ]
        rule(rule_name, f"N={n_stat} P={p_stat} + {env_var}={env_fset}", priority,
             [ant("Nitrogen", n_stat), ant("Phosphorus", p_stat), ant(env_var, env_fset)], cons)

nk_combos = [("low", "low"), ("low", "medium"), ("medium", "low")]
for n_stat, k_stat in nk_combos:
    for env_fset, env_var in rain_sets[:2]:
        rule_name = f"N{n_stat}_K{k_stat}_{env_var}_{env_fset}"
        n_amount = 25 if n_stat == "low" else 0
        k_amount = 20 if k_stat == "low" else 0
        cons = [
            cons_nutrient("Nitrogen", "increase" if n_stat == "low" else "maintain", n_amount, "%"),
            cons_nutrient("Potassium", "increase" if k_stat == "low" else "maintain", k_amount, "%"),
            cons_nutrient("Phosphorus", "maintain"),
            n_adj("high" if n_stat == "low" else "medium"),
            p_adj("high" if k_stat == "low" else "medium"),
            d_adj("high"), i_adj("medium"),
            cons_risk("high"), cons_conf("medium"),
        ]
        rule(rule_name, f"N={n_stat} K={k_stat} + {env_var}={env_fset}", priority,
             [ant("Nitrogen", n_stat), ant("Potassium", k_stat), ant(env_var, env_fset)], cons)

pk_combos = [("low", "low"), ("low", "medium")]
for p_stat, k_stat in pk_combos:
    rule_name = f"P{p_stat}_K{k_stat}_combo"
    p_amount = 20 if p_stat == "low" else 0
    k_amount = 20 if k_stat == "low" else 0
    rule(rule_name, f"P={p_stat} K={k_stat} combo", priority,
         [ant("Phosphorus", p_stat), ant("Potassium", k_stat)],
         [cons_nutrient("Phosphorus", "increase" if p_stat == "low" else "maintain", p_amount, "%"),
          cons_nutrient("Potassium", "increase" if k_stat == "low" else "maintain", k_amount, "%"),
          cons_nutrient("Nitrogen", "maintain"),
          n_adj("medium"), p_adj("high" if k_stat == "low" else "medium"),
          d_adj("high"), i_adj("medium"),
          cons_risk("high"), cons_conf("medium")])

# ── ENVIRONMENTAL COMBINATION RULES ──
priority = 8
for t_fset, t_var in temp_sets:
    for r_fset, r_var in rain_sets:
        if t_fset == "hot" and r_fset == "low":
            rule(f"Hot_Dry", "Hot+dry: stress mitigation", 9,
                 [ant("Temperature_Max", "hot"), ant("Rainfall", "low")],
                 [cons_nutrient("Nitrogen", "maintain"), cons_nutrient("Potassium", "increase", 10, "%"),
                  n_adj("medium"), p_adj("high"), i_adj("high"), d_adj("low"),
                  cons_risk("high"), cons_conf("medium")])
        elif t_fset == "hot" and r_fset == "high":
            rule(f"Hot_Wet", "Hot+wet: disease risk", 8,
                 [ant("Temperature_Max", "hot"), ant("Rainfall", "high")],
                 [cons_nutrient("Nitrogen", "reduce", 10, "%"), cons_nutrient("Phosphorus", "maintain"),
                  n_adj("low"), d_adj("low"), i_adj("low"),
                  cons_risk("high"), cons_conf("medium")])
        elif t_fset == "cool" and r_fset == "high":
            rule(f"Cool_Wet", "Cool+wet: reduce irrigation", 7,
                 [ant("Temperature_Max", "cool"), ant("Rainfall", "high")],
                 [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), i_adj("low"), d_adj("medium"),
                  cons_risk("low"), cons_conf("high")])
        elif t_fset == "cool" and r_fset == "low":
            rule(f"Cool_Dry", "Cool+dry: protect seedlings", 7,
                 [ant("Temperature_Max", "cool"), ant("Rainfall", "low")],
                 [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), i_adj("high"), d_adj("medium"),
                  cons_risk("medium"), cons_conf("medium")])
        elif t_fset == "moderate" and r_fset == "medium":
            rule(f"Moderate_MediumRain", "Moderate temp + medium rain: ideal", 5,
                 [ant("Temperature_Max", "moderate"), ant("Rainfall", "medium")],
                 [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), i_adj("medium"), d_adj("medium"),
                  cons_risk("low"), cons_conf("high")])
        elif t_fset == "moderate" and r_fset == "high":
            rule(f"Moderate_HighRain", "Moderate temp + high rain", 6,
                 [ant("Temperature_Max", "moderate"), ant("Rainfall", "high")],
                 [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), i_adj("low"), d_adj("medium"),
                  cons_risk("low"), cons_conf("high")])
        elif t_fset == "moderate" and r_fset == "low":
            rule(f"Moderate_LowRain", "Moderate temp + low rain", 6,
                 [ant("Temperature_Max", "moderate"), ant("Rainfall", "low")],
                 [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), i_adj("high"), d_adj("medium"),
                  cons_risk("medium"), cons_conf("medium")])

# ── pH × NUTRIENT RULES ──
priority = 8
for ph_fset, ph_var in pH_sets:
    for n_stat in ["low", "medium", "high"]:
        rule_name = f"pH_{ph_fset}_N{n_stat}"
        if ph_fset == "acidic" and n_stat == "low":
            cons = [cons_nutrient("Nitrogen", "increase", 25, "%"), n_adj("high"), d_adj("high"),
                    cons_risk("high"), cons_conf("medium")]
        elif ph_fset == "alkaline" and n_stat == "low":
            cons = [cons_nutrient("Nitrogen", "increase", 20, "%"), n_adj("high"), d_adj("high"),
                    cons_nutrient("Zinc", "apply_foliar_spray"),
                    cons_risk("medium"), cons_conf("high")]
        elif ph_fset == "alkaline" and n_stat == "high":
            cons = [cons_nutrient("Nitrogen", "reduce", 10, "%"), n_adj("low"), d_adj("low"),
                    cons_nutrient("Zinc", "monitor"),
                    cons_risk("medium"), cons_conf("high")]
        elif ph_fset == "acidic" and n_stat == "high":
            cons = [cons_nutrient("Nitrogen", "reduce", 15, "%"), n_adj("low"), d_adj("low"),
                    cons_risk("medium"), cons_conf("medium")]
        else:
            cons = [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), d_adj("medium"),
                    cons_risk("low"), cons_conf("high")]
        rule(rule_name, f"pH={ph_fset} + N={n_stat}", priority,
             [ant("Soil_pH", ph_fset), ant("Nitrogen", n_stat)], cons)

# ── ORGANIC CARBON RULES ──
priority = 7
for oc_fset in ["low", "medium", "high"]:
    for n_stat in ["low", "medium", "high"]:
        rule_name = f"OC_{oc_fset}_N{n_stat}"
        if oc_fset == "low" and n_stat == "low":
            cons = [cons_nutrient("Nitrogen", "increase", 30, "%"), n_adj("high"), d_adj("high"),
                    cons_risk("high"), cons_conf("low")]
        elif oc_fset == "low" and n_stat == "medium":
            cons = [cons_nutrient("Nitrogen", "increase", 15, "%"), n_adj("medium"), d_adj("medium"),
                    cons_risk("medium"), cons_conf("medium")]
        elif oc_fset == "high" and n_stat == "high":
            cons = [cons_nutrient("Nitrogen", "reduce", 10, "%"), n_adj("low"), d_adj("low"),
                    cons_risk("low"), cons_conf("high")]
        else:
            cons = [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), d_adj("medium"),
                    cons_risk("low"), cons_conf("high")]
        rule(rule_name, f"OC={oc_fset} + N={n_stat}", priority,
             [ant("Organic_Carbon", oc_fset), ant("Nitrogen", n_stat)], cons)

# ── ZINC DEFICIENCY RULES ──
priority = 8
for z_fset in ["low", "medium", "high"]:
    for ph_fset in ["acidic", "neutral", "alkaline"]:
        rule_name = f"Zinc_{z_fset}_pH_{ph_fset}"
        if z_fset == "low" and ph_fset == "alkaline":
            cons = [cons_nutrient("Zinc", "apply_foliar_spray"), d_adj("medium"),
                    cons_risk("high"), cons_conf("high")]
        elif z_fset == "low":
            cons = [cons_nutrient("Zinc", "apply_soil"), d_adj("medium"),
                    cons_risk("medium"), cons_conf("medium")]
        elif z_fset == "high":
            cons = [cons_nutrient("Zinc", "monitor"), d_adj("low"),
                    cons_risk("low"), cons_conf("high")]
        else:
            cons = [cons_nutrient("Zinc", "maintain"), d_adj("medium"),
                    cons_risk("low"), cons_conf("high")]
        rule(rule_name, f"Zinc={z_fset} + pH={ph_fset}", priority,
             [ant("Zinc", z_fset), ant("Soil_pH", ph_fset)], cons)

# ── GROWTH STAGE RULES ──
priority = 7
for gs_fset, gs_name in growth_sets:
    for n_stat in ["low", "medium", "high"]:
        rule_name = f"Stage_{gs_fset}_N{n_stat}"
        if gs_fset == "seedling" and n_stat == "low":
            cons = [cons_nutrient("Nitrogen", "increase", 15, "%"), n_adj("medium"), d_adj("medium"),
                    cons_risk("medium"), cons_conf("medium")]
        elif gs_fset == "flowering" and n_stat == "low":
            cons = [cons_nutrient("Nitrogen", "increase", 20, "%"), cons_nutrient("Phosphorus", "increase", 10, "%"),
                    n_adj("high"), d_adj("high"),
                    cons_risk("high"), cons_conf("medium")]
        elif gs_fset == "flowering" and n_stat == "medium":
            cons = [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), d_adj("medium"),
                    cons_risk("low"), cons_conf("high")]
        elif gs_fset == "maturity" and n_stat == "high":
            cons = [cons_nutrient("Nitrogen", "reduce", 20, "%"), n_adj("low"), d_adj("low"),
                    cons_risk("low"), cons_conf("high")]
        elif gs_fset == "maturity":
            cons = [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), d_adj("medium"),
                    cons_risk("low"), cons_conf("high")]
        else:
            cons = [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), d_adj("medium"),
                    cons_risk("low"), cons_conf("high")]
        rule(rule_name, f"Stage={gs_fset} + N={n_stat}", priority,
             [ant("Growth_Stage", gs_fset), ant("Nitrogen", n_stat)], cons)

# ── YIELD PREDICTION RULES ──
priority = 7
for y_fset in ["low", "medium", "high"]:
    for n_stat in ["low", "medium", "high"]:
        rule_name = f"Yield_{y_fset}_N{n_stat}"
        if y_fset == "low" and n_stat == "high":
            cons = [cons_nutrient("Nitrogen", "review"), n_adj("low"), d_adj("low"),
                    cons_risk("high"), cons_conf("medium")]
        elif y_fset == "low" and n_stat == "low":
            cons = [cons_nutrient("Nitrogen", "increase", 30, "%"), n_adj("high"), d_adj("high"),
                    cons_risk("high"), cons_conf("low")]
        elif y_fset == "high" and n_stat == "medium":
            cons = [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), d_adj("medium"),
                    cons_risk("low"), cons_conf("high")]
        elif y_fset == "high" and n_stat == "high":
            cons = [cons_nutrient("Nitrogen", "reduce", 15, "%"), n_adj("low"), d_adj("low"),
                    cons_risk("low"), cons_conf("high")]
        else:
            cons = [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), d_adj("medium"),
                    cons_risk("low"), cons_conf("high")]
        rule(rule_name, f"Yield={y_fset} + N={n_stat}", priority,
             [ant("Yield_Prediction", y_fset), ant("Nitrogen", n_stat)], cons)

# ── THREE-FACTOR COMBINATION RULES ──
priority = 6
for n_stat, p_stat, k_stat in product(["low", "medium", "high"], repeat=3):
    rule_name = f"N{n_stat}_P{p_stat}_K{k_stat}_tri"
    n_act = "increase" if n_stat == "low" else "reduce" if n_stat == "high" else "maintain"
    p_act = "increase" if p_stat == "low" else "reduce" if p_stat == "high" else "maintain"
    k_act = "increase" if k_stat == "low" else "reduce" if k_stat == "high" else "maintain"
    n_amt = 25 if n_stat == "low" else 15 if n_stat == "high" else 0
    p_amt = 20 if p_stat == "low" else 10 if p_stat == "high" else 0
    k_amt = 20 if k_stat == "low" else 10 if k_stat == "high" else 0
    risk = "high" if sum(1 for s in [n_stat, p_stat, k_stat] if s == "low") >= 2 else "medium" if any(s == "low" for s in [n_stat, p_stat, k_stat]) else "low"
    conf = "high" if all(s == "medium" for s in [n_stat, p_stat, k_stat]) else "medium"
    rule(rule_name, f"N={n_stat} P={p_stat} K={k_stat} 3-factor", priority,
         [ant("Nitrogen", n_stat), ant("Phosphorus", p_stat), ant("Potassium", k_stat)],
         [cons_nutrient("Nitrogen", n_act, n_amt, "%"), cons_nutrient("Phosphorus", p_act, p_amt, "%"),
          cons_nutrient("Potassium", k_act, k_amt, "%"),
          n_adj("high" if n_stat == "low" else "low" if n_stat == "high" else "medium"),
          p_adj("high" if k_stat == "low" else "low" if k_stat == "high" else "medium"),
          d_adj("high" if any(s == "low" for s in [n_stat, p_stat, k_stat]) else "medium"),
          i_adj("medium"), cons_risk(risk), cons_conf(conf)])

# ── pH × RAINFALL × TEMP RULES ──
priority = 6
for ph_fset in ["acidic", "neutral", "alkaline"]:
    for r_fset in ["low", "medium", "high"]:
        for t_fset in ["cool", "moderate", "hot"]:
            rule_name = f"pH_{ph_fset}_R{r_fset}_T{t_fset}"
            risk = "high" if (t_fset == "hot" and r_fset == "low") or (t_fset == "cool" and r_fset == "low") else "medium" if (t_fset == "hot" or r_fset == "low" or r_fset == "high") else "low"
            i_adj_val = "high" if r_fset == "low" else "low" if r_fset == "high" else "medium"
            d_adj_val = "low" if t_fset == "hot" and r_fset == "high" else "high" if t_fset == "hot" and r_fset == "low" else "medium"
            rule(rule_name, f"pH={ph_fset} Rain={r_fset} Temp={t_fset}", priority,
                 [ant("Soil_pH", ph_fset), ant("Rainfall", r_fset), ant("Temperature_Max", t_fset)],
                 [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), i_adj(i_adj_val), d_adj(d_adj_val),
                  cons_risk(risk), cons_conf("medium" if risk != "low" else "high")])

# ── ADDITIONAL SINGLE-FACTOR RULES ──
priority = 5
for n_stat in ["low", "medium", "high"]:
    rule(f"Single_N_{n_stat}", f"N={n_stat} alone", priority, [ant("Nitrogen", n_stat)],
         [cons_nutrient("Nitrogen", "increase" if n_stat == "low" else "reduce" if n_stat == "high" else "maintain", 20 if n_stat == "low" else 10 if n_stat == "high" else 0, "%"),
          n_adj("high" if n_stat == "low" else "low" if n_stat == "high" else "medium"), d_adj("medium"), i_adj("medium"),
          cons_risk("low" if n_stat == "medium" else "medium"), cons_conf("high" if n_stat == "medium" else "medium")])

for p_stat in ["low", "medium", "high"]:
    rule(f"Single_P_{p_stat}", f"P={p_stat} alone", priority, [ant("Phosphorus", p_stat)],
         [cons_nutrient("Phosphorus", "increase" if p_stat == "low" else "reduce" if p_stat == "high" else "maintain", 20 if p_stat == "low" else 10 if p_stat == "high" else 0, "%"),
          n_adj("medium"), d_adj("medium"), i_adj("medium"),
          cons_risk("low" if p_stat == "medium" else "medium"), cons_conf("high" if p_stat == "medium" else "medium")])

for k_stat in ["low", "medium", "high"]:
    rule(f"Single_K_{k_stat}", f"K={k_stat} alone", priority, [ant("Potassium", k_stat)],
         [cons_nutrient("Potassium", "increase" if k_stat == "low" else "reduce" if k_stat == "high" else "maintain", 20 if k_stat == "low" else 10 if k_stat == "high" else 0, "%"),
          p_adj("high" if k_stat == "low" else "low" if k_stat == "high" else "medium"), d_adj("medium"), i_adj("medium"),
          cons_risk("low" if k_stat == "medium" else "medium"), cons_conf("high" if k_stat == "medium" else "medium")])

for ph_fset in ["acidic", "neutral", "alkaline"]:
    rule(f"Single_pH_{ph_fset}", f"pH={ph_fset} alone", priority, [ant("Soil_pH", ph_fset)],
         [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), d_adj("medium"), i_adj("medium"),
          cons_risk("low" if ph_fset == "neutral" else "medium"), cons_conf("high" if ph_fset == "neutral" else "medium")])

for r_fset in ["low", "medium", "high"]:
    rule(f"Single_Rain_{r_fset}", f"Rain={r_fset} alone", priority, [ant("Rainfall", r_fset)],
         [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"),
          i_adj("high" if r_fset == "low" else "low" if r_fset == "high" else "medium"),
          d_adj("medium"), cons_risk("low" if r_fset == "medium" else "medium"), cons_conf("high")])

for t_fset in ["cool", "moderate", "hot"]:
    rule(f"Single_Temp_{t_fset}", f"Temp={t_fset} alone", priority, [ant("Temperature_Max", t_fset)],
         [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"),
          i_adj("high" if t_fset == "hot" else "medium"),
          d_adj("low" if t_fset == "hot" else "medium"),
          cons_risk("high" if t_fset == "hot" else "low" if t_fset == "moderate" else "medium"),
          cons_conf("high" if t_fset == "moderate" else "medium")])

for z_fset in ["low", "medium", "high"]:
    rule(f"Single_Zinc_{z_fset}", f"Zinc={z_fset} alone", priority, [ant("Zinc", z_fset)],
         [cons_nutrient("Zinc", "apply_foliar_spray" if z_fset == "low" else "monitor" if z_fset == "high" else "maintain"),
          d_adj("medium" if z_fset == "low" else "low"),
          cons_risk("medium" if z_fset == "low" else "low"), cons_conf("high" if z_fset != "low" else "medium")])

for oc_fset in ["low", "medium", "high"]:
    rule(f"Single_OC_{oc_fset}", f"OC={oc_fset} alone", priority, [ant("Organic_Carbon", oc_fset)],
         [cons_nutrient("Nitrogen", "increase" if oc_fset == "low" else "reduce" if oc_fset == "high" else "maintain", 10 if oc_fset == "low" else 0, "%"),
          n_adj("medium" if oc_fset == "low" else "low" if oc_fset == "high" else "medium"),
          d_adj("medium"), i_adj("medium"),
          cons_risk("low" if oc_fset != "low" else "medium"), cons_conf("high" if oc_fset != "low" else "medium")])

for gs_fset in ["seedling", "vegetative", "flowering", "maturity"]:
    rule(f"Single_Stage_{gs_fset}", f"Stage={gs_fset} alone", priority, [ant("Growth_Stage", gs_fset)],
         [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), d_adj("medium"), i_adj("medium"),
          cons_risk("low"), cons_conf("high")])

for y_fset in ["low", "medium", "high"]:
    rule(f"Single_Yield_{y_fset}", f"Yield={y_fset} alone", priority, [ant("Yield_Prediction", y_fset)],
         [cons_nutrient("Nitrogen", "maintain"), n_adj("medium"), d_adj("medium"), i_adj("medium"),
          cons_risk("low" if y_fset != "low" else "high"), cons_conf("high" if y_fset == "high" else "medium")])

# ── GROWTH STAGE × P/K RULES ──
priority = 6
for gs_fset in ["seedling", "vegetative", "flowering", "maturity"]:
    for p_stat in ["low", "high"]:
        rule(f"Stage_{gs_fset}_P{p_stat}", f"Stage={gs_fset} + P={p_stat}", priority,
             [ant("Growth_Stage", gs_fset), ant("Phosphorus", p_stat)],
             [cons_nutrient("Phosphorus", "increase" if p_stat == "low" else "reduce", 20 if p_stat == "low" else 10, "%"),
              n_adj("medium"), d_adj("medium"), i_adj("medium"),
              cons_risk("low" if gs_fset != "flowering" else "medium"), cons_conf("medium")])

for gs_fset in ["seedling", "vegetative", "flowering", "maturity"]:
    for k_stat in ["low", "high"]:
        rule(f"Stage_{gs_fset}_K{k_stat}", f"Stage={gs_fset} + K={k_stat}", priority,
             [ant("Growth_Stage", gs_fset), ant("Potassium", k_stat)],
             [cons_nutrient("Potassium", "increase" if k_stat == "low" else "reduce", 20 if k_stat == "low" else 10, "%"),
              p_adj("high" if k_stat == "low" else "low"), d_adj("medium"), i_adj("medium"),
              cons_risk("medium" if k_stat == "low" else "low"), cons_conf("medium")])

# ── OC × pH COMBO RULES ──
for oc_fset in ["low", "high"]:
    for ph_fset in ["acidic", "alkaline"]:
        rule(f"OC_{oc_fset}_pH_{ph_fset}", f"OC={oc_fset} + pH={ph_fset}", priority,
             [ant("Organic_Carbon", oc_fset), ant("Soil_pH", ph_fset)],
             [cons_nutrient("Nitrogen", "increase" if oc_fset == "low" else "maintain", 15 if oc_fset == "low" else 0, "%"),
              n_adj("high" if oc_fset == "low" else "medium"), d_adj("medium"), i_adj("medium"),
              cons_risk("high" if oc_fset == "low" and ph_fset == "acidic" else "medium"), cons_conf("medium")])

# ── ZINC × ENVIRONMENT RULES ──
for z_fset in ["low"]:
    for r_fset in ["low", "high"]:
        rule(f"Zinc_{z_fset}_Rain_{r_fset}", f"Zinc low + Rain={r_fset}", priority,
             [ant("Zinc", z_fset), ant("Rainfall", r_fset)],
             [cons_nutrient("Zinc", "apply_foliar_spray"),
              d_adj("medium"), i_adj("high" if r_fset == "low" else "low"),
              cons_risk("high" if r_fset == "low" else "medium"), cons_conf("medium")])

# ── ALL-NUTRIENT OPTIMAL RULES ──
rule("All_Optimal_NP_Kmedium", "All N,P medium, K medium", 3,
     [ant("Nitrogen", "medium"), ant("Phosphorus", "medium"), ant("Potassium", "medium")],
     [cons_nutrient("Nitrogen", "maintain"), cons_nutrient("Phosphorus", "maintain"), cons_nutrient("Potassium", "maintain"),
      n_adj("medium"), p_adj("medium"), i_adj("medium"), d_adj("medium"),
      cons_risk("low"), cons_conf("high")])

rule("All_High_NPK", "All N,P,K high: reduce all", 4,
     [ant("Nitrogen", "high"), ant("Phosphorus", "high"), ant("Potassium", "high")],
     [cons_nutrient("Nitrogen", "reduce", 20, "%"), cons_nutrient("Phosphorus", "reduce", 15, "%"), cons_nutrient("Potassium", "reduce", 15, "%"),
      n_adj("low"), p_adj("low"), i_adj("medium"), d_adj("low"),
      cons_risk("medium"), cons_conf("high")])

rule("All_Low_NPK", "All N,P,K low: emergency boost", 10,
     [ant("Nitrogen", "low"), ant("Phosphorus", "low"), ant("Potassium", "low")],
     [cons_nutrient("Nitrogen", "increase", 30, "%"), cons_nutrient("Phosphorus", "increase", 25, "%"), cons_nutrient("Potassium", "increase", 25, "%"),
      n_adj("high"), p_adj("high"), i_adj("medium"), d_adj("high"),
      cons_risk("high"), cons_conf("low")])

# ── WRITE YAML ──
output = {
    "metadata": {
        "version": "3.0",
        "description": "Expanded fuzzy expert rules (200+) for agronomist-style recommendations",
        "inference": "Mamdani + symbolic rule evaluation",
        "defuzzification": "centroid",
        "total_rules": len(RULES),
    },
    "variables": {
        "input": ["Nitrogen", "Phosphorus", "Potassium", "Zinc", "Soil_pH",
                   "Rainfall", "Temperature_Max", "Organic_Carbon", "Growth_Stage", "Yield_Prediction"],
        "output_numeric": ["Nitrogen_Adjustment", "Potash_Adjustment", "Irrigation_Adjustment", "Dose_Adjustment"],
        "output_symbolic": ["N_Action", "P_Action", "K_Action", "Zinc_Action", "Risk", "Confidence"],
    },
    "rules": RULES,
}

with open("agri_ai_agent/rules/fertilizer_rules.yaml", "w") as f:
    yaml.dump(output, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

print(f"Generated {len(RULES)} rules")
