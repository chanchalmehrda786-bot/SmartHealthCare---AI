"""
Smart HealthCare AI — Streamlit Web App
Run: streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
from datetime import date
from tensorflow.keras.models import load_model

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart HealthCare AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=DM+Sans:wght@400;500;600&display=swap');

.stApp {
    background: linear-gradient(135deg, #050d1f 0%, #0b1a3d 40%, #0a2545 70%, #040c1a 100%);
    background-attachment: fixed;
    font-family: 'DM Sans', sans-serif;
}
.stApp::before {
    content: '';
    position: fixed;
    top:0;left:0;right:0;bottom:0;
    background:
        radial-gradient(ellipse at 15% 20%, rgba(0,100,255,0.13) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 80%, rgba(0,180,140,0.09) 0%, transparent 55%),
        radial-gradient(ellipse at 70% 5%,  rgba(80,0,200,0.06) 0%, transparent 40%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(4,12,30,0.97) !important;
    border-right: 1px solid rgba(0,130,255,0.2) !important;
}
[data-testid="stSidebar"] * { color: #d0e4ff !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 { color: #ffffff !important; }

/* ── Global text ── */
h1,h2,h3,h4 { color:#ffffff !important; font-family:'Outfit',sans-serif !important; }
p, label, span { color: #c8dcff; }

/* ── Hero ── */
.hero-box {
    background: linear-gradient(135deg, rgba(0,80,220,0.28), rgba(0,180,160,0.18));
    border: 1px solid rgba(0,140,255,0.4);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero-box::after {
    content:'🩺';
    position:absolute; right:2.5rem; top:50%;
    transform:translateY(-50%);
    font-size:7rem; opacity:.06; pointer-events:none;
}
.hero-box h1 {
    font-family:'Outfit',sans-serif !important;
    font-size:2.7rem !important; font-weight:800 !important;
    background:linear-gradient(135deg,#ffffff 30%,#7dd8ff);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    margin:0 !important;
}
.hero-box p { color:#90c4f8 !important; margin:.5rem 0 0; font-size:1rem; }

/* ── Stat cards ── */
.stat-card {
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(0,140,255,0.22);
    border-radius:14px; padding:1.1rem 1.3rem;
}
.stat-card .lbl { font-size:.7rem; color:#5da4ff !important; text-transform:uppercase; letter-spacing:.1em; font-weight:600; }
.stat-card .val { font-size:2rem; font-weight:800; color:#ffffff !important; font-family:'Outfit',sans-serif; }

/* ── Glass card ── */
.glass-card {
    background:rgba(255,255,255,0.035);
    border:1px solid rgba(0,140,255,0.18);
    border-radius:16px; padding:1.6rem 1.8rem;
    margin-bottom:1.4rem;
}
.glass-card h3 { color:#7dd8ff !important; margin-top:0; font-size:1.15rem !important; }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input {
    background:rgba(255,255,255,0.06) !important;
    border:1px solid rgba(0,140,255,0.3) !important;
    border-radius:10px !important; color:#ffffff !important;
}
.stDateInput input {
    background:rgba(255,255,255,0.06) !important;
    border:1px solid rgba(0,140,255,0.3) !important;
    color:#ffffff !important; border-radius:10px !important;
}
.stSelectbox > div > div {
    background:rgba(255,255,255,0.06) !important;
    border:1px solid rgba(0,140,255,0.3) !important;
    border-radius:10px !important; color:#ffffff !important;
}
.stMultiSelect > div > div {
    background:rgba(255,255,255,0.06) !important;
    border:1px solid rgba(0,140,255,0.3) !important;
    border-radius:10px !important;
}
.stMultiSelect span { color:#e0eaff !important; }
.stMultiSelect [data-baseweb="tag"] {
    background:rgba(0,120,255,0.3) !important; color:#ffffff !important;
}
.stTextArea textarea {
    background:rgba(255,255,255,0.06) !important;
    border:1px solid rgba(0,140,255,0.3) !important;
    color:#ffffff !important; border-radius:10px !important;
}
.stTextInput label, .stNumberInput label, .stSelectbox label,
.stMultiSelect label, .stDateInput label, .stTextArea label,
.stRadio label { color:#a8c8ff !important; font-weight:500; }

/* ── Button ── */
div.stButton > button {
    background:linear-gradient(135deg,#0052e0,#00b4d8) !important;
    color:white !important; border:none !important;
    border-radius:12px !important; padding:.75rem 2rem !important;
    font-size:1.08rem !important; font-weight:700 !important;
    font-family:'Outfit',sans-serif !important; width:100% !important;
    box-shadow:0 4px 22px rgba(0,100,255,0.38) !important;
    transition:all .2s !important;
}
div.stButton > button:hover { transform:translateY(-2px) !important; }

/* ── Result card ── */
.result-card {
    background:linear-gradient(135deg,rgba(0,190,110,0.14),rgba(0,140,255,0.10));
    border:2px solid rgba(0,210,120,0.45);
    border-radius:20px; padding:2rem; text-align:center;
}
.result-card .disease-name {
    font-family:'Outfit',sans-serif; font-size:2.1rem; font-weight:800; color:#00ff9d !important;
}
.result-card .conf-text { color:#a0e4c0 !important; font-size:.97rem; }
.result-card .note-text { color:#6ea8ff !important; font-size:.74rem; margin-top:.8rem; }

/* ── Severity ── */
.sev-low      { color:#00ff9d !important; font-weight:700; }
.sev-medium   { color:#ffd166 !important; font-weight:700; }
.sev-high     { color:#ff6b6b !important; font-weight:700; }
.sev-critical { color:#ff44ff !important; font-weight:700; }

/* ── Info boxes ── */
.cause-box {
    background:rgba(255,90,90,0.07);
    border:1px solid rgba(255,90,90,0.28);
    border-radius:12px; padding:1rem 1.2rem; margin-top:.8rem;
}
.cause-box .ttl { color:#ff9999 !important; font-weight:700; font-size:.92rem; margin-bottom:.5rem; display:block; }
.cause-box li  { color:#ffcccc !important; font-size:.84rem; margin:.3rem 0; line-height:1.5; }

.advice-box {
    background:rgba(255,200,60,0.07);
    border:1px solid rgba(255,200,60,0.3);
    border-radius:12px; padding:1rem 1.2rem; margin-top:.8rem;
}
.advice-box .ttl { color:#ffd166 !important; font-weight:700; font-size:.92rem; margin-bottom:.5rem; display:block; }
.advice-box li  { color:#ffe999 !important; font-size:.84rem; margin:.3rem 0; line-height:1.5; }

.action-box {
    background:rgba(0,140,255,0.08);
    border:1px solid rgba(0,140,255,0.28);
    border-radius:12px; padding:1rem 1.2rem; margin-top:.8rem;
}
.action-box .ttl { color:#7dd8ff !important; font-weight:700; font-size:.92rem; margin-bottom:.4rem; display:block; }
.action-box p   { color:#b8e0ff !important; font-size:.86rem; margin:0; }

.extra-box {
    background:rgba(80,180,255,0.07);
    border:1px solid rgba(80,180,255,0.28);
    border-radius:12px; padding:.9rem 1.1rem; margin-top:.8rem;
}
.extra-box .ttl { color:#7dd8ff !important; font-weight:700; font-size:.88rem; }
.extra-box p    { color:#b8e0ff !important; font-size:.84rem; margin:.3rem 0 0; }

/* ── Warn box ── */
.warn-box {
    background:rgba(255,150,0,0.09);
    border:1px solid rgba(255,150,0,0.32);
    border-radius:10px; padding:.75rem 1rem;
    font-size:.8rem; color:#ffd580 !important;
}

/* ── Sidebar title ── */
.sb-title {
    font-family:'Outfit',sans-serif; font-size:1.45rem; font-weight:800;
    color:#ffffff !important; text-align:center; line-height:1.2;
}

/* scrollbar */
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background:rgba(0,140,255,0.35); border-radius:3px; }
</style>
""", unsafe_allow_html=True)

# ── LOAD RESOURCES ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔬 Loading AI model…")
def load_resources():
    for p in ["model/model.keras","model/model.h5","Model/model.keras",
              "Model/model.h5","Model/disease_model.h5","model.keras","model.h5"]:
        if os.path.exists(p):
            mdl = load_model(p); break
    else:
        st.error("❌ Model not found. Run: python train_model.py"); st.stop()

    for p in ["model/encoder.pkl","Model/encoder.pkl","Model/label_encoder.pkl","encoder.pkl"]:
        if os.path.exists(p):
            enc = joblib.load(p); break
    else:
        st.error("❌ Encoder not found."); st.stop()

    for p in ["data/Training.csv","Data/Training.csv","Training.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            df = df.loc[:, ~df.columns.str.startswith("Unnamed")]; break
    else:
        st.error("❌ Training.csv not found."); st.stop()

    return mdl, enc, df

model, encoder, train_df = load_resources()
FEATURE_COLS    = [c for c in train_df.columns if c != "prognosis"]
ALL_SYM_DISPLAY = sorted([s.replace("_"," ").title() for s in FEATURE_COLS])
SYM_DISP_TO_COL = {s.replace("_"," ").title(): s for s in FEATURE_COLS}

# ── DISEASE DATABASE ─────────────────────────────────────────────────────────
# Each entry: (icon, severity, immediate_action, [advice list], [causes list])
DISEASE_DB = {
    "Diabetes ": ("🩸","High",
        "See an endocrinologist immediately. Blood glucose test urgently needed.",
        ["Monitor blood glucose levels every day","Follow a low-sugar, low-carb diet strictly","Exercise at least 30 minutes daily","Take prescribed medication without skipping","Drink at least 2-3 litres of water daily","Avoid processed and packaged foods"],
        ["High-sugar and refined carbohydrate diet","Sedentary lifestyle and lack of exercise","Obesity or being overweight","Genetic predisposition or family history","Chronic stress and hormonal imbalances","Insulin resistance over time"]),
    "Heart attack": ("❤️","Critical",
        "EMERGENCY — Call 112/911 immediately. Do NOT drive yourself.",
        ["Chew aspirin (325mg) if available and not allergic","Sit or lie in a comfortable position immediately","Loosen all tight clothing around chest and neck","Stay calm — panic increases heart strain","Do not eat or drink anything until help arrives"],
        ["Blocked coronary arteries due to cholesterol plaque","High blood pressure left untreated for years","Smoking — damages blood vessels significantly","Obesity and unhealthy diet high in saturated fat","Diabetes — damages arteries over time","Chronic stress and sedentary lifestyle"]),
    "Hypertension ": ("📈","High",
        "Consult a cardiologist. Start home BP monitoring immediately.",
        ["Reduce salt intake to less than 5g per day","Exercise 30 minutes daily — walking, cycling, or swimming","Quit smoking and limit alcohol completely","Practice stress management: yoga, meditation, deep breathing","Take prescribed antihypertensive medications daily"],
        ["Excess salt and sodium in diet","Obesity — increases pressure on blood vessels","Sedentary lifestyle with little to no exercise","Chronic stress and anxiety","Smoking and heavy alcohol consumption","Kidney disease or genetic factors"]),
    "Pneumonia": ("🫁","High",
        "See a doctor immediately. Hospitalization may be required.",
        ["Complete bed rest — do not exert yourself","Drink 2-3 litres of warm fluids daily","Complete the FULL antibiotic course — never stop early","Avoid cold air, smoke, and dusty environments","Use steam inhalation to ease breathing"],
        ["Bacterial infection — most commonly Streptococcus pneumoniae","Viral infection — influenza, COVID-19, RSV","Weakened immune system from illness or medications","Smoking — damages lung defence mechanisms","Air pollution and exposure to chemical fumes"]),
    "Malaria": ("🦟","High",
        "Go to hospital urgently. Can be life-threatening if treatment is delayed.",
        ["Complete the full antimalarial drug course without skipping","Stay indoors at dawn and dusk when mosquitoes are most active","Use mosquito nets while sleeping every night","Apply mosquito repellent on exposed skin","Remove all stagnant water near your home"],
        ["Bite from Anopheles mosquito carrying Plasmodium parasite","Stagnant water in flowerpots, tyres, or containers near home","Travel to malaria-endemic areas without prevention","Not using mosquito nets or repellents","Weakened immune system"]),
    "Tuberculosis": ("🫁","High",
        "See a pulmonologist immediately. TB is curable with proper treatment.",
        ["Take ALL TB medications for 6 full months — never skip even one dose","Wear a surgical mask around others until declared non-infectious","Eat high-protein nutritious food to rebuild immunity","Open windows for ventilation — TB spreads in closed spaces","Get all household members tested"],
        ["Mycobacterium tuberculosis bacteria spread via air","Close contact with an infected person who coughs or sneezes","Weakened immune system — HIV, diabetes, malnutrition","Overcrowded living conditions with poor ventilation","Not completing a previous TB treatment course"]),
    "AIDS": ("⚠️","Critical",
        "Consult an HIV/ID specialist immediately. ART therapy is highly effective today.",
        ["Start antiretroviral therapy (ART) as soon as possible","Take ART medication at the same time every single day — never miss","Eat nutritious, immune-boosting food — fruits, vegetables, proteins","Regular CD4 count and viral load monitoring every 3-6 months","Avoid infections by maintaining strict personal hygiene"],
        ["HIV virus transmitted via unprotected sexual contact","Sharing needles or drug injection equipment","Infected blood transfusion (rare in modern medical settings)","Mother to child during birth or breastfeeding","NOT spread through casual contact, hugging, or sharing food"]),
    "Common Cold": ("🤧","Low",
        "Rest at home. See a doctor only if fever persists beyond 3 days.",
        ["Drink warm ginger-honey-lemon tea 3-4 times daily","Take steam inhalation twice daily to relieve congestion","Get 8-9 hours of sleep to help immune system fight the virus","OTC antihistamines or decongestants help with symptoms","Avoid cold foods, cold water, and air conditioning"],
        ["Rhinovirus — responsible for 40% of common colds","Coronavirus, RSV, and other respiratory viruses","Spread through infected droplets from coughs and sneezes","Touching contaminated surfaces then touching face","Low immunity from lack of sleep, stress, or poor diet"]),
    "Dengue": ("🦟","High",
        "Go to hospital for platelet count monitoring. Can become life-threatening.",
        ["Complete bed rest — any activity can worsen bleeding risk","Drink 2-3 litres of fluids and ORS frequently throughout the day","Get platelet count tested every day during fever","Use only paracetamol for fever — NEVER ibuprofen or aspirin","Use mosquito net even indoors — Aedes bites during the day"],
        ["Bite from Aedes aegypti mosquito — active during daytime","Stagnant water in flower pots, old tyres, coolers near home","Previous dengue infection increases risk of severe dengue","Dense urban areas with poor mosquito control","Not using mosquito repellents or protective clothing"]),
    "Migraine": ("🧠","Medium",
        "See a neurologist for proper diagnosis and prescription medication.",
        ["Rest immediately in a dark, quiet, and cool room","Apply a cold compress or ice pack to your forehead","Take prescribed triptan medication at the very first sign of migraine","Identify your triggers and strictly avoid them","Maintain consistent sleep schedule — even on weekends"],
        ["Genetic predisposition — migraines often run in families","Hormonal changes — especially during menstruation or ovulation","Triggers: stress, irregular sleep, certain foods (caffeine, red wine, aged cheese)","Bright or flickering lights, loud noises, strong smells","Skipping meals or dehydration"]),
    "Allergy": ("🌿","Low",
        "Allergy testing recommended to identify your specific triggers.",
        ["Take prescribed antihistamines at the same time daily","Avoid identified allergens strictly — pollen, dust, pet dander","Keep windows closed during high pollen season","Use an air purifier with HEPA filter at home","Carry epinephrine auto-injector (EpiPen) if prescribed for severe reactions"],
        ["Immune system overreaction to harmless substances","Common allergens: pollen, dust mites, pet dander, mold","Food allergens: peanuts, shellfish, dairy, eggs, wheat","Medications: antibiotics (penicillin), NSAIDs","Insect stings from bees or wasps"]),
    "Fungal infection": ("🍄","Low",
        "Antifungal cream or oral tablets available at pharmacy or doctor.",
        ["Keep the affected area completely clean and dry at all times","Apply antifungal cream for the FULL recommended course — even if better","Wear loose, breathable cotton clothing to allow air circulation","Do not share towels, socks, or shoes with anyone","Change socks and underwear daily"],
        ["Warm and moist conditions on the skin surface","Poor personal hygiene or infrequent bathing","Sharing personal items like towels or footwear","Weakened immune system or uncontrolled diabetes","Prolonged use of antibiotics killing protective bacteria"]),
    "GERD": ("🔥","Medium",
        "See a gastroenterologist for proper diagnosis and endoscopy.",
        ["Eat 5-6 small meals instead of 3 large heavy meals","Do not lie down or sleep within 2 hours of eating","Elevate the head end of your bed by 6-8 inches","Avoid spicy, oily, acidic, and fried foods completely","Avoid alcohol, carbonated drinks, coffee, and chocolate"],
        ["Weak lower esophageal sphincter (LES) allowing acid reflux","Obesity — increases abdominal pressure on the stomach","Hiatal hernia — stomach bulges into the chest","Spicy and fatty diet consumed regularly","Lying down immediately after meals"]),
    "Jaundice": ("💛","High",
        "Urgent liver function tests and hepatologist consultation needed today.",
        ["Complete bed rest — avoid all physical exertion","Avoid ALL alcohol — even a small amount damages the liver further","Eat light, easily digestible foods: khichdi, dal, boiled vegetables","Drink 3 litres of clean water daily to flush toxins","Avoid oily, fried, and heavily spiced foods completely"],
        ["Viral hepatitis (A, B, C, D, E) infecting liver cells","Bile duct blockage from gallstones or tumour","Haemolytic anaemia causing excess bilirubin in blood","Alcohol abuse over extended period damaging liver","Certain medications causing liver toxicity"]),
    "Typhoid": ("🦠","High",
        "Hospitalization may be needed. Blood culture test required urgently.",
        ["Complete the full antibiotic course — typically 10-14 days","Drink ONLY boiled water or sealed bottled water","Eat only soft, home-cooked food — avoid all street food","Wash hands thoroughly with soap before every meal","Rest completely — avoid all strenuous activities"],
        ["Salmonella typhi bacteria in contaminated water or food","Poor sanitation and open sewage contaminating water supply","Consuming street food or unwashed raw vegetables","Inadequate handwashing after using the toilet","Travel to areas with poor sanitation without proper vaccination"]),
    "Chicken pox": ("💊","Medium",
        "Rest at home. Highly contagious — isolate from others for 7 days.",
        ["Apply calamine lotion on blisters to relieve itching","Trim fingernails short to prevent scratching and skin infection","Take antiviral medication (acyclovir) if prescribed early","Take cool baths with baking soda or oatmeal to soothe skin","Avoid contact with pregnant women, newborns, and immune-compromised people"],
        ["Varicella-zoster virus (VZV) — extremely contagious","Airborne transmission through coughs, sneezes, and contact with blisters","Not having received the chickenpox vaccine","Close contact with an infected person","Weakened immune system making infection more severe"]),
    "Arthritis": ("🦴","Medium",
        "See a rheumatologist for joint assessment and treatment plan.",
        ["Do gentle physiotherapy exercises daily to maintain joint mobility","Apply warm compress for morning stiffness, cold pack for swelling","Follow an anti-inflammatory diet — turmeric, omega-3, fresh vegetables","Maintain a healthy weight — every extra kg puts 4x pressure on knee joints","Take prescribed NSAIDs or DMARDs as directed"],
        ["Autoimmune response attacking joint lining (Rheumatoid Arthritis)","Progressive cartilage wear from aging (Osteoarthritis)","Joint injury or trauma in the past","Obesity putting excessive stress on weight-bearing joints","Genetic predisposition or family history of arthritis"]),
    "Gastroenteritis": ("🤢","Medium",
        "See a doctor if symptoms are severe, or for children and elderly.",
        ["Take ORS (oral rehydration solution) in small frequent sips — do not gulp","Eat only bland BRAT diet: Banana, Rice, Applesauce, Toast","Avoid dairy, fatty, spicy, and raw foods until fully recovered","Wash hands with soap for 20 seconds before and after toilet","Complete rest — avoid going to work or school to prevent spreading"],
        ["Norovirus or rotavirus — most common causes worldwide","Bacterial infection from E. coli, Salmonella, or Campylobacter","Contaminated food left unrefrigerated for too long","Drinking contaminated or unfiltered water","Poor hand hygiene after using the toilet"]),
    "Bronchial Asthma": ("💨","High",
        "See a pulmonologist for proper inhaler prescription and action plan.",
        ["Always carry your reliever (blue) inhaler everywhere you go","Use preventer (brown/red) inhaler daily — even when feeling well","Avoid ALL triggers: dust, smoke, pet dander, cold air, strong perfumes","Cover your nose and mouth in cold weather before going outdoors","Check indoor air quality and use air purifier if needed"],
        ["Chronic airway inflammation from allergens like dust and pollen","Air pollution — vehicle emissions, industrial smoke, indoor smoke","Cigarette smoking — active or passive (secondhand smoke)","Respiratory infections in childhood increasing airway sensitivity","Genetics — family history of asthma or allergic conditions"]),
    "Urinary tract infection": ("💧","Medium",
        "See a doctor. Untreated UTI can spread to kidneys and become serious.",
        ["Drink minimum 2-3 litres of water every day — do not wait to feel thirsty","Complete the full antibiotic course — do not stop when symptoms improve","Urinate frequently — never hold urine for long periods","Urinate immediately after sexual activity","Wipe front to back after toilet use to prevent bacterial spread"],
        ["E. coli bacteria from bowel entering the urinary tract (80% of cases)","Poor genital hygiene practices","Holding urine for too long allowing bacteria to multiply","Catheter use in medical settings","Diabetes — high sugar in urine promotes bacterial growth"]),
    "Psoriasis": ("🧴","Medium",
        "See a dermatologist for topical or systemic treatment evaluation.",
        ["Moisturize skin at least 2-3 times daily with fragrance-free cream","Use prescribed topical steroids or vitamin D analogues as directed","Avoid harsh soaps, hot water, and skin irritants","Get moderate sunlight exposure — 10-15 minutes daily can help","Manage stress through yoga, meditation, or counselling"],
        ["Autoimmune condition causing abnormally rapid skin cell production","Genetic predisposition — family history increases risk significantly","Stress triggering or worsening flare-ups","Infections — streptococcal throat infection often precedes flares","Smoking, alcohol, and certain medications like lithium or beta-blockers"]),
    "Acne": ("😣","Low",
        "See a dermatologist if severe or leaving scars. OTC products for mild acne.",
        ["Wash face gently twice daily with a mild non-comedogenic cleanser","Never pop or squeeze pimples — causes scarring and spreads bacteria","Use oil-free, non-comedogenic moisturizer and sunscreen daily","Change pillowcase every 2-3 days","Avoid touching your face with unwashed hands throughout the day"],
        ["Hormonal changes during puberty, menstruation, or pregnancy","Excess sebum (oil) production from overactive sebaceous glands","Propionibacterium acnes bacteria on the skin","Clogged hair follicles from dead skin cells and oil","High-glycemic diet — white bread, sugar, dairy in some people"]),
    "Hypothyroidism": ("🧬","Medium",
        "Endocrinologist for thyroid function tests (TSH, Free T3, Free T4).",
        ["Take levothyroxine on empty stomach 30 minutes before breakfast","Never skip doses — thyroid hormone must be replaced consistently","Avoid excessive raw cruciferous vegetables (cabbage, kale) as they interfere","Get thyroid tests (TSH) checked every 6 months","Eat iodine-rich foods: seafood, iodized salt, dairy products"],
        ["Autoimmune Hashimoto's thyroiditis — most common cause worldwide","Iodine deficiency in diet — especially in inland areas","Previous thyroid surgery or radioactive iodine treatment","Certain medications: lithium, amiodarone, interferon","Radiation therapy to head or neck region"]),
    "Hyperthyroidism": ("🧬","Medium",
        "See an endocrinologist urgently — can cause serious heart complications.",
        ["Take anti-thyroid medication (methimazole/PTU) exactly as prescribed","Avoid excess iodine — reduce seafood and iodized salt intake","Get adequate rest and avoid physical overexertion","Beta-blockers may be prescribed to control heart rate — take regularly","Report any new symptoms immediately to your endocrinologist"],
        ["Grave's disease — autoimmune antibodies stimulating the thyroid","Toxic multinodular goiter — multiple overactive thyroid nodules","Thyroiditis — inflammation causing release of stored hormones","Excess iodine intake from diet or contrast dye in scans","Taking too much thyroid hormone medication"]),
    "Hypoglycemia": ("🩸","High",
        "Take glucose immediately. Adjust your medications with endocrinologist.",
        ["Immediately eat 15g fast-acting carbs: 3-4 glucose tablets or half cup fruit juice","Recheck blood glucose after 15 minutes — repeat if still low","Then eat a small balanced snack to maintain levels","Never skip meals — eat at regular, consistent intervals every day","Always carry glucose tablets or juice when going out"],
        ["Excessive insulin dose or oral hypoglycemic medication","Skipping or significantly delaying a meal","Excessive physical activity without adequate food intake","Heavy alcohol consumption especially without eating","Liver disease impairing glucose production"]),
    "Osteoarthristis": ("🦴","Medium",
        "See an orthopedic specialist for joint evaluation and management plan.",
        ["Do low-impact exercise daily: swimming, cycling, or water aerobics","Maintain a healthy weight — losing even 5kg significantly reduces joint pain","Apply hot compress for stiffness and cold pack for swelling","Use a walking aid if needed to reduce joint stress","Physiotherapy and specific joint exercises as guided by therapist"],
        ["Progressive wear of joint cartilage — primary cause is aging","Obesity putting excessive mechanical stress on joints","Previous joint injuries or fractures","Repetitive joint overuse in certain occupations or sports","Genetic factors and family history of joint disease"]),
    "Paralysis (brain hemorrhage)": ("🧠","Critical",
        "EMERGENCY — Call 112/911 immediately. Every minute causes irreversible brain damage.",
        ["Do NOT give food, water, or any medication by mouth","Keep the patient calm, still, and lying on their side if unconscious","Loosen all tight clothing around neck, chest, and waist","Note the exact time symptoms started — critical information for doctors","Do NOT let the patient sleep until medically assessed"],
        ["Uncontrolled high blood pressure rupturing a blood vessel","Head trauma or severe injury to the skull","Cerebral aneurysm — weakened blood vessel wall bursting","Blood thinning medications increasing bleeding risk","Arteriovenous malformation (AVM) — abnormal vessel structure"]),
    "Cervical spondylosis": ("🏥","Medium",
        "See an orthopedic doctor or neurologist for proper evaluation.",
        ["Do daily neck stretching and physiotherapy exercises as guided","Apply heat therapy to neck for pain and stiffness relief","Use an ergonomic chair and keep monitor at eye level","Take regular breaks from screen — look away every 20 minutes","Sleep on a firm mattress with a cervical pillow for proper neck support"],
        ["Age-related degeneration of cervical discs — most common cause","Prolonged poor posture from desk work or phone use","Previous neck injury or whiplash trauma","Heavy and repeated lifting putting strain on cervical spine","Genetic predisposition to early spinal degeneration"]),
    "Varicose veins": ("🦵","Medium",
        "Consult a vascular surgeon for treatment options.",
        ["Elevate your legs above heart level for 15-20 minutes several times daily","Wear graduated compression stockings from morning to bedtime","Walk and exercise regularly — avoid standing or sitting for long periods","Maintain a healthy weight to reduce venous pressure","Avoid crossing legs when sitting"]),
    "(vertigo) Paroymsal  Positional Vertigo": ("😵","Medium",
        "See an ENT specialist. BPPV is highly treatable with the Epley maneuver.",
        ["Perform the Epley maneuver only with professional doctor guidance","Move slowly and deliberately when getting up from lying down","Sleep with your head slightly elevated on two pillows","Avoid sudden head movements — turn your whole body instead","Do balance exercises as prescribed by your physiotherapist"],
        ["Calcium carbonate crystals (otoliths) displaced into semicircular canals","Aging — most common cause of BPPV after age 50","Head trauma or inner ear injury","Prolonged bed rest dislodging the crystals","Previous inner ear infections or inflammation"]),
    "Hepatitis B": ("🫀","High",
        "See a hepatologist urgently. Hepatitis B is manageable with antiviral treatment.",
        ["Take antiviral medication (tenofovir or entecavir) every day as prescribed","Avoid ALL alcohol — even small amounts damage the liver further","Ensure all family members and close contacts get vaccinated","Regular liver function tests and HBV DNA monitoring every 3-6 months","Eat a nutritious liver-friendly diet: fruits, vegetables, lean protein"],
        ["HBV virus transmitted through infected blood (needles, transfusions)","Unprotected sexual contact with an infected person","Mother to child transmission during childbirth","Sharing personal items that may have blood: razors, toothbrushes","NOT spread by hugging, sharing food, coughing, or casual contact"]),
    "Hepatitis C": ("🫀","High",
        "See a hepatologist urgently. New DAA drugs cure over 95% of cases.",
        ["Take the full direct-acting antiviral (DAA) course — 8 to 12 weeks","Never miss a dose — consistent blood levels are critical for cure","Avoid ALL alcohol completely during and after treatment","Do not share needles, syringes, razors, or toothbrushes","Get regular liver function tests and HCV RNA monitoring"],
        ["HCV transmitted primarily through sharing needles or drug equipment","Blood transfusions before 1992 (before screening was introduced)","Unsterilized medical or dental equipment in some settings","Healthcare worker accidental needlestick injury","Sexual contact — less common but possible with multiple partners"]),
    "Hepatitis D": ("🫀","High",
        "See a hepatologist. HDV only occurs alongside Hepatitis B — both must be treated.",
        ["Interferon-based therapy as prescribed by hepatologist","Take Hepatitis B treatment — controlling HBV helps control HDV","Avoid ALL alcohol — critical for liver recovery","Regular liver function monitoring every 3 months","Prevent HBV infection in others through vaccination — this prevents HDV too"],
        ["HDV requires HBV to replicate — cannot exist without HBV","Transmitted through infected blood or unprotected sexual contact","Sharing needles or drug injection equipment","Only occurs in people who already have Hepatitis B infection","No vaccine exists specifically for HDV — HBV vaccination prevents it"]),
    "Hepatitis E": ("🫀","High",
        "See a doctor. Especially dangerous in pregnant women — seek urgent care.",
        ["Complete bed rest during the acute phase of illness","Drink 2-3 litres of safe, boiled, or bottled water daily","Avoid all alcohol for at least 6 months after recovery","Eat light, easily digestible foods — avoid fatty and spicy foods","Most healthy adults recover fully within 4-6 weeks"],
        ["HEV transmitted through contaminated water — most common route","Consuming undercooked pork, wild boar, or shellfish","Traveling to areas with poor sanitation (Asia, Africa, Middle East)","Poor hand hygiene after toilet use","Contaminated food from street vendors or open markets"]),
    "hepatitis A": ("🫀","Medium",
        "Doctor visit for supportive care. Usually self-limiting with full recovery.",
        ["Complete rest — avoid all strenuous activity during illness","Avoid all alcohol for at least 6 months — liver needs time to heal","Eat small, light, nutritious meals — avoid heavy or fatty foods","Drink 2-3 litres of clean water daily","Strict hand hygiene — wash hands for 20 seconds after toilet"],
        ["HAV in contaminated food or water — most common route","Poor hand hygiene after using the toilet","Consuming raw or undercooked shellfish (oysters, clams)","Traveling to regions with poor water sanitation","Close contact with an infected person at home"]),
    "Alcoholic hepatitis": ("🍺","High",
        "Stop ALL alcohol immediately. See a hepatologist urgently — can be life-threatening.",
        ["Permanent and complete alcohol abstinence — this is the primary treatment","High-calorie, high-protein diet to support liver regeneration","Vitamin B1 (thiamine) supplementation — prescribed by doctor","Corticosteroids may be prescribed for severe cases — take as directed","Regular liver function monitoring — monthly initially"],
        ["Exclusively caused by excessive long-term alcohol consumption","Amount and duration of drinking are the key determining factors","Genetic factors affecting how the body processes alcohol","Malnutrition — commonly accompanies heavy alcohol use","Pre-existing liver conditions worsened by alcohol"]),
    "Chronic cholestasis": ("🟡","High",
        "See a gastroenterologist for proper diagnosis and liver workup.",
        ["Take ursodeoxycholic acid (UDCA) exactly as prescribed","Follow a strict low-fat diet to reduce strain on bile flow","Supplement with fat-soluble vitamins A, D, E, and K as prescribed","Avoid all alcohol and hepatotoxic drugs — paracetamol in large doses","Regular liver function tests and imaging every 6 months"],
        ["Primary biliary cholangitis (PBC) — autoimmune bile duct destruction","Primary sclerosing cholangitis (PSC) — bile duct scarring","Gallstones blocking bile flow from the liver","Certain medications causing drug-induced cholestasis","Viral hepatitis or other liver diseases"]),
    "Drug Reaction": ("💊","High",
        "Stop the suspected drug IMMEDIATELY. Go to emergency if breathing difficulty or swelling.",
        ["Discontinue the causative drug at once — do not take another dose","Take antihistamines (cetirizine) for mild rashes as recommended","For anaphylaxis — use epinephrine auto-injector (EpiPen) immediately","Inform ALL future doctors and pharmacists of this drug allergy","Wear a medical alert bracelet indicating the drug allergy"],
        ["Adverse immune reaction to medications — can occur with any drug","Most common triggers: penicillin antibiotics, sulfa drugs, NSAIDs","Anticonvulsants (carbamazepine, phenytoin) cause severe skin reactions","Allopurinol and contrast dye used in CT scans","History of other allergies increases the risk significantly"]),
    "Peptic ulcer diseae": ("🔥","Medium",
        "See a gastroenterologist for endoscopy and H. pylori test. Highly treatable.",
        ["Complete the full PPI (proton pump inhibitor) + antibiotic course for H. pylori","Avoid NSAIDs and aspirin completely — use paracetamol instead if needed","Eat 5-6 small meals daily — do not leave stomach empty for long","Avoid spicy, acidic, and fried foods, alcohol, and carbonated drinks","Quit smoking — it delays ulcer healing significantly"],
        ["H. pylori bacterial infection — responsible for 70% of peptic ulcers","Chronic overuse of NSAIDs or aspirin damaging stomach lining","Excess stomach acid production (Zollinger-Ellison syndrome)","Smoking — significantly impairs the stomach's protective mucous lining","Heavy alcohol consumption eroding the stomach lining"]),
    "Dimorphic hemmorhoids(piles)": ("🩺","Medium",
        "See a proctologist if conservative treatment fails or bleeding is heavy.",
        ["Eat a high-fibre diet: fruits, vegetables, whole grains, legumes daily","Drink minimum 2-3 litres of water daily — dehydration worsens constipation","Take a warm sitz bath for 15 minutes twice daily to reduce swelling","Never strain during bowel movements — use a step stool (squatting position)","Use prescribed topical anaesthetic or steroid creams for pain relief"],
        ["Chronic constipation and straining during bowel movements","Low-fibre diet causing hard stools that damage anal veins","Prolonged sitting — especially on the toilet for extended periods","Pregnancy — increased pressure on pelvic veins","Obesity and genetic predisposition to weak vein walls"]),
    "Impetigo": ("🩹","Low",
        "See a doctor for antibiotic prescription. Highly contagious — keep child at home.",
        ["Apply mupirocin antibiotic cream to affected area as prescribed","Keep the area clean with antiseptic soap and warm water twice daily","Cover sores loosely with a sterile bandage to prevent spreading","Do not share towels, bedding, or clothing with anyone in the household","Wash hands thoroughly after touching the affected area"],
        ["Staphylococcus aureus bacteria — most common cause","Streptococcus pyogenes entering through cuts or insect bites","Poor personal hygiene especially in children","Warm and humid weather conditions favouring bacterial growth","Close contact in schools, nurseries, or crowded living situations"]),
}
DEFAULT_DB = ("⚕️","Medium",
    "Consult a qualified healthcare professional for proper diagnosis.",
    ["Rest adequately and maintain proper hydration","Avoid self-medication — consult a doctor before taking any drugs","Keep a diary of your symptoms to share with your doctor"],
    ["Multiple factors may contribute to your condition","A doctor can perform specific tests to identify the exact cause","Lifestyle factors, diet, genetics, and infections may all play a role"])

SEV_CLASS = {"Low":"sev-low","Medium":"sev-medium","High":"sev-high","Critical":"sev-critical"}

# ── Plotly dark theme (no duplicate axis keys)
def dark_fig_layout(fig, height=300, xaxis_title="", yaxis_title="",
                    yaxis_reversed=False, show_legend=False):
    yax = dict(gridcolor="rgba(0,140,255,0.12)", color="#90b8f8",
               zerolinecolor="rgba(0,140,255,0.2)", title=yaxis_title)
    if yaxis_reversed:
        yax["autorange"] = "reversed"
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(8,16,40,0.7)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c0d8ff", family="DM Sans"),
        xaxis=dict(gridcolor="rgba(0,140,255,0.12)", color="#90b8f8",
                   zerolinecolor="rgba(0,140,255,0.2)", title=xaxis_title),
        yaxis=yax,
        margin=dict(l=10, r=50, t=30, b=30),
        showlegend=show_legend,
        legend=dict(font=dict(color="#a0c4ff"), bgcolor="rgba(0,0,0,0)"),
    )

# helper — build bullet HTML list
def bullets(items, color):
    lis = "".join(f"<li>{i}</li>" for i in items)
    return f"<ul style='color:{color};font-size:.84rem;line-height:1.7;margin:.4rem 0 0 0;padding-left:1.2rem'>{lis}</ul>"

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='sb-title'>🩺 Smart<br>HealthCare AI</div>", unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("", ["🏠 Predict Disease","📊 Data Explorer","ℹ️ About Us"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<div class='warn-box'>⚠️ <b>Disclaimer:</b> For educational use only. Always consult a qualified doctor.</div>",
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT DISEASE
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Predict Disease":

    st.markdown("""<div class='hero-box'>
        <h1>🩺 Smart HealthCare AI</h1>
        <p>Early disease detection powered by Deep Learning · AI for Social Good</p>
    </div>""", unsafe_allow_html=True)

    # Stat cards (FIX 6: removed from sidebar, kept here only)
    c1,c2,c3,c4 = st.columns(4)
    for col,lbl,val in [(c1,"Diseases","41"),(c2,"Symptoms",str(len(FEATURE_COLS))),
                        (c3,"Training Samples",str(len(train_df))),(c4,"Model","Deep NN")]:
        with col:
            st.markdown(f"<div class='stat-card'><div class='lbl'>{lbl}</div><div class='val'>{val}</div></div>",
                        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── PATIENT INFO ──────────────────────────────────────────────────────────
    st.markdown("<div class='glass-card'><h3>👤 Patient Information</h3>", unsafe_allow_html=True)
    pr1,pr2,pr3 = st.columns(3)
    with pr1: patient_name    = st.text_input("Full Name *", placeholder="Enter your full name")
    with pr2: patient_age     = st.number_input("Age *", min_value=1, max_value=120, value=25)
    with pr3: patient_dob     = st.date_input("Date of Birth", value=date(2000,1,1),
                                              min_value=date(1900,1,1), max_value=date.today())
    pr4,pr5,pr6 = st.columns(3)
    with pr4: patient_gender  = st.selectbox("Gender *", ["Select…","Male","Female","Other","Prefer not to say"])
    with pr5: patient_blood   = st.selectbox("Blood Group", ["Select…","A+","A-","B+","B-","AB+","AB-","O+","O-","Unknown"])
    with pr6: patient_contact = st.text_input("Phone / Contact", placeholder="Optional")
    st.markdown("</div>", unsafe_allow_html=True)

    # ── SYMPTOMS (FIX 4: only dropdown, NO checkbox grid) ─────────────────────
    st.markdown("<div class='glass-card'><h3>🔍 Select Symptoms</h3>", unsafe_allow_html=True)
    st.caption("Type in the box below to search and select all your symptoms. You can select multiple.")

    selected_symptoms = st.multiselect(
        "🔽 Search and select your symptoms:",
        options=ALL_SYM_DISPLAY,
        placeholder="Start typing a symptom — e.g. Fever, Headache, Joint Pain…",
    )

    st.markdown("---")
    additional = st.text_area(
        "📝 Any additional symptoms not in the list? Describe in your own words:",
        placeholder="e.g. I feel cold at night, left arm feels numb, very thirsty since 3 days…",
        height=80,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    n_sel = len(selected_symptoms)
    if n_sel:
        st.success(f"✅ {n_sel} symptom(s) selected: {', '.join(selected_symptoms)}")
    else:
        st.info("☝️ Please select at least one symptom, then click Predict.")

    st.markdown("<br>", unsafe_allow_html=True)
    go_btn = st.button("🔮  Analyse & Predict Disease")

    # ── RESULT ────────────────────────────────────────────────────────────────
    if go_btn:
        err = []
        if not patient_name.strip():   err.append("patient name")
        if patient_gender == "Select…": err.append("gender")
        if n_sel == 0:                  err.append("at least one symptom")
        if err:
            st.warning("Please provide: " + ", ".join(err))
        else:
            # Build input vector
            all_selected = dict.fromkeys(FEATURE_COLS, False)
            for d in selected_symptoms:
                ck = SYM_DISP_TO_COL.get(d)
                if ck: all_selected[ck] = True

            vec = np.array([int(all_selected[s]) for s in FEATURE_COLS],
                           dtype=np.float32).reshape(1,-1)

            with st.spinner("🔬 Deep learning model analysing your symptoms…"):
                probs = model.predict(vec, verbose=0)[0]

            top5_idx   = probs.argsort()[::-1][:5]
            top5_names = encoder.inverse_transform(top5_idx)
            top5_probs = probs[top5_idx]

            best      = top5_names[0]
            best_prob = top5_probs[0]
            icon, sev, action, advice_list, causes_list = DISEASE_DB.get(best, DEFAULT_DB)
            sev_cls   = SEV_CLASS.get(sev,"sev-medium")

            st.markdown("---")
            st.markdown("### 🩺 Prediction Result")
            st.markdown(
                f"<p style='color:#8ab4ff;font-size:.9rem'>"
                f"Patient: <b style='color:#fff'>{patient_name}</b> &nbsp;|&nbsp; "
                f"Age: <b style='color:#fff'>{patient_age}</b> &nbsp;|&nbsp; "
                f"Gender: <b style='color:#fff'>{patient_gender}</b> &nbsp;|&nbsp; "
                f"Blood Group: <b style='color:#fff'>{patient_blood}</b></p>",
                unsafe_allow_html=True)

            left, right = st.columns([1,1], gap="large")

            with left:
                st.markdown(f"""<div class='result-card'>
                    <div style='font-size:4rem;margin-bottom:.6rem'>{icon}</div>
                    <div class='disease-name'>{best.strip()}</div>
                    <div class='conf-text'>Confidence: <b>{best_prob*100:.1f}%</b></div>
                    <div class='note-text'>⚠️ AI prediction only — not a substitute for medical diagnosis</div>
                </div>""", unsafe_allow_html=True)

            with right:
                # Severity + Action (FIX 5: professional boxes with bullet points)
                st.markdown(
                    f"<p style='margin:.2rem 0'><span style='color:#a0c4ff;font-weight:600'>🚨 Severity:</span> "
                    f"<span class='{sev_cls}'>{sev}</span></p>",
                    unsafe_allow_html=True)

                st.markdown(f"""<div class='action-box'>
                    <span class='ttl'>👨‍⚕️ Immediate Action Required</span>
                    <p>{action}</p>
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""<div class='cause-box'>
                    <span class='ttl'>🔍 Possible Causes & Risk Factors</span>
                    {bullets(causes_list, '#ffcccc')}
                </div>""", unsafe_allow_html=True)

                st.markdown(f"""<div class='advice-box'>
                    <span class='ttl'>💡 Health Advice & Home Care</span>
                    {bullets(advice_list, '#ffe999')}
                </div>""", unsafe_allow_html=True)

                if additional.strip():
                    st.markdown(f"""<div class='extra-box'>
                        <span class='ttl'>📋 Additional Symptoms You Described</span>
                        <p>{additional.strip()}</p>
                    </div>""", unsafe_allow_html=True)

            # ── CHARTS (FIX 1 & 2: no duplicate xaxis/yaxis keys) ─────────────
            st.markdown("---")
            st.markdown("### 📊 Detailed Analysis")

            r1c1, r1c2 = st.columns(2)

            with r1c1:
                st.markdown("#### 🏆 Top 5 Possible Conditions")
                fig1 = go.Figure(go.Bar(
                    x=top5_probs*100,
                    y=[n.strip() for n in top5_names],
                    orientation="h",
                    marker_color=["#0070ff","#0095ff","#00b8ff","#00d4f0","#00eedd"],
                    text=[f"{p*100:.1f}%" for p in top5_probs],
                    textposition="outside",
                    textfont=dict(color="#ffffff", size=12),
                ))
                dark_fig_layout(fig1, height=270, xaxis_title="Confidence (%)", yaxis_reversed=True)
                st.plotly_chart(fig1, use_container_width=True)

            with r1c2:
                st.markdown("#### 🎯 Prediction Confidence")
                fig2 = go.Figure(go.Pie(
                    values=[best_prob*100, 100-best_prob*100],
                    labels=[best.strip(), "Other conditions"],
                    hole=0.68,
                    marker_colors=["#0088ff","rgba(0,100,200,0.12)"],
                    textinfo="none",
                ))
                fig2.add_annotation(text=f"<b>{best_prob*100:.1f}%</b>",
                    x=0.5, y=0.5, font_size=26, font_color="#ffffff", showarrow=False)
                dark_fig_layout(fig2, height=270, show_legend=True)
                st.plotly_chart(fig2, use_container_width=True)

            # Disease symptom charts
            disease_rows = train_df[train_df["prognosis"] == best]
            if not disease_rows.empty:
                sym_avg  = disease_rows[FEATURE_COLS].mean()
                top_syms = sym_avg[sym_avg > 0.15].sort_values(ascending=False).head(12)

                if not top_syms.empty:
                    r2c1, r2c2 = st.columns(2)

                    with r2c1:
                        st.markdown(f"#### 🔬 Symptom Profile: *{best.strip()}*")
                        fig3 = go.Figure(go.Bar(
                            x=[s.replace("_"," ").title() for s in top_syms.index],
                            y=top_syms.values*100,
                            marker=dict(
                                color=top_syms.values*100,
                                colorscale=[[0,"#003a80"],[0.5,"#0070e0"],[1,"#00ccff"]],
                            ),
                            text=[f"{v*100:.0f}%" for v in top_syms.values],
                            textposition="outside",
                            textfont=dict(color="#ffffff", size=9),
                        ))
                        dark_fig_layout(fig3, height=320, yaxis_title="Frequency (%)")
                        fig3.update_layout(xaxis_tickangle=-42,
                                           xaxis_tickfont=dict(size=8, color="#90b8f8"))
                        st.plotly_chart(fig3, use_container_width=True)

                    with r2c2:
                        st.markdown("#### 📈 Symptom Frequency Trend")
                        sorted_s = top_syms.sort_values(ascending=True)
                        fig4 = go.Figure(go.Scatter(
                            x=sorted_s.values*100,
                            y=[s.replace("_"," ").title() for s in sorted_s.index],
                            mode="lines+markers",
                            line=dict(color="#00ccff", width=2.5, shape="spline"),
                            marker=dict(color="#ffffff", size=9,
                                        line=dict(color="#00ccff", width=2)),
                            fill="tozerox",
                            fillcolor="rgba(0,140,255,0.10)",
                        ))
                        dark_fig_layout(fig4, height=320, xaxis_title="Frequency (%)")
                        st.plotly_chart(fig4, use_container_width=True)

            # Comparison chart
            active_syms = [SYM_DISP_TO_COL[d] for d in selected_symptoms if d in SYM_DISP_TO_COL]
            if active_syms and not disease_rows.empty:
                disease_freq = [float(sym_avg.get(s,0))*100 for s in active_syms]
                st.markdown("#### 🧬 Your Symptoms vs Disease Profile")
                fig5 = go.Figure()
                fig5.add_trace(go.Bar(
                    name="You reported this symptom",
                    x=[s.replace("_"," ").title() for s in active_syms],
                    y=[100]*len(active_syms),
                    marker_color="rgba(0,220,130,0.35)",
                    marker_line=dict(color="#00ff9d", width=1.5),
                ))
                fig5.add_trace(go.Bar(
                    name=f"% of {best.strip()} patients with this symptom",
                    x=[s.replace("_"," ").title() for s in active_syms],
                    y=disease_freq,
                    marker_color="rgba(0,140,255,0.45)",
                    marker_line=dict(color="#00ccff", width=1.5),
                ))
                dark_fig_layout(fig5, height=310, yaxis_title="Frequency (%)", show_legend=True)
                fig5.update_layout(barmode="overlay",
                                   xaxis_tickangle=-38,
                                   xaxis_tickfont=dict(size=8, color="#90b8f8"))
                st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Data Explorer":

    st.markdown("""<div class='hero-box'>
        <h1>📊 Dataset Explorer</h1>
        <p>Interactive visualisation of the disease prediction training dataset</p>
    </div>""", unsafe_allow_html=True)

    # Disease distribution
    st.markdown("#### 🦠 Disease Distribution")
    dist = train_df["prognosis"].value_counts().reset_index()
    dist.columns = ["Disease","Count"]
    fig = go.Figure(go.Bar(
        x=dist["Count"], y=dist["Disease"], orientation="h",
        marker=dict(color=dist["Count"],
                    colorscale=[[0,"#003080"],[0.5,"#0075e0"],[1,"#00d4ff"]]),
        text=dist["Count"], textposition="outside",
        textfont=dict(color="#ffffff", size=8),
    ))
    dark_fig_layout(fig, height=950, xaxis_title="Sample Count", yaxis_reversed=True)
    st.plotly_chart(fig, use_container_width=True)

    da1, da2 = st.columns(2)

    with da1:
        st.markdown("#### 💊 Top 30 Symptoms — Bar Chart")
        sym_f = train_df[FEATURE_COLS].sum().sort_values(ascending=False).head(30)
        fig2 = go.Figure(go.Bar(
            x=[s.replace("_"," ").title() for s in sym_f.index],
            y=sym_f.values,
            marker=dict(color=sym_f.values,
                        colorscale=[[0,"#003a80"],[0.5,"#0090e0"],[1,"#00eedd"]]),
            text=sym_f.values, textposition="outside",
            textfont=dict(color="#ffffff", size=8),
        ))
        dark_fig_layout(fig2, height=420, yaxis_title="Count")
        fig2.update_layout(xaxis_tickangle=-45, xaxis_tickfont=dict(size=7, color="#90b8f8"))
        st.plotly_chart(fig2, use_container_width=True)

    with da2:
        st.markdown("#### 📈 Symptom Frequency — Line Chart")
        fig3 = go.Figure(go.Scatter(
            x=[s.replace("_"," ").title() for s in sym_f.index],
            y=sym_f.values,
            mode="lines+markers",
            line=dict(color="#00ccff", width=2.5, shape="spline"),
            marker=dict(color="#ffffff", size=7, line=dict(color="#00ccff", width=2)),
            fill="tozeroy", fillcolor="rgba(0,140,255,0.1)",
        ))
        dark_fig_layout(fig3, height=420, yaxis_title="Count")
        fig3.update_layout(xaxis_tickangle=-45, xaxis_tickfont=dict(size=7, color="#90b8f8"))
        st.plotly_chart(fig3, use_container_width=True)

    # Heatmap
    st.markdown("#### 🌡️ Disease × Symptom Heatmap (top 25 symptoms)")
    top25  = train_df[FEATURE_COLS].sum().sort_values(ascending=False).head(25).index.tolist()
    matrix = train_df.groupby("prognosis")[top25].mean()
    fig4   = go.Figure(go.Heatmap(
        z=matrix.values,
        x=[s.replace("_"," ").title() for s in top25],
        y=[d.strip() for d in matrix.index],
        colorscale=[[0,"rgba(0,15,50,0.9)"],[0.5,"#0070e0"],[1,"#00ffe0"]],
        text=np.round(matrix.values,2),
        texttemplate="%{text}", textfont=dict(size=7, color="white"),
    ))
    dark_fig_layout(fig4, height=920)
    fig4.update_layout(xaxis_tickangle=-42,
                       xaxis_tickfont=dict(size=7, color="#90b8f8"),
                       yaxis_tickfont=dict(size=8, color="#90b8f8"))
    st.plotly_chart(fig4, use_container_width=True)

    # Pie
    st.markdown("#### 🍕 Disease Proportion")
    fig5 = go.Figure(go.Pie(
        labels=[d.strip() for d in dist["Disease"]], values=dist["Count"],
        hole=0.38, textinfo="label+percent",
        textfont=dict(size=8, color="white"),
        marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=1)),
    ))
    dark_fig_layout(fig5, height=620)
    fig5.update_layout(showlegend=False)
    st.plotly_chart(fig5, use_container_width=True)

    with st.expander("📄 Raw Training Data (first 100 rows)"):
        st.dataframe(train_df.head(100), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT US  (FIX 3: renamed from "About" to "About Us")
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About Us":

    st.markdown("""<div class='hero-box'>
        <h1>ℹ️ About Us</h1>
        <p>Smart HealthCare AI · AI for Social Good · Deep Learning · Early Disease Detection</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    ### 🏥 Who We Are

    **Smart HealthCare AI** is an AI-powered disease detection system built as part of the
    *AI for Social Good* initiative. Our goal is to make early medical screening accessible
    to everyone — regardless of location or access to healthcare facilities.

    We are a team of developers, data scientists, and healthcare enthusiasts committed to
    using deep learning technology to save lives through early detection.

    ---

    ### 🎯 Our Mission

    Early detection of diseases like **diabetes, heart disease, malaria, and tuberculosis**
    is challenging without automated screening tools. Delays in diagnosis lead to poor
    health outcomes — especially in under-resourced communities.

    Our AI system analyses patient-reported symptoms and predicts the most likely disease
    instantly, along with severity assessment, probable causes, and personalised health advice.

    ---

    ### 🧠 Model Architecture

    | Layer | Type | Units |
    |-------|------|-------|
    | Input | Binary symptoms | 132 |
    | Hidden 1 | Dense + BatchNorm + Dropout(0.4) | 512 |
    | Hidden 2 | Dense + BatchNorm + Dropout(0.3) | 256 |
    | Hidden 3 | Dense + BatchNorm + Dropout(0.2) | 128 |
    | Hidden 4 | Dense ReLU | 64 |
    | Output | Dense Softmax | 41 diseases |

    **Optimizer:** Adam + ReduceLROnPlateau &nbsp;|&nbsp; **Loss:** Categorical Cross-Entropy &nbsp;|&nbsp; **Early Stopping:** Patience = 10

    ---

    ### 📦 Technology Stack

    | Tool | Purpose |
    |------|---------|
    | Python 3.10+ | Core programming language |
    | TensorFlow / Keras | Deep Learning model training |
    | Scikit-learn | Preprocessing & evaluation metrics |
    | Streamlit | Web application framework |
    | Plotly | Interactive data visualisation |
    | Pandas / NumPy | Data processing and analysis |

    ---

    ### 📂 Dataset Details
    - **Source:** Kaggle — Disease Prediction Using Machine Learning
    - **Training samples:** 4,920 &nbsp;|&nbsp; **Test samples:** 42
    - **Features:** 132 binary symptoms &nbsp;|&nbsp; **Disease classes:** 41
    - **Balance:** 120 samples per disease — perfectly balanced dataset

    ---

    ### ⚠️ Important Disclaimer
    This application is built strictly for **educational and research purposes**.
    It is **not** a replacement for professional medical advice, clinical diagnosis,
    or medical treatment. Always consult a qualified and licensed healthcare provider
    for any medical concerns.
    """)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#2a4a7f;font-size:.78rem'>🩺 Smart HealthCare AI · AI for Social Good · Built with Streamlit & TensorFlow</p>",
            unsafe_allow_html=True)