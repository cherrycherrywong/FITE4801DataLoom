from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import io
import requests
import json
from dotenv import load_dotenv

# ==========================================
# 🔧 CONFIGURATION (env-aware)
# ==========================================
# Load .env if present (for local development):
load_dotenv()

# Prefer environment variables. Fallback to embedded defaults if not set.
BASE44_API_KEY = os.getenv('BASE44_API_KEY', "8232d62b6cf24e7eb74c7f765abb6e10")
BASE44_ENDPOINT_URL = os.getenv('BASE44_ENDPOINT_URL', "https://app.base44.com/api/apps/69160892745be891ce4c021a/entities/TravelDataProfile")

# Keep the hardcoded data as a safety backup 
BACKUP_CSV_DATA = """full_name,email,trip_name,category,description,file_urls,anonymized,share_with_enterprises,earn_rewards,completeness_score,quality_score,token_price,tokens_earned,ai_insights,status,credit_score,credit_rating,bank_action,avg_spend_per_trip,premium_vs_budget,ancillary_spending,cross_border_frequency,trip_frequency_annual,business_vs_leisure_ratio,destination_consistency,booking_lead_time_days,insurance_purchase_rate,payment_timing,loyalty_tier,activity_preferences,digital_touchpoint
Tom Chan,tom.chan@fyp.edu,London Biz,Business,Premium Q1 business trip,ipfs://Qm1,true,true,true,98,9.5,250,150,High-trust traveler,active,812,Excellent,Approve,7500,Premium,1200,12,15,90/10,High,45,100,Early,Platinum,"Networking, Dining",Concierge
Tom Chan,tom.chan@fyp.edu,Tokyo Tech,Business,Tech summit attendance,ipfs://Qm2,true,true,true,95,9.2,200,120,Stable professional,active,805,Excellent,Approve,6800,Premium,800,12,15,90/10,High,30,95,Early,Platinum,"Tech Expo",Desktop Web
Priya Sharma,priya@example.com,SG Tech,Business,Regional developer meet,ipfs://Qm3,true,true,true,90,8.8,150,90,Reliable borrower,active,745,Good,Approve,4200,Mid-Range,500,8,10,70/30,High,32,90,On-Time,Gold,"Workshops",Mobile App
James Wilson,j.wilson@demo.com,Bali Surf,Adventure,Weekend surfing trip,ipfs://Qm4,false,true,true,45,4.5,50,10,Spontaneous patterns,active,520,Poor,Decline,850,Budget,100,1,2,0/100,Low,2,0,Late,None,"Surfing",Mobile App
James Wilson,j.wilson@demo.com,Phuket Fun,Leisure,Island beach party,ipfs://Qm5,false,true,true,40,4.2,40,5,High risk profile,active,510,Poor,Decline,700,Budget,50,1,2,0/100,Low,1,5,Late,None,"Party",Mobile App
Elena Rossi,elena@milan.it,Milan Fashion,Business,Fashion week attendance,ipfs://Qm6,true,true,true,92,9.1,220,180,High spending power,active,790,Excellent,Approve,6100,Premium,2000,8,12,80/20,High,28,100,Early,Platinum,"Retail, Runway",Concierge
Li Wei,li.wei@hk.cn,Summer HK,Family,Annual family visit,ipfs://Qm7,true,true,true,85,8.4,120,80,Family-oriented,active,680,Good,Review,3500,Mid-Range,800,2,4,20/80,Medium,55,85,On-Time,Silver,"Dining, Culture",Desktop Web
Sarah Jenkins,sarah@alps.uk,Alps Hike,Adventure,Mountain trekking,ipfs://Qm8,true,true,true,88,8.5,130,95,Risk-taker pattern,active,710,Good,Approve,2200,Mid-Range,400,3,6,10/90,Medium,40,95,On-Time,Gold,"Climbing",Mobile App
Kevin Varma,kevin@vegas.com,Vegas Solo,Leisure,Solo weekend getaway,ipfs://Qm9,false,true,true,50,5.0,80,20,Low stability,active,450,Poor,Decline,1500,Budget,500,0,3,5/95,Low,1,10,Late,Bronze,"Gambling",Mobile App
Anita Desai,anita@finance.in,NYC Summit,Business,Global finance meet,ipfs://Qm10,true,true,true,97,9.6,300,200,Top-tier profile,active,835,Excellent,Approve,9200,Premium,3000,15,20,95/5,High,50,100,Early,Platinum,"Banking",Concierge
Mark Zubak,mark@berlin.de,Berlin Club,Leisure,Nightlife weekend,ipfs://Qm11,false,true,true,55,5.5,70,15,Unpredictable,active,590,Fair,Review,1100,Budget,400,0,5,0/100,Low,7,20,Late,None,"Nightlife",Mobile App
Chloe Dubois,chloe@paris.fr,Paris Food,Leisure,Culinary tour,ipfs://Qm12,true,true,true,82,8.0,140,110,Discretionary spender,active,725,Good,Approve,4800,Mid-Range,1500,5,8,30/70,Medium,30,75,On-Time,Gold,"Dining",Desktop Web
S. Al-Fayed,sami@dubai.ae,Dubai Lux,Leisure,Desert luxury stay,ipfs://Qm13,true,true,true,99,9.8,500,300,Elite spender,active,845,Excellent,Approve,12000,Premium,5000,20,25,50/50,High,14,100,Early,Platinum,"Luxury",Concierge
Jack Thorne,jack@yukon.ca,Yukon Camp,Adventure,Wilderness camping,ipfs://Qm14,false,true,true,65,6.5,90,40,Robust history,active,640,Fair,Review,1800,Budget,300,0,2,10/90,Low,48,60,On-Time,Silver,"Camping",Desktop Web
M. Takahashi,masa@kyoto.jp,Kyoto Fall,Leisure,Temple autumn visit,ipfs://Qm15,true,true,true,94,9.4,180,140,Consistent planner,active,760,Excellent,Approve,5500,Mid-Range,1200,4,6,10/90,High,90,100,Early,Gold,"Sightseeing",Desktop Web
Rosa Mendez,rosa@cancun.mx,Cancun Sun,Family,Beach resort stay,ipfs://Qm16,true,true,true,75,7.5,110,60,Standard family,active,610,Fair,Review,2900,Mid-Range,600,1,3,5/95,Medium,45,40,On-Time,Bronze,"Swimming",Mobile App
David Cohen,david@telaviv.io,Israel Tech,Business,Startup pitch week,ipfs://Qm17,true,true,true,91,9.0,200,150,Growth pattern,active,805,Excellent,Approve,6700,Premium,1000,10,12,85/15,High,20,100,Early,Gold,"Innovation",Desktop Web
Nina Patel,nina@mumbai.in,Wedding,Family,Relative wedding,ipfs://Qm18,true,true,true,80,8.2,160,100,Stable long-lead,active,695,Good,Review,4100,Mid-Range,1200,2,5,10/90,High,120,90,Early,Silver,"Ceremony",Desktop Web
Alex Karev,alex@seattle.edu,Med Conf,Business,Hospital conference,ipfs://Qm19,true,true,true,89,8.9,140,110,Professional risk,active,730,Good,Approve,3800,Mid-Range,400,4,8,80/20,High,15,80,On-Time,Gold,"Healthcare",Desktop Web
Omar Hassan,omar@cairo.eg,Giza Tour,Leisure,History tour,ipfs://Qm20,false,true,true,50,5.2,60,10,Low verification,active,540,Poor,Decline,950,Budget,100,0,2,5/95,Low,5,15,Late,None,"Museums",Mobile App
Hans Muller,hans@munich.de,Trade Expo,Business,Industrial fair,ipfs://Qm21,true,true,true,96,9.5,240,160,Industrial stable,active,785,Excellent,Approve,5200,Premium,1100,12,14,95/5,High,40,100,Early,Platinum,"Trade",Concierge
Tom Chan,tom.chan@fyp.edu,Seoul Solo,Leisure,Personal exploration,ipfs://Qm22,true,true,true,93,9.3,100,80,Versatile traveler,active,802,Excellent,Approve,2100,Mid-Range,500,12,15,90/10,High,20,100,Early,Platinum,"Exploration",Mobile App
Sarah Jenkins,sarah@alps.uk,Andes Trip,Adventure,South America peaks,ipfs://Qm23,true,true,true,85,8.2,150,120,Explorer profile,active,705,Good,Approve,2800,Mid-Range,600,4,6,10/90,Medium,45,90,On-Time,Gold,"Climbing",Mobile App
R. Sterling,raheem@london.uk,Euro Derby,Leisure,Major football match,ipfs://Qm24,true,true,true,88,8.8,300,250,High value enthusiast,active,740,Good,Approve,5000,Premium,2500,10,18,20/80,Medium,14,50,Early,Silver,"Sports",Concierge
Kim Soo-Jin,sj.kim@seoul.kr,Fashion Expo,Business,Fashion logistics meet,ipfs://Qm25,true,true,true,95,9.5,210,190,Strategic business,active,795,Excellent,Approve,6400,Premium,1800,9,12,85/15,High,22,100,Early,Gold,"Fashion",Desktop Web
Carlos Ruiz,carlos@madrid.es,Family Visit,Family,Grandparent visit,ipfs://Qm26,true,true,true,70,7.2,80,40,Steady family,active,655,Good,Review,2100,Mid-Range,300,2,5,15/85,Medium,35,70,On-Time,Silver,"Visiting",Desktop Web
Fatima Zahra,fatima@mecca.sa,Hajj 2025,Family,Religious pilgrimage,ipfs://Qm27,true,true,true,100,10,400,350,Extreme planning,active,715,Good,Approve,3900,Mid-Range,1000,1,3,0/100,High,180,100,Early,Gold,"Religious",Travel Agent
James Wilson,j.wilson@demo.com,Full Moon,Leisure,Koh Phangan party,ipfs://Qm28,false,true,true,40,4.0,30,5,Inconsistent habits,active,510,Poor,Decline,700,Budget,100,1,2,0/100,Low,1,5,Late,None,"Party",Mobile App
Ben Affleck,ben@hollywood.com,Film Shoot,Business,Production logistics,ipfs://Qm29,true,true,true,98,9.7,500,450,High worth entity,active,810,Excellent,Approve,15000,Premium,5000,10,12,100/0,High,10,100,Early,Platinum,"Film",Concierge
Zoe Kravitz,zoe@nyc.com,Fashion Tour,Business,Brand campaign,ipfs://Qm30,true,true,true,94,9.2,450,400,Creative elite,active,780,Excellent,Approve,11000,Premium,4000,12,15,90/10,High,5,100,Early,Gold,"Art",Concierge
Mark Zubak,mark@berlin.de,Prague Beer,Leisure,Bachelor party,ipfs://Qm31,false,true,true,52,5.0,60,10,Poor risk control,active,575,Fair,Review,900,Budget,200,0,5,0/100,Low,4,10,Late,None,"Party",Mobile App
M. Jordan,goat@chicago.com,Golf Lux,Leisure,Private club tour,ipfs://Qm32,true,true,true,100,10,1000,900,Wealth standard,active,830,Excellent,Approve,20000,Premium,8000,5,10,10/90,Medium,2,100,Early,Platinum,"Golf",Concierge
H. Potter,harry@hogwarts.uk,Train Tour,Family,Steam train tour,ipfs://Qm33,true,true,true,80,8.0,90,70,Magical reliability,active,620,Fair,Review,1200,Budget,200,0,3,10/90,Low,10,90,On-Time,Silver,"Trains",Desktop Web
Diana Prince,diana@themyscira.com,Hist Athens,Leisure,Archaeology dig,ipfs://Qm34,true,true,true,98,9.8,200,160,Immortal planning,active,845,Excellent,Approve,4500,Mid-Range,1000,20,30,20/80,High,50,100,Early,Gold,"History",Desktop Web
Bruce Wayne,bruce@wayne.com,Gotham Charity,Business,Philanthropy summit,ipfs://Qm35,true,true,true,100,10,5000,4500,The standard,active,848,Excellent,Approve,30000,Premium,15000,30,40,95/5,High,1,100,Early,Platinum,"Gala",Concierge
Clark Kent,clark@dailyplanet.com,Kansas Home,Family,Parents farm visit,ipfs://Qm36,true,true,true,99,9.9,60,40,Humble trust,active,760,Excellent,Approve,600,Budget,100,2,4,40/60,High,10,100,Early,Bronze,"Farm",Desktop Web
Peter Parker,peter@dailybugle.com,Venice Sch,Leisure,High school trip,ipfs://Qm37,false,true,true,45,4.5,40,10,Struggling student,active,510,Poor,Decline,300,Budget,50,1,2,0/100,Low,1,0,On-Time,None,"Science",Mobile App
Tony Stark,tony@stark.com,Vegas Tech,Business,AI defense demo,ipfs://Qm38,true,true,true,100,10,6000,5500,Future wealth,active,849,Excellent,Approve,45000,Premium,20000,40,50,100/0,High,1,100,Early,Platinum,"Iron",Concierge
Li Wei,li.wei@hk.cn,Shanghai Biz,Business,Supplier audit,ipfs://Qm39,true,true,true,90,8.8,160,130,Reliable auditor,active,715,Good,Approve,4400,Mid-Range,900,4,6,90/10,Medium,120,90,Early,Silver,"Audit",Desktop Web
Steve Rogers,steve@avengers.com,DC Memorial,Leisure,Memorial visit,ipfs://Qm40,true,true,true,100,10,120,100,Unmatched integrity,active,800,Excellent,Approve,1500,Mid-Range,300,5,8,10/90,High,30,100,Early,Silver,"History",Desktop Web
Wanda Maxim,wanda@westview.com,Berlin Visit,Family,Sister visit,ipfs://Qm41,true,true,true,60,6.0,80,40,Unstable context,active,580,Fair,Review,1000,Budget,200,2,4,20/80,Low,5,30,On-Time,Bronze,"Magic",Mobile App
B. Banner,bruce@gamma.edu,Brazil Med,Business,Research lab tour,ipfs://Qm42,true,true,true,92,9.2,180,140,Doctor trust,active,750,Good,Approve,2500,Mid-Range,400,6,10,100/0,Medium,20,100,Early,Gold,"Science",Desktop Web
Thor Odinson,thor@asgard.no,Norway Myth,Adventure,Northern lights,ipfs://Qm43,true,true,true,98,9.8,500,450,Divine wealth,active,820,Excellent,Approve,10000,Premium,4000,15,25,10/90,High,1,100,Early,Platinum,"Myth",Concierge
Loki Laufeys,loki@asgard.no,Norway Hike,Adventure,Fjord exploring,ipfs://Qm44,false,true,true,65,6.5,400,300,Mischief risk,active,480,Poor,Decline,4000,Premium,1500,15,25,10/90,High,1,0,Late,Bronze,"Trickery",Concierge
S. Strange,stephen@sanctum.com,Nepal Study,Business,Medical research,ipfs://Qm45,true,true,true,95,9.5,250,220,Surgical precision,active,815,Excellent,Approve,5500,Premium,1800,12,18,100/0,High,15,100,Early,Gold,"Medicine",Desktop Web
T'Challa,panther@wakanda.com,Wakanda Exp,Business,Trade negotiations,ipfs://Qm46,true,true,true,100,10,8000,7500,Royal standard,active,847,Excellent,Approve,40000,Premium,18000,20,30,100/0,High,5,100,Early,Platinum,"Trade",Concierge
A. Curry,arthur@atlantis.com,Hawaii Surf,Leisure,Pro surf meet,ipfs://Qm47,true,true,true,90,9.0,300,280,Oceanic wealth,active,705,Good,Approve,6000,Mid-Range,2000,5,10,10/90,High,40,90,On-Time,Gold,"Surfing",Mobile App
Barry Allen,barry@central.com,Star City,Leisure,Visit friends,ipfs://Qm48,true,true,true,75,7.5,60,50,Fast but safe,active,690,Good,Review,800,Budget,100,4,8,20/80,Low,1,100,Early,Silver,"Visiting",Desktop Web
Hal Jordan,hal@coast.com,Coast City,Business,Pilot conference,ipfs://Qm49,true,true,true,90,9.0,280,240,Fearless credit,active,740,Good,Approve,7000,Premium,2000,15,20,95/5,High,10,80,On-Time,Gold,"Flight",Desktop Web
V. Stone,victor@cyber.com,Cyber Summit,Business,Robotics demo,ipfs://Qm50,true,true,true,98,9.8,350,320,Integrated intelligence,active,825,Excellent,Approve,9000,Premium,4000,18,22,100/0,High,25,100,On-Time,Platinum,"Tech",Concierge"""


app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# --- 1. LOAD MODEL ---
try:
    model = joblib.load(os.path.join(os.path.dirname(__file__), 'credit_brain.pkl'))
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ CRITICAL: Could not load model. Error: {e}")
    print("⚠️ Please run 'python train_model.py' first!")
    model = None

# --- 2. DATA LOADING LOGIC (Hybrid: Base44 -> Fallback CSV) ---
def get_data_frame():
    """
    Tries to fetch from Base44. If it fails, falls back to CSV.
    """
    print("🔄 Attempting to fetch real-time data from Base44...")
    
    # ⚠️ NOTICE: Using 'api_key' in header as per your JS snippet
    headers = {
        'api_key': BASE44_API_KEY, 
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(BASE44_ENDPOINT_URL, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data_json = response.json()
                # Assuming the data is in 'items' key as per Base44 structure
            if isinstance(data_json, dict) and 'items' in data_json:
                records = data_json['items']
            elif isinstance(data_json, list):
                records = data_json
            else:
                records = [data_json] # Handle single object return

            df = pd.DataFrame(records)
            print(f"✅ Success! Loaded {len(df)} records from Base44.")
            return df
        else:
            print(f"⚠️ API Error (Status {response.status_code}). Response: {response.text}")
    except Exception as e:
        print(f"⚠️ Connection Failed: {e}.")

    # Fallback to CSV if API fails
    print("📂 Loading backup CSV data...")
    return pd.read_csv(io.StringIO(BACKUP_CSV_DATA))

# Load data initially
df = get_data_frame()

# --- 3. DATA MODELS ---
class IntegratedResponse(BaseModel):
    calculated_score: int
    risk_rating: str
    probability: float
    trips: int
    total_spend: float
    avg_completeness: float
    average_spend_per_trip: float
    premium_vs_budget: str
    ancillary_spending: str
    currency_habits: str
    trip_frequency: int
    travel_purpose_ratio: str
    destination_consistency: str
    geographic_consistency: str
    booking_lead_times: str
    insurance_purchase_rate: str
    payment_timing: str
    loyalty_tiers: str
    activity_preferences: str
    obstacles_challenges: str
    digital_touchpoints: str
    recommendation: str

# --- 4. ROUTES ---
@app.get("/")
def read_root():
    return {"message": "Alternative Credit Scoring API (Hybrid Mode)"}

@app.get("/search")
def search_page(request: Request):
    return templates.TemplateResponse("search.html", {"request": request})

@app.get("/insights")
def insights_page(request: Request, name: str = None):
    return templates.TemplateResponse("insights.html", {"request": request, "name": name})

def get_user_features(name: str):
    # Refresh data on every search to get "Real-Time" updates
    global df
    df = get_data_frame()

    print(f"🔍 Searching for: {name}")

    try:
        # Case-insensitive search for Name OR Email
        # Ensure 'full_name' and 'email' exist in df columns before filtering
        if 'full_name' not in df.columns or 'email' not in df.columns:
            print("❌ Error: 'full_name' or 'email' columns missing from data source.")
            return None

        matching_users = df[
            (df['full_name'].astype(str).str.contains(name, case=False, na=False)) | 
            (df['email'].astype(str).str.contains(name, case=False, na=False))
        ]
        
        if matching_users.empty:
            return None

        # Get the email of the first match to aggregate all their trips
        user_email = matching_users['email'].iloc[0]
        user_trips = df[df['email'] == user_email]
        
        # Calculate Aggregates
        trips_count = len(user_trips)
        
        # Handle 'tokens_earned' (Use 0 if column missing or null)
        spend_col = 'tokens_earned' if 'tokens_earned' in user_trips.columns else 'spend'
        if spend_col in user_trips.columns:
            avg_spend = pd.to_numeric(user_trips[spend_col], errors='coerce').sum()
        else:
            avg_spend = 0
            
        # Handle 'completeness_score'
        comp_col = 'completeness_score'
        if comp_col in user_trips.columns:
            completeness_score = pd.to_numeric(user_trips[comp_col], errors='coerce').mean()
        else:
            completeness_score = 0
        
        # Handle 'category' for business trips
        business_trips = 0
        if 'category' in user_trips.columns:
            business_trips = (user_trips['category'] == 'Business').sum()
            
        return trips_count, avg_spend, completeness_score, business_trips

    except Exception as e:
        print(f"❌ Error processing data columns: {e}")
        return None

@app.get("/get-credit-score-by-name", response_model=IntegratedResponse)
def predict_by_name(name: str):
    features = get_user_features(name)
    if features is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    trips, spend, completeness, business_trips = features
    
    # 🧠 AI Prediction
    # Handle NaN values safely before prediction
    spend = 0 if pd.isna(spend) else spend
    completeness = 0 if pd.isna(completeness) else completeness
    trips = 0 if pd.isna(trips) else trips

    features_df = pd.DataFrame([[trips, spend, completeness]], columns=['trips_count', 'avg_spend', 'completeness_score'])
    probability = model.predict_proba(features_df)[0][1]
    score = int(round(300 + (probability * 550)))
    
    # Rating Logic
    if score >= 800: risk_rating = "Excellent"
    elif score >= 740: risk_rating = "Very Good"
    elif score >= 670: risk_rating = "Good"
    elif score >= 580: risk_rating = "Fair"
    else: risk_rating = "Poor"
    
    average_spend_per_trip = spend / trips if trips > 0 else 0
    
    return {
        "calculated_score": score,
        "risk_rating": risk_rating,
        "probability": probability,
        "trips": trips,
        "total_spend": spend,
        "avg_completeness": completeness,
        "average_spend_per_trip": average_spend_per_trip,
        "premium_vs_budget": "Premium" if spend > 1000 else "Budget",
        "ancillary_spending": "High" if completeness > 80 else "Low",
        "currency_habits": "International" if business_trips > 0 else "Domestic",
        "trip_frequency": trips,
        "travel_purpose_ratio": f"{business_trips}/{trips}",
        "destination_consistency": "Consistent" if trips > 5 else "Variable",
        "geographic_consistency": "Stable" if completeness > 70 else "Unstable",
        "booking_lead_times": "Advance" if completeness > 90 else "Last-minute",
        "insurance_purchase_rate": "High" if completeness > 85 else "Low",
        "payment_timing": "Prompt" if completeness > 75 else "Delayed",
        "loyalty_tiers": "High" if trips > 10 else "Standard",
        "activity_preferences": "Business" if business_trips > trips / 2 else "Leisure",
        "obstacles_challenges": "Minimal" if completeness > 80 else "Frequent",
        "digital_touchpoints": "Mobile App",
        "recommendation": "Approve Instantly" if score > 720 else "Request Manual Review"
    }

#pip install fastapi uvicorn pandas scikit-learn joblib jinja2 requests python-multipart
#python train_model.py
#uvicorn main:app --reload
#http://127.0.0.1:8000/search