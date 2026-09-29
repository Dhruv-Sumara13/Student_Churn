"""Shared deterministic features for training and serving."""
import numpy as np

HIGH_CHURN_DISTS = ['SABARKANTHA','KUTCH','BANASKANTHA','MEHSANA','BOTAD']
LOCAL_DISTS      = ['GANDHINAGAR','AHMEDABAD']
ALL_DISTRICTS    = sorted([
    "AHMEDABAD","AMRELI","ARVALLI","BANASKANTHA","BHAVNAGAR","BOTAD",
    "GANDHINAGAR","GIR SOMNATH","JAMNAGAR","JUNAGADH","KHEDA","KUTCH",
    "MEHSANA","MORBI","PATAN","RAJKOT","SABARKANTHA","SURENDRANAGAR",
    "ANAND","AURANGABAD","DAHOD","DEVBHUMI DWARKA","DUNGARPUR","GONDIYA",
    "HALVAD","HIMMATNAGAR","MAHARASHTRA","MAHISAGAR","NAVSARI","PORBANDAR",
    "SIROHI","SURAT",
])

SEM_HIST_RATE = {1: 1.000, 2: 0.809, 3: 0.032, 4: 0.000, 5: 0.000, 6: 0.000}

COLLEGE_SEMESTERS = {
    "BPCCS": [1, 2, 3, 4, 5, 6],
    "SVICS-G": [1, 2, 3]
}

SEM_LABELS = {
    1: "1 -- First Semester",
    2: "2 -- Second Semester",
    3: "3 -- Third Semester",
    4: "4 -- Fourth Semester",
    5: "5 -- Fifth Semester",
    6: "6 -- Sixth Semester",
}

def build_admission_features(exam_pct, gender, college, year_gap,
                              spec, board, caste, religion, district):
    gen  = 1 if gender=="Female" else 0
    col  = 0 if "BPCCS" in college else 1
    fees = 18000 if "BPCCS" in college else 27000
    obc  = 1 if caste=="OBC" else 0
    sct  = 1 if caste=="SCST" else 0
    sbc  = 1 if caste=="SEBC" else 0
    opn  = 1 if caste=="OPEN" else 0
    mus  = 1 if religion=="Muslim" else 0
    hin  = 1 if religion=="Hindu" else 0
    sci  = 1 if spec=="SCIENCE" else 0
    art  = 1 if spec=="ARTS" else 0
    com  = 1 if spec=="COMMERCE" else 0
    cbse = 1 if "CBSE" in board.upper() else 0
    gseb = 1 if any(x in board.upper() for x in ["GSEB","GHSEB","G.H.S.E.B","G.S.E.B"]) else 0
    dh   = 1 if district in HIGH_CHURN_DISTS else 0
    dl   = 1 if district in LOCAL_DISTS else 0
    pdev = exam_pct - 61.5
    pdz  = 1 if 50<=exam_pct<65 else 0
    pvl  = 1 if exam_pct<45 else 0
    rs   = min(5, obc+sci+art+dh+mus+pdz+cbse)
    return {
        'exam_pct': exam_pct, 'pct_sq': (exam_pct/100)**2,
        'pct_dev': pdev, 'pct_dev_sq': pdev**2,
        'pct_danger': pdz, 'pct_very_low': pvl,
        'gender': gen, 'college': col, 'fees': fees, 'year_gap': year_gap,
        'cast_obc': obc, 'cast_scst': sct, 'cast_sebc': sbc, 'cast_open': opn,
        'rel_muslim': mus, 'rel_hindu': hin,
        'spec_science': sci, 'spec_arts': art, 'spec_commerce': com,
        'board_cbse': cbse, 'board_gseb': gseb,
        'dist_high': dh, 'dist_local': dl,
        'pct_x_obc': exam_pct*obc, 'pct_x_science': exam_pct*sci,
        'pct_x_dist': exam_pct*dh, 'pct_x_college': exam_pct*col,
        'bpccs_obc': (1-col)*obc, 'female_svics': gen*col,
        'obc_science': obc*sci,
        'risk_score': rs, 'risk_x_pct': rs*exam_pct,
    }

def rebuild_features(raw_df):
    df = raw_df.copy()
    df['exam_pct']   = df['Last Exam Percentage']
    df['gender']     = (df['Gender']=='Female').astype(int)
    df['college']    = (df['Institute']=='SVICS-G').astype(int)
    df['fees']       = df['Total Fees']
    df['cast_obc']   = (df['Admission Cast Category']=='OBC').astype(int)
    df['cast_scst']  = (df['Admission Cast Category']=='SCST').astype(int)
    df['cast_sebc']  = (df['Admission Cast Category']=='SEBC').astype(int)
    df['cast_open']  = (df['Admission Cast Category']=='OPEN').astype(int)
    df['rel_muslim'] = (df['Religion']=='Muslim').astype(int)
    df['rel_hindu']  = (df['Religion']=='Hindu').astype(int)
    df['spec_science']  = (df['Specialisation']=='SCIENCE').astype(int)
    df['spec_arts']     = (df['Specialisation']=='ARTS').astype(int)
    df['spec_commerce'] = (df['Specialisation']=='COMMERCE').astype(int)
    passing_year = df['Last Exam Passing'].astype(str).str.extract(r'^(\d{4})', expand=False).astype(float)
    df['year_gap'] = (df['Admission Year'] - passing_year).clip(1, 5)
    bs = df['Last Exam Board/Uni.'].str.upper().fillna('')
    df['board_cbse'] = bs.str.contains('CBSE',na=False).astype(int)
    df['board_gseb'] = bs.str.contains(r'GSEB|GHSEB|G\.H\.S\.E\.B|G\.S\.E\.B',na=False).astype(int)
    df['dist_high']  = df['Permanent District'].isin(HIGH_CHURN_DISTS).astype(int)
    df['dist_local'] = df['Permanent District'].isin(LOCAL_DISTS).astype(int)
    df['pct_dev']    = df['exam_pct'] - 61.5
    df['pct_dev_sq'] = df['pct_dev']**2
    df['pct_danger'] = ((df['exam_pct']>=50)&(df['exam_pct']<65)).astype(int)
    df['pct_very_low'] = (df['exam_pct']<45).astype(int)
    df['pct_sq']       = (df['exam_pct']/100)**2
    df['pct_x_obc']    = df['exam_pct']*df['cast_obc']
    df['pct_x_science']= df['exam_pct']*df['spec_science']
    df['pct_x_dist']   = df['exam_pct']*df['dist_high']
    df['pct_x_college']= df['exam_pct']*df['college']
    df['bpccs_obc']    = (1-df['college'])*df['cast_obc']
    df['female_svics'] = df['gender']*df['college']
    df['obc_science']  = df['cast_obc']*df['spec_science']
    df['risk_score']   = (df['cast_obc']+df['spec_science']+df['spec_arts']+
                          df['dist_high']+df['rel_muslim']+
                          df['pct_danger']+df['board_cbse']).clip(0,5)
    df['risk_x_pct']   = df['risk_score']*df['exam_pct']
    return df


FEATURES = list(build_admission_features(60, 'Male', 'BPCCS', 1, 'COMMERCE', 'GSEB', 'OPEN', 'Hindu', 'AHMEDABAD'))

def semester_probability(base, semesters, rates):
    values = np.asarray(semesters, dtype=int)
    weights = np.array([{1: .20, 2: .15, 3: .05}.get(int(s), 0.) for s in values])
    signal = np.array([rates.get(int(s), 0.) for s in values])
    return np.clip((1 - weights) * np.asarray(base) + weights * signal, 0, 1)
