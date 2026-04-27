import cv2
import numpy as np
import os
import sys
import warnings
import mediapipe as mp
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
import datetime
import math

sys.path.append(r"E:\python\FIT_ME_UP\MediaPipe\code")
from predict import predict_posture

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ── 경로 ──────────────────────────────────────────────────
YOLO_MODEL    = r"E:\python\FIT_ME_UP\YOLO\fit_me_up\combined_gpu\weights\best.pt"
IMAGE_PATH    = r"E:\python\FIT_ME_UP\test (3).jpg"
MODEL_PATH_MP = r"E:\python\FIT_ME_UP\MediaPipe\models\pose_landmarker.task"
LOGO_PATH     = r"E:\python\FIT_ME_UP\logo.png"
FONT_BOLD     = r"E:\python\FIT_ME_UP\Fonts\malgunbd.ttf"
FONT_REG      = r"E:\python\FIT_ME_UP\Fonts\malgun.ttf"

# ── 출력 해상도 ────────────────────────────────────────────
TW, TH = 2560, 1440
IW     = 1200   # 이미지 영역
PW     = TW-IW  # 패널 영역 1360px

# ── 폰트 ──────────────────────────────────────────────────
FP  = FONT_BOLD
FPR = FONT_REG
F10 = ImageFont.truetype(FPR, 13)
F12 = ImageFont.truetype(FPR, 15)
F14 = ImageFont.truetype(FPR, 17)
F16 = ImageFont.truetype(FP,  19)
F18 = ImageFont.truetype(FP,  21)
F22 = ImageFont.truetype(FP,  26)
F26 = ImageFont.truetype(FP,  30)
F32 = ImageFont.truetype(FP,  36)
F40 = ImageFont.truetype(FP,  44)

# ── 컬러 ──────────────────────────────────────────────────
C = {
    'base':    (7,   12,  26),
    'panel':   (11,  19,  42),
    'card':    (17,  29,  58),
    'card2':   (22,  38,  72),
    'header':  (9,   16,  36),
    'tw':      (230, 238, 255),
    'tg':      (130, 150, 190),
    'td':      (65,  80,  120),
    'brand':   (0,   215, 190),
    'brand2':  (0,   155, 255),
    'good':    (42,  215, 130),
    'bad':     (255, 65,  65),
    'ideal':   (255, 215, 0),
    'range':   (0,   160, 255),
    'blue':    (35,  105, 235),
    'skel':    (0,   215, 190),
    'chair':   (255, 150, 40),
    'desk':    (40,  205, 255),
    'monitor': (185, 70,  255),
}

CLASS_NAMES = {0:'chair', 1:'desk', 2:'monitor'}

FEEDBACK = {
    'CVA': {
        'no':'01','label':'목굴곡각','eng':'CVA','range':'0° ~ 20°','cat':'posture',
        'good':'머리·경추 수직 정렬 유지\n경추 부담 최소화 상태',
        'bad': '전방두부자세(FHP) 의심\n모니터를 눈높이로 올리세요\n1시간마다 목 스트레칭 시행'
    },
    'TIA': {
        'no':'02','label':'몸통굴곡각','eng':'TIA','range':'0° ~ 10°','cat':'posture',
        'good':'척추 수직 정렬 양호\n요추 압박 최소화 상태',
        'bad': '과도한 몸통 전굴 감지\n등받이에 허리 완전 밀착\n의자 깊숙이 앉으세요'
    },
    '팔꿈치': {
        'no':'03','label':'팔꿈치 각도','eng':'Elbow','range':'90° ~ 120°','cat':'posture',
        'good':'상지 관절 부하 최적 범위\nVDT 고시 제6조 2항 충족',
        'bad': '팔꿈치 각도 기준 이탈\n의자 높이 조정 필요\n팔꿈치·책상면 수평 유지'
    },
    '무릎': {
        'no':'04','label':'무릎 각도','eng':'Knee','range':'85° ~ 100°','cat':'posture',
        'good':'하지 혈액순환 원활\nVDT 고시 제6조 6항 충족',
        'bad': '무릎 각도 기준 이탈\n의자 높이 조절 필요\n발받침대 사용 권장'
    },
    '손목': {
        'no':'05','label':'손목 각도','eng':'Wrist','range':'±15° 이내','cat':'posture',
        'good':'손목 중립 자세 유지\nCTS 위험 최소화 상태',
        'bad': '손목 과굴곡 감지\n손목 받침대 설치 필요\n키보드 앞 15cm 확보'
    },
    '시선각': {
        'no':'06','label':'모니터 시선각','eng':'Gaze','range':'하방 10°~15°','cat':'env',
        'good':'시선각 VDT 기준 충족\n경추 부담 최소화',
        'bad': '시선각 기준 이탈\n모니터 상단 눈높이 맞춤\n화면 거리 40cm 이상'
    },
    '책상높이': {
        'no':'07','label':'작업대 높이','eng':'Desk','range':'팔꿈치 수평 ±10%','cat':'env',
        'good':'작업대·팔꿈치 정렬 양호\n상지 부담 최소화',
        'bad': '작업대 높이 불일치\n책상 65cm 전후 조정\n의자 높이로 보정 가능'
    },
    '등받이': {
        'no':'08','label':'의자 등받이','eng':'Chair','range':'골반너비 20% 이내','cat':'env',
        'good':'등받이 지지 충분\n요추 안정성 확보',
        'bad': '등받이 지지 부족\n의자 깊숙이 착석\n허리 완전 밀착 필요'
    },
}

yolo = YOLO(YOLO_MODEL)

try:
    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    USE_LEGACY = True
except AttributeError:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    USE_LEGACY = False
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH_MP)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1, min_pose_detection_confidence=0.5
    )
    pose = vision.PoseLandmarker.create_from_options(options)

def calc_angle(A,B,C_):
    v1 = np.array(A)-np.array(B)
    v2 = np.array(C_)-np.array(B)
    ct = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
    return np.degrees(np.arccos(np.clip(ct,-1,1)))

def calc_vert(A,B):
    v=(B[0]-A[0],B[1]-A[1])
    return np.degrees(np.arctan2(abs(v[0]),abs(v[1])))

def calc_gaze(eye,mc):
    return np.degrees(np.arctan2(mc[1]-eye[1],mc[0]-eye[0]))

def judge(v,mn,mx):
    return 'Good' if mn<=v<=mx else 'Bad'

# ── 측면 스켈레톤 (우측만) ────────────────────────────────
def draw_skeleton(img, lm, h, w):
    conns = [
        (8, 12),   # 귀 → 어깨
        (12, 14),  # 어깨 → 팔꿈치
        (14, 16),  # 팔꿈치 → 손목
        (16, 20),  # 손목 → 손가락
        (12, 24),  # 어깨 → 골반
        (24, 26),  # 골반 → 무릎
        (26, 28),  # 무릎 → 발목
    ]
    sc  = C['skel']
    col = (sc[2], sc[1], sc[0])

    for a, b in conns:
        ax,ay = int(lm[a].x*w), int(lm[a].y*h)
        bx,by = int(lm[b].x*w), int(lm[b].y*h)
        cv2.line(img,(ax,ay),(bx,by),col,2,cv2.LINE_AA)

    for idx in [8,12,14,16,20,24,26,28]:
        x,y = int(lm[idx].x*w), int(lm[idx].y*h)
        cv2.circle(img,(x,y),5,col,-1,cv2.LINE_AA)
        cv2.circle(img,(x,y),5,(255,255,255),1,cv2.LINE_AA)

# ── 정상범위 호(arc) + 이상적 포인트 표시 ─────────────────
def draw_ideal_arc(img, center, p1, p2, current_angle,
                   good_min, good_max, is_good, radius=45):
    cx,cy = int(center[0]),int(center[1])
    x1,y1 = int(p1[0]),int(p1[1])
    x2,y2 = int(p2[0]),int(p2[1])

    v1 = np.array([x1-cx,y1-cy],dtype=float)
    v2 = np.array([x2-cx,y2-cy],dtype=float)

    def ang(v):
        return math.degrees(math.atan2(v[1],v[0]))

    a1 = ang(v1)
    a2 = ang(v2)

    range_col = C['range']
    ov        = img.copy()
    cv2.ellipse(ov,(cx,cy),(radius,radius),0,
                int(min(a1,a2)),int(max(a1,a2)),
                (range_col[2],range_col[1],range_col[0]),4)
    cv2.addWeighted(ov,0.5,img,0.5,0,img)

    ideal_ang_rad = math.radians((a1+a2)/2)
    ix = int(cx + radius*math.cos(ideal_ang_rad))
    iy = int(cy + radius*math.sin(ideal_ang_rad))
    ic = C['ideal']
    cv2.circle(img,(ix,iy),8,(ic[2],ic[1],ic[0]),-1,cv2.LINE_AA)
    cv2.circle(img,(ix,iy),8,(255,255,255),1,cv2.LINE_AA)

# ── YOLO bbox ──────────────────────────────────────────────
def draw_yolo_boxes(img, bbox):
    pil = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    d   = ImageDraw.Draw(pil)
    cm  = {'chair':C['chair'],'desk':C['desk'],'monitor':C['monitor']}
    lm_ = {'chair':'의자','desk':'책상','monitor':'모니터'}
    for name,b in bbox.items():
        if b is None: continue
        rc = cm[name]
        pc = (rc[0],rc[1],rc[2])
        x1,y1,x2,y2 = b['x_min'],b['y_min'],b['x_max'],b['y_max']
        ov = img.copy()
        cv2.rectangle(ov,(x1,y1),(x2,y2),(rc[2],rc[1],rc[0]),2)
        cv2.addWeighted(ov,0.7,img,0.3,0,img)
        pil = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        d   = ImageDraw.Draw(pil)
        sz  = 16
        for cx_,cy_,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            d.line([(cx_,cy_),(cx_+dx*sz,cy_)],fill=pc,width=3)
            d.line([(cx_,cy_),(cx_,cy_+dy*sz)],fill=pc,width=3)
        tw = len(lm_[name])*12+16
        d.rounded_rectangle([(x1,y1-28),(x1+tw,y1-2)],radius=4,fill=pc)
        d.text((x1+8,y1-25),lm_[name],font=F14,fill=(255,255,255))
    img[:] = cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)

# ── 관절 포인트 ────────────────────────────────────────────
def draw_point(img, pt, is_good, label):
    x,y = int(pt[0]),int(pt[1])
    sc_ = C['good'] if is_good else C['bad']
    col = (sc_[2],sc_[1],sc_[0])
    ov  = img.copy()
    cv2.circle(ov,(x,y),18,col,-1)
    cv2.addWeighted(ov,0.45,img,0.55,0,img)
    cv2.circle(img,(x,y),18,col,2,cv2.LINE_AA)
    cv2.circle(img,(x,y),5,(255,255,255),-1,cv2.LINE_AA)
    pil = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    d   = ImageDraw.Draw(pil)
    tw  = len(label)*9+16
    d.rounded_rectangle([(x+22,y-16),(x+22+tw,y+14)],radius=5,fill=(7,12,26))
    d.text((x+27,y-14),label,font=F14,fill=sc_)
    img[:] = cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)

def wrap(text,mc=24):
    lines=text.split('\n'); res=[]
    for l in lines:
        while len(l)>mc: res.append(l[:mc]); l=l[mc:]
        res.append(l)
    return res

# ── 카드 (2열) ─────────────────────────────────────────────
def draw_card(draw, x, y, W, key, value, is_good):
    fb    = FEEDBACK[key]
    sc_   = C['good'] if is_good else C['bad']
    stat  = 'GOOD' if is_good else 'BAD'
    msg   = fb['good'] if is_good else fb['bad']
    lines = wrap(msg,24)
    cc    = C['card']
    H     = 24+26+24+len(lines)*18+16

    draw.rounded_rectangle([(x,y),(x+W,y+H)],radius=8,fill=(cc[0],cc[1],cc[2]))
    draw.rounded_rectangle([(x,y),(x+W,y+5)],radius=4,fill=(sc_[0],sc_[1],sc_[2]))

    td = C['td']
    draw.text((x+10,y+10),fb['no'],font=F12,fill=(td[0],td[1],td[2]))
    tw_ = C['tw']
    draw.text((x+36,y+9),fb['label'],font=F16,fill=(tw_[0],tw_[1],tw_[2]))
    tg_ = C['tg']
    draw.text((x+36+len(fb['label'])*12,y+12),f"  {fb['eng']}",
              font=F10,fill=(tg_[0],tg_[1],tg_[2]))

    bw=70
    draw.rounded_rectangle([(x+W-bw-6,y+9),(x+W-6,y+27)],
                           radius=3,fill=(sc_[0],sc_[1],sc_[2]))
    draw.text((x+W-bw+6,y+11),stat,font=F12,fill=(255,255,255))

    bl = C['blue']
    draw.rounded_rectangle([(x+8,y+32),(x+W-8,y+54)],
                           radius=4,fill=(bl[0]//5,bl[1]//5,bl[2]//5+15))
    draw.text((x+14,y+35),
              f"정상범위  {fb['range']}   |   측정값  {value}",
              font=F10,fill=(bl[0],bl[1],bl[2]))

    my=y+58
    for l in lines:
        draw.text((x+10,my),l,font=F14,fill=(tg_[0],tg_[1],tg_[2]))
        my+=18
    return H

# ── 섹션 헤더 ──────────────────────────────────────────────
def draw_sec(draw,x,y,W,title,sub,col):
    cc = C['card2']
    draw.rounded_rectangle([(x,y),(x+W,y+52)],radius=6,fill=(cc[0],cc[1],cc[2]))
    draw.rounded_rectangle([(x,y),(x+5,y+52)],radius=3,fill=(col[0],col[1],col[2]))
    tw_ = C['tw']
    draw.text((x+18,y+7),title,font=F18,fill=(col[0],col[1],col[2]))
    tg_ = C['tg']
    draw.text((x+18,y+32),sub,font=F12,fill=(tg_[0],tg_[1],tg_[2]))
    return y+60

# ── 패널 빌드 ──────────────────────────────────────────────
def build_panel(img, pd_, ed_, is_good, conf):
    INNER = PW-36
    GAP   = 12
    CW    = (INNER-GAP)//2
    px    = IW+18

    img_r = cv2.resize(img,(IW,TH),interpolation=cv2.INTER_LANCZOS4)
    panel = np.zeros((TH,PW,3),dtype=np.uint8)
    bg    = C['base']
    panel[:] = (bg[2],bg[1],bg[0])

    canvas = np.hstack([img_r,panel])
    pil    = Image.fromarray(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
    draw   = ImageDraw.Draw(pil)

    py = 0

    # ── 헤더 ───────────────────────────────────────────────
    hh = C['header']
    draw.rectangle([(IW,0),(TW,110)],fill=(hh[0],hh[1],hh[2]))

    logo_ok = False
    try:
        logo = Image.open(LOGO_PATH).convert("L")
        logo_color = Image.new("RGB", logo.size, (0,215,190))
        logo_alpha = logo.point(lambda p: int(p * 0.85))
        logo_rgba  = Image.merge("RGBA",[logo_color.split()[0],
                                         logo_color.split()[1],
                                         logo_color.split()[2],
                                         logo_alpha])
        lh = 80
        lw = int(logo.width*(lh/logo.height))
        logo_rgba = logo_rgba.resize((lw,lh),Image.LANCZOS)
        bg_patch  = Image.new("RGBA",(lw,lh),(hh[0],hh[1],hh[2],255))
        bg_patch.alpha_composite(logo_rgba)
        pil.paste(bg_patch.convert("RGB"),(px,15))
        lx = px+lw+20
        logo_ok = True
    except:
        lx = px

    ac = C['brand']
    if logo_ok:
        tw_ = C['tw']
        draw.text((lx,16),"Fit Me Up",font=F26,fill=(tw_[0],tw_[1],tw_[2]))
        tg_ = C['tg']
        draw.text((lx,52),"beyond the hospital, in to your life",
                  font=F12,fill=(tg_[0],tg_[1],tg_[2]))
        draw.text((lx,70),"VDT 근로자 자세·환경 AI 분석 시스템",
                  font=F12,fill=(tg_[0],tg_[1],tg_[2]))
    else:
        draw.text((px,18),"Fit Me Up",font=F32,fill=(ac[0],ac[1],ac[2]))
        tg_ = C['tg']
        draw.text((px,58),"beyond the hospital, in to your life",
                  font=F14,fill=(tg_[0],tg_[1],tg_[2]))

    now = datetime.datetime.now().strftime("%Y.%m.%d  %H:%M")
    td_ = C['td']
    tg_ = C['tg']
    for i,(k,v) in enumerate([("분석 일시",now),
                               ("근거 법령","VDT 고시 제2020-17호"),
                               ("평가 기준","RULA · 산업안전보건법")]):
        draw.text((TW-260,14+i*30),k,font=F10,fill=(td_[0],td_[1],td_[2]))
        draw.text((TW-260,26+i*30),v,font=F12,fill=(tg_[0],tg_[1],tg_[2]))

    draw.rectangle([(IW,107),(TW,111)],fill=(ac[0],ac[1],ac[2]))
    py = 120

    # ── 범례 ───────────────────────────────────────────────
    legend_items = [
        (C['skel'],  "측면 골격선"),
        (C['good'],  "정상 판정"),
        (C['bad'],   "교정 필요"),
        (C['ideal'], "이상적 포인트"),
        (C['range'], "정상범위 호"),
    ]
    lx_l, ly_l = 20, TH-160
    draw.rounded_rectangle([(lx_l-8,ly_l-8),(lx_l+200,ly_l+len(legend_items)*24+8)],
                           radius=6,fill=(7,12,26))
    for i,(col,label) in enumerate(legend_items):
        cy_ = ly_l+i*24+8
        draw.ellipse([(lx_l,cy_),(lx_l+14,cy_+14)],fill=(col[0],col[1],col[2]))
        draw.text((lx_l+20,cy_-1),label,font=F12,fill=(130,150,190))

    # ── 종합 판정 카드 ─────────────────────────────────────
    all_d    = {**pd_,**ed_}
    good_cnt = sum(1 for _,(_,g) in all_d.items() if g)
    total    = len(all_d)
    score    = int(good_cnt/total*100)
    sc_      = C['good'] if is_good else C['bad']
    cc_      = C['card2']

    draw.rounded_rectangle([(px,py),(px+INNER,py+105)],
                           radius=10,fill=(cc_[0],cc_[1],cc_[2]))
    draw.rounded_rectangle([(px,py),(px+7,py+105)],
                           radius=5,fill=(sc_[0],sc_[1],sc_[2]))

    cr  = 42
    ccx = px+INNER-cr-20
    ccy = py+52
    draw.ellipse([(ccx-cr,ccy-cr),(ccx+cr,ccy+cr)],
                 fill=(sc_[0]//5,sc_[1]//5,sc_[2]//5))
    draw.ellipse([(ccx-cr,ccy-cr),(ccx+cr,ccy+cr)],
                 outline=(sc_[0],sc_[1],sc_[2]),width=3)
    draw.text((ccx-24,ccy-22),f"{score}",font=F32,fill=(sc_[0],sc_[1],sc_[2]))
    draw.text((ccx-8,ccy+14),"점",font=F12,fill=(sc_[0],sc_[1],sc_[2]))

    icon = "✅" if is_good else "❌"
    stat = "GOOD — 자세 양호" if is_good else "BAD — 교정 필요"
    tw_  = C['tw']
    draw.text((px+22,py+10),f"{icon}  {stat}",font=F26,fill=(sc_[0],sc_[1],sc_[2]))
    tg_  = C['tg']
    detail = "자세 지표 양호 · 작업 환경 분석 진행" if is_good \
             else "신체 자세 교정이 우선적으로 필요합니다"
    draw.text((px+22,py+50),detail,font=F16,fill=(tg_[0],tg_[1],tg_[2]))
    draw.text((px+22,py+72),
              f"AI 신뢰도  {conf*100:.1f}%   ·   통과  {good_cnt}/{total}개",
              font=F14,fill=(tg_[0],tg_[1],tg_[2]))

    bx,by = px+22,py+90
    bw_   = max(10, INNER - (ccx - px) - cr*2 - 40)
    c2_   = C['card']
    draw.rounded_rectangle([(bx,by),(bx+bw_,by+9)],radius=4,fill=(c2_[0],c2_[1],c2_[2]))
    fw = int(bw_*score/100)
    if fw>0:
        draw.rounded_rectangle([(bx,by),(bx+fw,by+9)],radius=4,fill=(sc_[0],sc_[1],sc_[2]))
    py += 116

    # ── 카드 렌더링 (2열) ──────────────────────────────────
    def render_cards(draw, data, py):
        keys = list(data.keys())
        i    = 0
        while i < len(keys):
            lk     = keys[i]
            lv,lg  = data[lk]
            lh     = draw_card(draw,px,py,CW,lk,lv,lg)
            if i+1 < len(keys):
                rk    = keys[i+1]
                rv,rg = data[rk]
                rh    = draw_card(draw,px+CW+GAP,py,CW,rk,rv,rg)
                row_h = max(lh,rh)
            else:
                row_h = lh
            py += row_h+10
            i  += 2
            if py > TH-60: break
        return py

    if not is_good:
        bad_ = C['bad']
        py   = draw_sec(draw,px,py,INNER,
                        "📐  신체 자세 지표 분석",
                        "RULA Group B 기준 · 목·몸통·팔꿈치·무릎·손목 평가",
                        bad_)
        py = render_cards(draw,pd_,py)

    else:
        gd_ = C['good']
        py  = draw_sec(draw,px,py,INNER,
                       "✅  신체 자세 지표 분석",
                       "RULA Group B 기준 충족 · 세부 측정값 확인",
                       gd_)
        py = render_cards(draw,pd_,py)

        py += 6
        br_ = C['brand']
        py  = draw_sec(draw,px,py,INNER,
                       "🖥  작업 환경 지표 분석",
                       "VDT 고시 제6조 기준 · 모니터·책상·의자 환경 평가",
                       br_)
        py = render_cards(draw,ed_,py)

    # ── 이미지 하단 정보 ───────────────────────────────────
    td_ = C['td']
    draw.text((20,TH-50),"MediaPipe Pose  ·  YOLOv8  ·  MobileNetV2",
              font=F10,fill=(td_[0],td_[1],td_[2]))
    draw.text((20,TH-32),"Fit Me Up AI Ergonomic Analysis Engine v1.0",
              font=F10,fill=(td_[0],td_[1],td_[2]))

    # ── 푸터 ───────────────────────────────────────────────
    hh_ = C['header']
    draw.rectangle([(IW,TH-38),(TW,TH)],fill=(hh_[0],hh_[1],hh_[2]))
    draw.rectangle([(IW,TH-39),(TW,TH-38)],fill=(ac[0],ac[1],ac[2]))
    draw.text((px,TH-26),
              "Fit Me Up  ·  beyond the hospital, in to your life  "
              "·  VDT 고시 제2020-17호 / RULA / 산업안전보건법",
              font=F10,fill=(td_[0],td_[1],td_[2]))

    return cv2.cvtColor(np.array(pil),cv2.COLOR_RGB2BGR)

# ── 메인 ───────────────────────────────────────────────────
def analyze(image_path):
    img     = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    h,w     = img.shape[:2]

    if USE_LEGACY:
        res = pose.process(img_rgb)
        if not res.pose_landmarks:
            print("❌ 사람 인식 실패! 측면 전신 사진으로 재촬영하세요.")
            return
        lm = res.pose_landmarks.landmark
    else:
        mpi = mp.Image(image_format=mp.ImageFormat.SRGB,data=img_rgb)
        res = pose.detect(mpi)
        if not res.pose_landmarks:
            print("❌ 사람 인식 실패!")
            return
        lm = res.pose_landmarks[0]

    print("✅ 인체 랜드마크 검출 완료")

    posture = predict_posture(image_path)
    if 'error' in posture:
        print(f"❌ 모델 오류: {posture['error']}")
        return

    is_good = posture['label']=='good'
    conf    = posture['confidence']
    print(f"📊 {'GOOD ✅' if is_good else 'BAD ❌'} (신뢰도 {conf*100:.1f}%)")

    def gxy(idx):
        return (lm[idx].x*w, lm[idx].y*h)

    ear_r   = gxy(8)
    sh_l    = gxy(11)
    sh_r    = gxy(12)
    elbow_r = gxy(14)
    wrist_r = gxy(16)
    finger  = gxy(20)
    hp_l    = gxy(23)
    hp_r    = gxy(24)
    knee_r  = gxy(26)
    ankle_r = gxy(28)
    eye_c   = ((lm[1].x+lm[4].x)/2*w,(lm[1].y+lm[4].y)/2*h)
    sh_mid  = ((sh_l[0]+sh_r[0])/2,(sh_l[1]+sh_r[1])/2)
    hp_mid  = ((hp_l[0]+hp_r[0])/2,(hp_l[1]+hp_r[1])/2)

    yr   = yolo(img)[0]
    bbox = {'chair':None,'desk':None,'monitor':None}
    for box in yr.boxes:
        cls_  = int(box.cls[0])
        conf_ = float(box.conf[0])
        name  = CLASS_NAMES.get(cls_)
        if name and conf_>=0.25:
            x1,y1,x2,y2 = map(int,box.xyxy[0])
            bbox[name]={'x_min':x1,'y_min':y1,'x_max':x2,'y_max':y2}

    cva = round(calc_vert(ear_r,sh_r),1)
    tia = round(calc_vert(sh_mid,hp_mid),1)
    el  = round(calc_angle(sh_r,elbow_r,wrist_r),1)
    kn  = round(calc_angle(hp_r,knee_r,ankle_r),1)
    wr  = round(calc_angle(elbow_r,wrist_r,finger),1)

    gaze=None
    if bbox['monitor']:
        mx=(bbox['monitor']['x_min']+bbox['monitor']['x_max'])/2
        my=(bbox['monitor']['y_min']+bbox['monitor']['y_max'])/2
        gaze=round(calc_gaze(eye_c,(mx,my)),1)

    hd,dc=None,None
    if bbox['desk']:
        dty=bbox['desk']['y_min']
        ref=abs(hp_mid[1]-sh_mid[1])
        hd=round(abs(dty-elbow_r[1])/(ref+1e-8),3)
        dc=((bbox['desk']['x_min']+bbox['desk']['x_max'])//2,dty)

    gr=None
    if bbox['chair']:
        cbx=bbox['chair']['x_max']
        hw=abs(hp_l[0]-hp_r[0]) or abs(sh_l[0]-sh_r[0])
        gr=round(abs(hp_r[0]-cbx)/(hw+1e-8),3)

    pd_ = {
        'CVA':    (f"{cva}°",  judge(cva,0,20)=='Good'),
        'TIA':    (f"{tia}°",  judge(tia,0,10)=='Good'),
        '팔꿈치': (f"{el}°",   judge(el,90,120)=='Good'),
        '무릎':   (f"{kn}°",   judge(kn,85,100)=='Good'),
        '손목':   (f"{wr}°",   judge(wr,165,180)=='Good'),
    }
    ed_ = {
        '시선각':  (f"{gaze}°" if gaze else 'N/A',
                    judge(gaze,10,15)=='Good' if gaze else False),
        '책상높이':(f"{hd}" if hd else 'N/A',
                    judge(hd,0,0.10)=='Good' if hd else False),
        '등받이':  (f"{gr}" if gr else 'N/A',
                    judge(gr,0,0.20)=='Good' if gr else False),
    }

    print("\n"+"─"*50)
    for k,(v,g) in {**pd_,**ed_}.items():
        fb=FEEDBACK[k]
        print(f"  {fb['no']}. {fb['label']:12s} {v:10s} {'✅' if g else '❌'}")
    print("─"*50)

    # ── 시각화 ─────────────────────────────────────────────
    draw_skeleton(img,lm,h,w)

    draw_ideal_arc(img,elbow_r,sh_r,wrist_r,el,90,120,pd_['팔꿈치'][1],50)
    draw_ideal_arc(img,knee_r, hp_r,ankle_r,kn,85,100,pd_['무릎'][1],  50)
    draw_ideal_arc(img,wrist_r,elbow_r,finger,wr,165,180,pd_['손목'][1],40)

    draw_yolo_boxes(img,bbox)

    draw_point(img,ear_r,   pd_['CVA'][1],    '①목')
    draw_point(img,sh_mid,  pd_['TIA'][1],    '②허리')
    draw_point(img,elbow_r, pd_['팔꿈치'][1], '③팔꿈치')
    draw_point(img,knee_r,  pd_['무릎'][1],   '④무릎')
    draw_point(img,wrist_r, pd_['손목'][1],   '⑤손목')

    if is_good:
        if gaze is not None:
            draw_point(img,eye_c, ed_['시선각'][1],  '⑥시선')
        if dc is not None:
            draw_point(img,dc,    ed_['책상높이'][1],'⑦책상')
        if gr is not None:
            draw_point(img,hp_r,  ed_['등받이'][1],  '⑧등받이')

    final = build_panel(img,pd_,ed_,is_good,conf)
    out   = r"E:\python\FIT_ME_UP\result.jpg"
    cv2.imwrite(out,final,[cv2.IMWRITE_JPEG_QUALITY,98])
    print(f"\n✅ 저장 완료 ({TW}×{TH}): {out}")

if __name__ == '__main__':
    analyze(IMAGE_PATH)
