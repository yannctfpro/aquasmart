from __future__ import annotations

from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

from src.core.recommendation_engine import (
    CLUSTER_NAMES,
    CROP_DISPLAY_NAMES,
    CROPS,
    GROWTH_STAGE_ENCODING,
    ModelNotReadyError,
    SUPPORTED_CROPS,
    build_farmer_recommendation,
    get_growth_stage,
    get_model_info,
    get_model_status,
    refresh_model_store,
)

st.set_page_config(page_title="AquaSmart", page_icon="💧", layout="wide", initial_sidebar_state="expanded")

RUNTIME_OPTIONS = ("Local engine", "FastAPI backend")
DEFAULT_API_URL = "http://127.0.0.1:8000"

# Crop emojis covering all 15 crops
CROP_EMOJIS = {
    "winter_wheat": "🌾", "durum_wheat": "🌾", "winter_barley": "🌿", "oats": "🌾", "triticale": "🌾",
    "corn": "🌽", "sunflower": "🌻", "sorghum": "🌾", "soybean": "🫘",
    "rapeseed": "🌼", "winter_pea": "🫛", "faba_bean": "🫛",
    "potato": "🥔", "sugar_beet": "🥕", "field_vegetables": "🥬",
}

# Rough season display — derived from sowing month + expected harvest
CROP_SEASONS = {
    "winter_wheat": "Oct → Jul", "durum_wheat": "Oct → Jul", "winter_barley": "Oct → Jun",
    "oats": "Oct → Jun", "triticale": "Oct → Jul",
    "corn": "Apr → Sep", "sunflower": "Apr → Sep", "sorghum": "May → Oct", "soybean": "May → Oct",
    "rapeseed": "Sep → Jul", "winter_pea": "Nov → Jun", "faba_bean": "Nov → Jun",
    "potato": "Apr → Sep", "sugar_beet": "Mar → Oct", "field_vegetables": "Apr → Oct",
}

# Sort crops alphabetically by display name for the dropdown
SORTED_CROPS = sorted(SUPPORTED_CROPS, key=lambda c: CROP_DISPLAY_NAMES.get(c, c))


def svg_to_uri(svg: str) -> str:
    compact = " ".join(line.strip() for line in svg.strip().splitlines())
    return f"data:image/svg+xml;utf8,{quote(compact)}"


def farm_scene_uri() -> str:
    return svg_to_uri(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 620">
          <defs>
            <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stop-color="#84d2ff"/><stop offset="52%" stop-color="#f9e19e"/><stop offset="100%" stop-color="#f4b36b"/>
            </linearGradient>
            <linearGradient id="green" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stop-color="#9bc965"/><stop offset="100%" stop-color="#5f8a40"/>
            </linearGradient>
          </defs>
          <rect width="900" height="620" fill="url(#sky)"/>
          <circle cx="702" cy="118" r="72" fill="#ffd861"/>
          <g fill="#fff" opacity=".9">
            <ellipse cx="170" cy="118" rx="58" ry="25"/><ellipse cx="214" cy="113" rx="46" ry="22"/><ellipse cx="248" cy="120" rx="32" ry="16"/>
            <ellipse cx="516" cy="145" rx="46" ry="20"/><ellipse cx="555" cy="142" rx="34" ry="16"/>
          </g>
          <path d="M0 272 C132 226 281 267 410 240 C562 210 682 190 900 240 L900 620 L0 620 Z" fill="#96bc69"/>
          <path d="M0 334 C170 286 301 347 438 318 C628 278 737 257 900 305 L900 620 L0 620 Z" fill="#789c51"/>
          <path d="M0 394 C175 356 322 398 456 371 C618 341 752 338 900 374 L900 620 L0 620 Z" fill="#5c8240"/>
          <path d="M0 420 L220 360 L438 430 L268 620 L0 620 Z" fill="#d3ad4f"/>
          <path d="M192 370 L456 318 L694 420 L420 620 L142 620 Z" fill="url(#green)"/>
          <g opacity=".26" stroke="#f9efc8" stroke-width="4" fill="none">
            <path d="M42 620 C95 550 132 500 186 386"/><path d="M92 620 C154 544 194 482 250 375"/><path d="M146 620 C215 536 250 476 314 368"/>
            <path d="M214 620 C290 520 330 465 386 355"/><path d="M356 620 C414 548 468 482 540 334"/><path d="M430 620 C488 554 548 475 620 336"/>
            <path d="M507 620 C568 556 625 478 694 350"/><path d="M618 620 C670 545 728 476 805 378"/>
          </g>
          <g transform="translate(545 248)">
            <rect x="0" y="50" width="106" height="82" rx="6" fill="#d35f44"/><polygon points="-8,54 54,8 116,54" fill="#7d3f2b"/>
            <rect x="40" y="82" width="28" height="50" rx="4" fill="#77462c"/><rect x="12" y="70" width="16" height="16" rx="2" fill="#f7dfb0"/><rect x="78" y="70" width="16" height="16" rx="2" fill="#f7dfb0"/>
            <rect x="124" y="28" width="28" height="104" rx="8" fill="#ead9c0"/><path d="M124 28 L138 2 L152 28" fill="#ccb492"/>
          </g>
          <g transform="translate(150 422)">
            <path d="M0 88 C18 40 35 12 52 0 C46 34 52 58 74 88 Z" fill="#2e7d60"/><path d="M50 88 C62 34 78 14 95 2 C95 30 102 56 124 88 Z" fill="#4f9d68"/>
            <path d="M98 88 C112 48 130 26 150 14 C147 38 156 60 180 88 Z" fill="#71b276"/>
          </g>
          <g transform="translate(680 420)">
            <path d="M0 86 Q24 22 64 0 Q52 43 86 86 Z" fill="#f1d457"/><path d="M46 86 Q72 20 112 4 Q98 45 134 86 Z" fill="#e4b63a"/>
            <circle cx="64" cy="66" r="18" fill="#6f4d2a"/><rect x="60" y="82" width="8" height="46" fill="#3f7c51"/>
          </g>
        </svg>
        """
    )


def leaf_badge_uri() -> str:
    return svg_to_uri(
        """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320">
          <defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#fff4d3"/><stop offset="100%" stop-color="#ffe4c4"/></linearGradient></defs>
          <rect width="320" height="320" rx="48" fill="url(#bg)"/><circle cx="240" cy="76" r="32" fill="#ffd666"/>
          <ellipse cx="160" cy="278" rx="102" ry="18" fill="#e6c59d"/><rect x="96" y="164" width="128" height="24" rx="12" fill="#c87b49"/>
          <path d="M106 178 H214 L194 252 H126 Z" fill="#96572d"/><rect x="154" y="110" width="12" height="58" rx="6" fill="#3e7d51"/>
          <path d="M160 142 C126 136 104 116 98 82 C134 84 156 104 160 142 Z" fill="#6aaf68"/><path d="M160 134 C194 130 224 106 232 72 C194 74 168 98 160 134 Z" fill="#3b915f"/>
          <path d="M150 166 C118 168 90 188 78 216 C114 214 144 196 150 166 Z" fill="#84c272"/><path d="M170 164 C208 166 236 188 250 216 C210 216 180 198 170 164 Z" fill="#56a36d"/>
        </svg>
        """
    )


def inject_styles() -> None:
    farm_scene = farm_scene_uri()
    leaf_badge = leaf_badge_uri()
    st.markdown(
        f"""
        <style>
        :root {{
            --ink:#19362d; --teal:#0f7c6b; --water:#3b86a8; --leaf:#6a9444; --sun:#e5b54b; --soil:#8b5e34; --alert:#bf5b2c;
            --line:rgba(25,54,45,.12); --shadow:0 22px 50px rgba(31,58,40,.12);
        }}
        .stApp {{
            background: radial-gradient(circle at top left, rgba(255,214,92,.18), transparent 24%), radial-gradient(circle at top right, rgba(59,134,168,.16), transparent 28%), linear-gradient(180deg, #eee1bf 0%, #f6efde 24%, #fbf7f0 100%);
            color:var(--ink);
        }}
        .block-container {{ padding-top:1.3rem; padding-bottom:2.5rem; }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(254,249,237,.98), rgba(246,236,216,.98));
            border-right:1px solid rgba(139,94,52,.1);
        }}
        [data-testid="stForm"] {{
            background: linear-gradient(180deg, rgba(255,255,255,.94), rgba(248,241,228,.98));
            border:1px solid rgba(139,94,52,.12); border-radius:28px; padding:1rem 1.1rem .75rem; box-shadow:var(--shadow);
        }}
        [data-testid="stMetric"] {{
            background:rgba(255,255,255,.74); border:1px solid rgba(25,54,45,.1); border-radius:22px; padding:.9rem .85rem; box-shadow:0 12px 28px rgba(31,58,40,.06);
        }}
        [data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap:.55rem; margin-bottom:.7rem; }}
        [data-testid="stTabs"] [data-baseweb="tab"] {{ background:rgba(255,250,240,.84); border:1px solid rgba(139,94,52,.1); border-radius:999px; }}
        [data-testid="stTabs"] [aria-selected="true"] {{ background:linear-gradient(135deg, rgba(229,181,75,.18), rgba(106,148,68,.2)); }}
        button[kind="primary"] {{ background:linear-gradient(135deg, #6c9440, #2d8a73) !important; border:none !important; color:#fff !important; border-radius:999px !important; }}
        .hero-shell {{
            border-radius:34px; padding:30px; margin-bottom:1rem; overflow:hidden; position:relative; color:#fbfaf3;
            background:linear-gradient(135deg, rgba(18,89,75,.96), rgba(43,122,89,.9) 45%, rgba(226,177,79,.44) 100%);
            box-shadow:0 28px 56px rgba(28,58,48,.18);
        }}
        .hero-grid {{ display:grid; grid-template-columns:minmax(0,1.15fr) minmax(280px,.85fr); gap:1.1rem; align-items:stretch; }}
        .hero-kicker {{ text-transform:uppercase; letter-spacing:.22em; font-size:.76rem; opacity:.82; margin-bottom:.5rem; font-family:"Trebuchet MS","Lucida Sans Unicode",sans-serif; }}
        .hero-title {{ font-family:Georgia,"Times New Roman",serif; font-size:clamp(2.3rem,3vw,3.35rem); line-height:.98; margin:0; max-width:12ch; }}
        .hero-copy {{ max-width:720px; margin-top:.9rem; font-size:1.03rem; line-height:1.68; opacity:.97; }}
        .chip-row {{ display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1rem; }}
        .chip {{ padding:.48rem .8rem; border-radius:999px; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.18); font-size:.84rem; }}
        .hero-art {{
            background:linear-gradient(180deg, rgba(255,255,255,.08), rgba(255,255,255,0)), url("{farm_scene}");
            background-size:cover; background-position:center; min-height:310px; border-radius:28px; border:1px solid rgba(255,255,255,.18); position:relative; overflow:hidden;
        }}
        .art-note, .art-footer {{
            position:absolute; border-radius:18px; padding:.72rem .84rem; font-size:.84rem; line-height:1.35; box-shadow:0 12px 24px rgba(32,58,41,.1);
        }}
        .art-note {{ top:16px; right:16px; max-width:170px; background:rgba(255,251,243,.9); color:var(--ink); }}
        .art-footer {{ left:16px; bottom:16px; max-width:215px; background:rgba(18,64,54,.75); color:#fbfaf4; border:1px solid rgba(255,255,255,.16); }}
        .mini-card, .section-card, .crop-strip, .field-notes {{
            background:linear-gradient(180deg, rgba(255,255,255,.86), rgba(253,248,239,.94));
            border:1px solid var(--line); border-radius:28px; box-shadow:var(--shadow);
        }}
        .mini-card {{ padding:1rem; min-height:134px; }}
        .section-card, .field-notes {{ padding:1.12rem 1.18rem; }}
        .metric-label {{ text-transform:uppercase; letter-spacing:.12em; font-size:.71rem; color:rgba(25,54,45,.64); margin-bottom:.35rem; }}
        .metric-value {{ font-family:Georgia,"Times New Roman",serif; font-size:1.95rem; line-height:1; color:var(--ink); margin-bottom:.35rem; }}
        .metric-copy {{ color:rgba(25,54,45,.76); font-size:.94rem; line-height:1.45; }}
        .crop-strip {{ padding:1rem; margin-bottom:1rem; }}
        .crop-head-title {{ font-family:Georgia,"Times New Roman",serif; font-size:1.38rem; color:var(--ink); }}
        .crop-head-copy {{ color:rgba(25,54,45,.74); font-size:.94rem; margin:.2rem 0 .8rem; }}
        .crop-gallery {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.75rem; }}
        .crop-tile {{ background:rgba(255,255,255,.78); border:1px solid rgba(25,54,45,.09); border-radius:24px; padding:.9rem .85rem; min-height:146px; box-shadow:0 12px 24px rgba(31,58,40,.05); }}
        .crop-tile.active {{ background:linear-gradient(180deg, rgba(234,244,216,.95), rgba(255,255,255,.84)); border-color:rgba(106,148,68,.28); transform:translateY(-2px); }}
        .crop-emoji {{ font-size:2rem; line-height:1; }} .crop-name {{ font-family:Georgia,"Times New Roman",serif; font-size:1.03rem; margin-top:.55rem; color:var(--ink); }}
        .crop-season {{ color:rgba(25,54,45,.68); font-size:.84rem; margin-top:.24rem; }} .crop-chip {{ display:inline-block; margin-top:.58rem; padding:.32rem .55rem; border-radius:999px; font-size:.76rem; color:var(--teal); background:rgba(59,134,168,.1); border:1px solid rgba(59,134,168,.14); }}
        .result-card {{
            border-radius:30px; padding:1.45rem 1.5rem; margin-bottom:.85rem; position:relative; overflow:hidden;
            border:1px solid rgba(20,84,68,.14); background:linear-gradient(135deg, rgba(255,255,255,.94), rgba(255,250,240,.92)); box-shadow:0 24px 40px rgba(31,58,40,.12);
        }}
        .result-card::after {{ content:""; position:absolute; right:-6px; bottom:-6px; width:182px; height:182px; background:url("{leaf_badge}") no-repeat center/contain; opacity:.9; }}
        .result-pill {{ display:inline-block; padding:.42rem .72rem; border-radius:999px; background:rgba(15,124,107,.1); color:var(--teal); font-size:.82rem; border:1px solid rgba(15,124,107,.16); margin-bottom:.8rem; }}
        .result-copy {{ max-width:66%; color:rgba(25,54,45,.75); font-size:.95rem; line-height:1.56; }} .result-chips {{ display:flex; gap:.5rem; flex-wrap:wrap; margin-top:1rem; max-width:72%; }}
        .result-chip {{ padding:.38rem .62rem; border-radius:999px; background:rgba(229,181,75,.14); border:1px solid rgba(229,181,75,.18); font-size:.79rem; color:#6f5626; }}
        .story-title {{ font-family:Georgia,"Times New Roman",serif; font-size:1.45rem; color:var(--ink); margin-bottom:.24rem; }} .story-copy {{ color:rgba(25,54,45,.75); font-size:.95rem; line-height:1.58; max-width:78%; margin-bottom:.95rem; }}
        .step-card {{ border-left:4px solid var(--water); background:rgba(255,255,255,.66); border-radius:18px; padding:.84rem .95rem; margin-bottom:.72rem; }}
        .step-label {{ color:var(--water); text-transform:uppercase; letter-spacing:.13em; font-size:.72rem; margin-bottom:.25rem; }}
        .field-notes {{ position:relative; overflow:hidden; }} .field-notes::after {{ content:""; position:absolute; right:-20px; top:-14px; width:150px; height:150px; background:radial-gradient(circle, rgba(255,214,92,.3), rgba(255,214,92,0) 70%); }}
        .status-good {{ color:var(--teal); font-weight:600; }} .status-warn {{ color:var(--alert); font-weight:600; }}
        @media (max-width:1100px) {{ .hero-grid {{ grid-template-columns:1fr; }} .crop-gallery {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} .result-copy,.result-chips,.story-copy {{ max-width:100%; }} }}
        @media (max-width:760px) {{ .crop-gallery {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .hero-shell {{ padding:22px 20px; }} .hero-art {{ min-height:240px; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(ttl=60, show_spinner=False)
def fetch_backend_health(api_url: str) -> dict[str, Any]:
    response = requests.get(f"{api_url.rstrip('/')}/health", timeout=8)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=180, show_spinner=False)
def fetch_backend_model_info(api_url: str) -> list[dict[str, Any]]:
    response = requests.get(f"{api_url.rstrip('/')}/model-info", timeout=8)
    response.raise_for_status()
    return response.json()


def format_crop_name(crop_key: str) -> str:
    return CROP_DISPLAY_NAMES.get(crop_key, crop_key.replace("_", " ").title())


def format_crop_label(crop_key: str) -> str:
    """Dropdown label: emoji + name + cluster tag."""
    cluster = CROPS[crop_key]["cluster"]
    return f"{CROP_EMOJIS.get(crop_key, '🌱')} {format_crop_name(crop_key)} — Cluster {cluster}"


def build_runtime_snapshot(runtime_mode: str, api_url: str) -> tuple[dict[str, Any], list[dict[str, Any]], str | None]:
    if runtime_mode == "Local engine":
        return get_model_status(), get_model_info(), None
    try:
        return fetch_backend_health(api_url), fetch_backend_model_info(api_url), None
    except requests.RequestException as exc:
        return (
            {"loaded_crops": [], "missing_crops": SUPPORTED_CROPS, "cluster_errors": {}, "crops_total": len(SUPPORTED_CROPS)},
            [],
            str(exc),
        )


def recommendation_via_backend(api_url: str, **kwargs: Any) -> dict[str, Any]:
    response = requests.post(f"{api_url.rstrip('/')}/recommend", json=kwargs, timeout=20)
    response.raise_for_status()
    return response.json()


def render_hero(runtime_mode: str, loaded_count: int, total_count: int, selected_crop: str) -> None:
    cluster_id = CROPS[selected_crop]["cluster"]
    cluster_label = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-grid">
                <div>
                    <div class="hero-kicker">Smart Irrigation For Small Farms</div>
                    <h1 class="hero-title">AquaSmart Farm Companion</h1>
                    <div class="hero-copy">
                        Data-driven irrigation decisions for 15 French crops grouped into 4 agronomic clusters.
                        One ML model per cluster replaces per-crop training while keeping crop-specific features
                        like Kc coefficients and soil water capacity.
                    </div>
                    <div class="chip-row">
                        <div class="chip">Runtime: {runtime_mode}</div>
                        <div class="chip">Clusters ready: {loaded_count}/{total_count}</div>
                        <div class="chip">Focus: {CROP_EMOJIS.get(selected_crop, "🌱")} {format_crop_name(selected_crop)} — {cluster_label}</div>
                    </div>
                </div>
                <div class="hero-art">
                    <div class="art-note"><strong>Field mood</strong><br>Farm rows, sunrise tones, crop symbols, and a softer agricultural visual language.</div>
                    <div class="art-footer">France coverage across 15 crops in 4 agronomic clusters.</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_status_cards(status: dict[str, Any], runtime_mode: str, runtime_error: str | None) -> None:
    loaded_clusters = status.get("loaded_clusters", [])
    total_clusters = status.get("clusters_total", 4)
    loaded_count = len(loaded_clusters)
    missing_count = total_clusters - loaded_count

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="mini-card"><div class="metric-label">Runtime</div>'
            f'<div class="metric-value">{runtime_mode}</div>'
            f'<div class="metric-copy">One recommendation engine, shared by Streamlit and FastAPI.</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="mini-card"><div class="metric-label">Model Readiness</div>'
            f'<div class="metric-value">{loaded_count}/{total_clusters}</div>'
            f'<div class="metric-copy">Cluster bundles missing: {missing_count}.</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        status_label = "Healthy" if loaded_count == total_clusters else "Waiting for models"
        status_class = "status-good" if loaded_count == total_clusters else "status-warn"
        runtime_copy = runtime_error or "Live weather still comes from Open-Meteo at request time."
        st.markdown(
            f'<div class="mini-card"><div class="metric-label">System Status</div>'
            f'<div class="metric-value {status_class}">{status_label}</div>'
            f'<div class="metric-copy">{runtime_copy}</div></div>',
            unsafe_allow_html=True,
        )


def render_crop_showcase(active_crop: str) -> None:
    cards = []
    for crop in SORTED_CROPS:
        active_class = " active" if crop == active_crop else ""
        cluster_id = CROPS[crop]["cluster"]
        cards.append(
            f'<div class="crop-tile{active_class}">'
            f'<div class="crop-emoji">{CROP_EMOJIS.get(crop, "🌱")}</div>'
            f'<div class="crop-name">{format_crop_name(crop)}</div>'
            f'<div class="crop-season">Season {CROP_SEASONS.get(crop, "")}</div>'
            f'<div class="crop-chip">Cluster {cluster_id}</div>'
            f'</div>'
        )
    st.markdown(
        f'<section class="crop-strip"><div class="crop-head-title">Crop Deck</div>'
        f'<div class="crop-head-copy">15 French crops grouped into 4 agronomic clusters. '
        f'The highlighted tile follows the crop currently selected in the app.</div>'
        f'<div class="crop-gallery">{"".join(cards)}</div></section>',
        unsafe_allow_html=True,
    )


def render_result(result: dict[str, Any]) -> None:
    decision = "Irrigation recommended" if result["irrigate"] else "No irrigation today"
    cluster_id = result.get("cluster", "?")
    cluster_name = result.get("cluster_name", "")
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-pill">{decision}</div>
            <div class="metric-value">{result['amount_liters']:,.0f} L</div>
            <div class="result-copy">{result['message']}</div>
            <div class="result-chips">
                <div class="result-chip">Depth {result['amount_mm']:.2f} mm</div>
                <div class="result-chip">Confidence {result['confidence'].title()}</div>
                <div class="result-chip">Stage {result['growth_stage'].replace('_', ' ').title()}</div>
                <div class="result-chip">Cluster {cluster_id} — {cluster_name}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Water Volume", f"{result['amount_liters']:,.0f} L")
    metric2.metric("Depth", f"{result['amount_mm']:.2f} mm")
    metric3.metric("Confidence", result["confidence"].title())
    metric4.metric("Growth Stage", result["growth_stage"].replace("_", " ").title())
    st.caption(
        f"Location: {result['location']} | Weather: {result['weather_summary']} | "
        f"Recommendation date: {result['recommendation_date']}"
    )
    weather = result["data_sources"]
    card_cols = st.columns(3)
    labels = [
        ("🌡 Temperature", f"{weather['temperature_2m_mean']:.1f} C"),
        ("💧 Humidity", f"{weather['relative_humidity_2m_mean']:.0f}%"),
        ("🌧 Rain", f"{weather['precipitation_sum']:.1f} mm"),
        ("☀ ET0", f"{weather['et0_fao_evapotranspiration']:.1f} mm/day"),
        ("🍃 Wind", f"{weather['wind_speed_10m_max']:.1f} km/h"),
        ("🌱 Soil Moisture", f"{weather['soil_moisture_0_to_7cm_mean']:.3f}"),
    ]
    for index, (label, value) in enumerate(labels):
        with card_cols[index % 3]:
            st.markdown(
                f'<div class="mini-card"><div class="metric-label">{label}</div>'
                f'<div class="metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )
    if weather.get("soil_moisture_source"):
        st.caption(f"Soil moisture source: {weather['soil_moisture_source']}")
    with st.expander("Raw recommendation payload"):
        st.json(result)


def render_steps(crop: str, growth_stage: str) -> None:
    cluster_id = CROPS[crop]["cluster"]
    cluster_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
    st.markdown(
        f"""
        <div class="section-card">
            <div class="story-title">How AquaSmart Decides</div>
            <div class="story-copy">Your active crop is <strong>{format_crop_name(crop)}</strong>, mapped to <strong>{cluster_name}</strong>. Today's field phase is <strong>{growth_stage.replace('_', ' ').title()}</strong>.</div>
            <div class="step-card"><div class="step-label">Step 1</div><strong>Geocode the field city.</strong><br>AquaSmart resolves the location and fetches same-day weather from Open-Meteo.</div>
            <div class="step-card"><div class="step-label">Step 2</div><strong>Detect crop stage & cluster.</strong><br>The sowing-month calendar maps the selected crop to its current phase, and the crop is routed to its agronomic cluster model.</div>
            <div class="step-card"><div class="step-label">Step 3</div><strong>Build the feature vector.</strong><br>9 static features (weather + Kc + soil capacity) are combined with 7 temporal features (rolling water balance + irrigation history).</div>
            <div class="step-card"><div class="step-label">Step 4</div><strong>Run the two-stage model.</strong><br>Classification answers "irrigate?", then regression answers "how much?" — the output mm are converted to liters using your field area.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_operator_notes(runtime_error: str | None) -> None:
    st.markdown('<div class="field-notes">', unsafe_allow_html=True)
    st.subheader("Field Notes")
    st.markdown(
        """
        - Local engine mode runs the shared AquaSmart inference code directly inside Streamlit.
        - FastAPI mode calls `/recommend` on the configured backend URL.
        - If the crop is in a fallow period, AquaSmart intentionally returns zero irrigation.
        - Soil moisture prefers `0-7cm` live data and falls back to `0-1cm` when Open-Meteo does not expose the deeper band.
        - Recent irrigation history is essential: models were trained with rolling 7/14-day irrigation sums.
        """,
    )
    if runtime_error:
        st.warning(f"Backend status check failed: {runtime_error}")
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    inject_styles()
    st.sidebar.title("AquaSmart")
    runtime_mode = st.sidebar.radio("Execution mode", RUNTIME_OPTIONS, index=0)
    api_url = st.sidebar.text_input("FastAPI URL", DEFAULT_API_URL, disabled=runtime_mode == "Local engine")
    if st.sidebar.button("Reload model cache", use_container_width=True):
        refresh_model_store()
        fetch_backend_health.clear()
        fetch_backend_model_info.clear()
        st.sidebar.success("Model cache cleared.")

    status, model_info, runtime_error = build_runtime_snapshot(runtime_mode, api_url)
    loaded_count = len(status.get("loaded_clusters", []))
    total_count = status.get("clusters_total", 4)
    selected_crop = st.session_state.get("last_crop", "corn")

    st.sidebar.markdown("### Coverage (15 crops)")
    for crop in SORTED_CROPS:
        cluster_id = CROPS[crop]["cluster"]
        st.sidebar.markdown(
            f"- {CROP_EMOJIS.get(crop, '🌱')} {format_crop_name(crop)} `C{cluster_id}`"
        )
    st.sidebar.markdown("### Model Readiness")
    st.sidebar.caption(f"{loaded_count}/{total_count} cluster bundles available in the current runtime.")
    if status.get("missing_clusters"):
        missing_ids = ", ".join(f"C{cid}" for cid in status["missing_clusters"])
        st.sidebar.caption(f"Missing clusters: {missing_ids}")

    render_hero(runtime_mode, loaded_count, total_count, selected_crop)
    render_status_cards(status, runtime_mode, runtime_error)
    render_crop_showcase(selected_crop)

    form_col, guide_col = st.columns((1.15, 0.85))
    with form_col:
        st.subheader("Request a Recommendation")
        st.caption(
            "15 crops, 4 cluster models. The crop you pick is routed to its agronomic cluster automatically."
        )
        with st.form("recommendation_form", clear_on_submit=False):
            city = st.text_input("City", value="Chartres", help="Closest city or town to the field.")
            surface_hectares = st.number_input(
                "Field size (hectares)", min_value=0.1, max_value=500.0, value=5.0, step=0.5
            )
            default_index = (
                SORTED_CROPS.index(selected_crop) if selected_crop in SORTED_CROPS
                else SORTED_CROPS.index("corn")
            )
            crop = st.selectbox(
                "Crop",
                SORTED_CROPS,
                index=default_index,
                format_func=format_crop_label,
            )
            auto_stage = get_growth_stage(crop)
            override_stage = st.toggle("Override auto-detected growth stage", value=False)
            stage_name: str | None = None
            if override_stage:
                stage_name = st.selectbox(
                    "Growth stage",
                    list(GROWTH_STAGE_ENCODING.keys()),
                    index=list(GROWTH_STAGE_ENCODING.keys()).index(auto_stage),
                    format_func=lambda value: value.replace("_", " ").title(),
                )
            else:
                st.caption(
                    f"Auto stage for today: {auto_stage.replace('_', ' ').title()} "
                    f"for {format_crop_name(crop)}."
                )

            with st.expander("🚿 Recent irrigation history (optional but recommended)", expanded=False):
                st.caption(
                    "The model was trained with rolling 7/14-day irrigation sums. "
                    "Providing them improves accuracy, especially for repeat-recommendation scenarios."
                )
                irrigation_last_7d = st.number_input(
                    "Irrigation applied in the last 7 days (mm)",
                    min_value=0.0, max_value=200.0, value=0.0, step=5.0,
                    help="Total millimeters irrigated on this field during the past week.",
                )
                irrigation_last_14d = st.number_input(
                    "Irrigation applied in the last 14 days (mm)",
                    min_value=0.0, max_value=400.0, value=0.0, step=5.0,
                    help="Total millimeters over the past two weeks (≥ 7-day value).",
                )
                days_since_last_irrigation = st.number_input(
                    "Days since last irrigation",
                    min_value=0, max_value=30, value=30, step=1,
                    help="How many days ago was the field last irrigated? Capped at 30.",
                )

            submitted = st.form_submit_button("Generate smart irrigation plan", use_container_width=True)

        st.session_state["last_crop"] = crop
        if submitted:
            effective_14d = max(float(irrigation_last_14d), float(irrigation_last_7d))
            try:
                with st.spinner("Running weather fetch and cluster-specific prediction..."):
                    call_kwargs = dict(
                        city=city,
                        surface_hectares=surface_hectares,
                        crop=crop,
                        growth_stage=stage_name,
                        irrigation_last_7d=float(irrigation_last_7d),
                        irrigation_last_14d=effective_14d,
                        days_since_last_irrigation=float(days_since_last_irrigation),
                    )
                    if runtime_mode == "Local engine":
                        result = build_farmer_recommendation(**call_kwargs)
                    else:
                        result = recommendation_via_backend(api_url=api_url, **call_kwargs)
                st.session_state["last_result"] = result
                st.session_state["last_crop"] = crop
                st.session_state["last_stage"] = result["growth_stage"]
            except ModelNotReadyError as exc:
                st.error(
                    f"{exc}\n\nTrain the cluster bundles under `models/cluster_<N>/` "
                    "before using local inference."
                )
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"The FastAPI backend returned an error: {detail}")
            except requests.RequestException as exc:
                st.error(f"Network request failed: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
        if "last_result" in st.session_state:
            render_result(st.session_state["last_result"])
    with guide_col:
        current_crop = st.session_state.get("last_crop", SORTED_CROPS[0])
        current_stage = st.session_state.get("last_stage", get_growth_stage(current_crop))
        render_steps(current_crop, current_stage)
        render_operator_notes(runtime_error)


if __name__ == "__main__":
    main()