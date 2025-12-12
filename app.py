import datetime as dt
import math
import requests
import pandas as pd
import streamlit as st

API_BASE = "https://api.octopus.energy/v1"

def get_agile_rates(product_code: str, region_code: str,
                    start: dt.datetime, end: dt.datetime,
                    api_key: str | None = None) -> pd.DataFrame:
    """
    Fetch Agile half‑hourly unit rates (inc. VAT) between start and end (UTC).
    product_code e.g. 'AGILE-24-10-01'
    region_code e.g. 'H' (for East Midlands etc)
    """
    tariff_code = f"E-1R-{product_code}-{region_code}"
    url = f"{API_BASE}/products/{product_code}/electricity-tariffs/{tariff_code}/standard-unit-rates/"
    params = {
        "period_from": start.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "period_to": end.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "page_size": 1500,
    }

    auth = (api_key, "") if api_key else None
    results = []
    while url:
        r = requests.get(url, params=params if not results else None, auth=auth, timeout=15)
        r.raise_for_status()
        data = r.json()
        results.extend(data.get("results", []))
        url = data.get("next")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    # Convert to sensible types
    df["valid_from"] = pd.to_datetime(df["valid_from"], utc=True)
    df["valid_to"] = pd.to_datetime(df["valid_to"], utc=True)
    # Octopus returns p/kWh in 'value_inc_vat' field for unit rates
    df["price_p_per_kwh"] = df["value_inc_vat"].astype(float)
    # Sort chronologically
    df = df.sort_values("valid_from").reset_index(drop=True)
    return df[["valid_from", "valid_to", "price_p_per_kwh"]]

def classify_cheap(df: pd.DataFrame,
                   go_offpeak_rate_p: float,
                   cheap_threshold_pct: float = 0.9) -> pd.DataFrame:
    """
    Define 'cheap' as Agile price < cheap_threshold_pct * Go off‑peak rate.
    Confidence = how far below that threshold the price is, scaled to 0–100.
    """
    threshold = cheap_threshold_pct * go_offpeak_rate_p
    df = df.copy()
    df["is_cheap"] = df["price_p_per_kwh"] < threshold

    # Confidence: 0 at threshold, 100 when at or below 50% of threshold (clip)
    # You can tweak this formula as desired.
    def confidence(price):
        if price >= threshold:
            return 0.0
        min_price = 0.5 * threshold
        if price <= min_price:
            return 100.0
        # Linear between min_price and threshold
        return 100.0 * (threshold - price) / (threshold - min_price)

    df["confidence"] = df["price_p_per_kwh"].apply(confidence)
    return df

def build_streamlit_app():
    st.set_page_config(page_title="Agile vs Intelligent Go Helper", layout="wide")

    st.title("Agile vs Intelligent Octopus Go – Next 48 Hours")

    with st.sidebar:
        st.header("Inputs")

        api_key = st.text_input(
            "Octopus API key",
            type="password",
            help="Find this in your Octopus dashboard developer settings."
        )

        st.markdown("**Agile settings**")
        product_code = st.text_input(
            "Agile product code",
            value="AGILE-24-10-01",
            help="See a current Agile product code from Octopus or tools like Guy Lipman's generic API tool."
        )
        region_code = st.text_input(
            "Region code (A–P)",
            value="H",
            help="Your DNO region letter, used in Agile tariff codes."
        )

        st.markdown("**Intelligent Go settings**")
        go_offpeak_rate_p = st.number_input(
            "Your Intelligent Go off‑peak unit rate (p/kWh)",
            min_value=0.0,
            value=7.5,
            step=0.1,
            help="Enter the off‑peak unit rate from your Intelligent Octopus Go tariff (including VAT)."
        )
        cheap_threshold_pct = st.slider(
            "Cheap threshold vs Go off‑peak",
            min_value=0.5,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Agile is 'cheap' when price < this × Go off‑peak."
        )

        tz_choice = st.selectbox(
            "Display time in",
            ["Local time (UK)", "UTC"],
            index=0
        )

        if st.button("Fetch and analyse next 48 hours"):
            st.session_state["run"] = True

    if "run" not in st.session_state or not st.session_state["run"]:
        st.info("Enter your details in the sidebar and click **Fetch and analyse next 48 hours**.")
        return

    # Time window: next 48h UTC
    now_utc = dt.datetime.now(dt.timezone.utc)
    end_utc = now_utc + dt.timedelta(hours=48)

    try:
        with st.spinner("Retrieving Agile half‑hourly prices…"):
            df = get_agile_rates(
                product_code=product_code.strip(),
                region_code=region_code.strip().upper(),
                start=now_utc,
                end=end_utc,
                api_key=api_key.strip() if api_key else None
            )
    except Exception as e:
        st.error(f"Error retrieving Agile prices: {e}")
        return

    if df.empty:
        st.warning("No Agile price data returned for the given product and region over the next 48 hours.")
        return

    df = classify_cheap(df, go_offpeak_rate_p=go_offpeak_rate_p,
                        cheap_threshold_pct=cheap_threshold_pct)

    # Convert to UK local time if requested
    if tz_choice.startswith("Local"):
        # UK uses Europe/London rules; pandas 2.x has 'Europe/London' if zoneinfo is present
        df["start_local"] = df["valid_from"].dt.tz_convert("Europe/London")
        df["end_local"] = df["valid_to"].dt.tz_convert("Europe/London")
        time_cols = ["start_local", "end_local"]
    else:
        df["start_utc"] = df["valid_from"]
        df["end_utc"] = df["valid_to"]
        time_cols = ["start_utc", "end_utc"]

    st.subheader("Cheap periods and confidence")

    # Summary stats
    cheap_slots = df[df["is_cheap"]]
    st.markdown(
        f"- Total half‑hour slots: **{len(df)}**  \n"
        f"- Cheap slots (Agile < threshold): **{len(cheap_slots)}**  \n"
        f"- Mean price in cheap slots: **{cheap_slots['price_p_per_kwh'].mean():.2f} p/kWh**  \n"
        f"- Threshold price: **{cheap_threshold_pct * go_offpeak_rate_p:.2f} p/kWh**"
    )

    # Compact schedule view
    schedule_df = df[time_cols + ["price_p_per_kwh", "is_cheap", "confidence"]].copy()
    schedule_df.rename(columns={
        time_cols[0]: "start",
        time_cols[1]: "end",
        "price_p_per_kwh": "agile_price_p_per_kwh"
    }, inplace=True)

    # Color coding helper
    def color_cheap(val):
        return "background-color: #d1ffd1" if val else ""

    st.dataframe(
        schedule_df.style.apply(
            lambda s: [color_cheap(v) for v in schedule_df["is_cheap"]],
            axis=0
        ).format({
            "agile_price_p_per_kwh": "{:.2f}",
            "confidence": "{:.0f}"
        }),
        use_container_width=True,
        height=500
    )

    # Chart
    st.subheader("Price and confidence over time")
    chart_df = df.copy()
    chart_df["time"] = chart_df[time_cols[0]]
    chart_df = chart_df[["time", "price_p_per_kwh", "confidence", "is_cheap"]]

    st.line_chart(
        chart_df.set_index("time")[["price_p_per_kwh"]],
        height=250
    )
    st.area_chart(
        chart_df.set_index("time")[["confidence"]],
        height=200
    )

    st.caption(
        "This tool compares Agile half‑hourly prices with your Intelligent Octopus Go off‑peak rate "
        "to highlight periods that should be attractive for shifting demand, using a simple confidence metric."
    )

if __name__ == "__main__":
    build_streamlit_app()
