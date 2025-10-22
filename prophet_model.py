import pandas as pd
from prophet import Prophet

def predict_next_period(tracked_periods):
    if len(tracked_periods) < 3:
        return "⚠️ Not enough valid historical data to make a prediction."

    df = pd.DataFrame(tracked_periods)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df = df.sort_values("start_date")

    # Calculate cycle lengths
    df["cycle_length"] = df["start_date"].diff().dt.days
    df = df.dropna(subset=["cycle_length"])  # Drop first row

    # ✅ Rename for Prophet
    df.rename(columns={"start_date": "ds", "cycle_length": "y"}, inplace=True)

    # ✅ Check if 'ds' and 'y' exist
    if "ds" not in df.columns or "y" not in df.columns:
        return "⚠️ Failed to prepare data for Prophet."

    # Train the model
    model = Prophet()
    model.fit(df)

    # Predict next cycle length
    future = model.make_future_dataframe(periods=1, freq="D")
    forecast = model.predict(future)

    # Get last predicted cycle length
    predicted_cycle_length = forecast["yhat"].iloc[-1]

    # Get last actual period date
    last_period_date = df["ds"].iloc[-1]

    # Predict next period date
    predicted_start = last_period_date + pd.Timedelta(days=round(predicted_cycle_length))

    return predicted_start
