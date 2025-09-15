# A Disaster Response Assistant with Predictive Analytics for Risk Management (NCR-Focused)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import os

CSV_FILE = "risk.csv" 
df_orig = pd.read_csv(CSV_FILE)

risk_map = {"low": 1, "medium": 2, "high": 3}
non_hazard_cols = ["NCR", "place", "cluster", "predicted_risk", "lat", "lon", "recommendation"]
hazard_cols = [c for c in df_orig.columns if c not in non_hazard_cols]

df_num = df_orig.copy()
for col in hazard_cols:
    df_num[col] = df_orig[col].map(risk_map)

imputer = SimpleImputer(strategy="most_frequent")
df_num[hazard_cols] = imputer.fit_transform(df_num[hazard_cols])
scaler = StandardScaler()
X = scaler.fit_transform(df_num[hazard_cols])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_orig["cluster"] = kmeans.fit_predict(X)

cluster_means = df_num.groupby(df_orig["cluster"])[hazard_cols].mean().mean(axis=1).sort_values()
risk_levels = {c: label for c, label in zip(cluster_means.index, ["low", "medium", "high"])}
df_orig["predicted_risk"] = df_orig["cluster"].map(risk_levels)

def get_recommendation(risk):
    if risk == "low":
        return "Stay alert and monitor weather updates."
    elif risk == "medium":
        return "Prepare emergency kit and evacuation plan."
    else:
        return "Follow LGU evacuation orders immediately."

df_orig["recommendation"] = df_orig["predicted_risk"].apply(get_recommendation)

def predict_risk(user_input):
    user_places = [p.strip().lower() for p in user_input]
    place_col_name = "NCR" if "NCR" in df_orig.columns else "place"
    mask = df_orig[place_col_name].str.lower().apply(lambda x: any(place in x for place in user_places))
    df_filtered = df_orig[mask]

    if df_filtered.empty:
        return None, None, None

    #Excel
    excel_file = os.path.join("static", "risk_report.xlsx")
    df_filtered.to_excel(excel_file, index=False)

    #PDF
    pdf_file = os.path.join("static", "risk_report.pdf")
    doc = SimpleDocTemplate(pdf_file)
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "Helvetica"
    styles["Title"].fontName = "Helvetica-Bold"
    story = []
    story.append(Paragraph("Disaster Risk Report in National Capital Region", styles["Title"]))
    story.append(Spacer(1, 12))
    table_data = [["NCR Location", "Predicted Future Risk", "Recommendation"]]
    for _, row in df_filtered.iterrows():
        table_data.append([row[place_col_name], row["predicted_risk"], row["recommendation"]])
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica"),
        ("BOTTOMPADDING", (0,0), (-1,0), 8),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
    ]))
    story.append(table)
    doc.build(story)

    return df_filtered, excel_file, pdf_file
