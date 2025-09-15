from flask import Flask, render_template, request, send_from_directory
import os
from model import predict_risk

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        places = request.form.getlist("places")
        places = [p.strip().lower() for p in places]
        df_filtered, excel_path, pdf_path = predict_risk(places)
        if df_filtered is None:
            return render_template("result.html", error="No matching locations found.")
        
        table_html = df_filtered.to_html(classes="table table-striped", index=False)
        return render_template("result.html",
                               table_html=table_html,
                               excel_file=os.path.basename(excel_path),
                               pdf_file=os.path.basename(pdf_path))
    return render_template("index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(debug=True)
