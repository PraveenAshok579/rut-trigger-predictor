# Rut-Trigger Predictor (Streamlit app)

Predicts when a flexible pavement section will first reach a 10 mm rut depth, with
90% bootstrap bands, a risk tier and a data-coverage check. It uses the XGBoost-Cox
model of the manuscript and gives the same predictions as the offline HTML tool.

## Files
- `app.py` – the Streamlit interface (single section, batch CSV, model guidance)
- `rut_engine.py` – the prediction engine (model, Breslow baseline, bootstrap bands, kNN coverage)
- `rut_trigger_model.json` – the trained model, written by `10_rut_trigger_tool.py` (folder `10_tool`)
- `requirements.txt`, `.streamlit/config.toml`

## Run on your computer
```
pip install -r requirements.txt
streamlit run app.py
```
The app opens in the browser at http://localhost:8501.

## Update the model
Re-run `10_rut_trigger_tool.py`, then copy the new `rut_trigger_model.json`
from `output_survival/10_tool/` into this folder (replace the old one).

## Publish online (free public link for the manuscript)
1. Create a GitHub repository and upload all files in this folder, including `.streamlit/`.
2. Sign in at https://share.streamlit.io with GitHub, choose **Create app**, select the
   repository, branch `main` and main file `app.py`, then **Deploy**.
3. The app receives a permanent address such as `https://<name>.streamlit.app`.
   Cite that address (and a Zenodo DOI of the repository, if made) in the manuscript.
