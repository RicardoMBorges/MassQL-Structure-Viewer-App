import html
import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="MassQL Structure Viewer", layout="wide")

# -----------------------------
# LOGOs (optional)
# -----------------------------
STATIC_DIR = Path(__file__).parent / "static"
for logo_name in ["LAABio.png"]:  # "logo_massQL.png",
    p = STATIC_DIR / logo_name
    try:
        from PIL import Image
        st.sidebar.image(Image.open(p), use_container_width=True)
    except Exception:
        pass

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create a lowercase/trimmed lookup without changing original headers."""
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


CANONICAL_NAMES = {
    "scan": ["#Scan#", "Scan", "scan", "scan_number"],
    "query_validation": ["query_validation", "Query_Validation", "validation"],
    "compound_name": ["Compound_Name", "compound_name", "Name", "name"],
    "smiles": ["Smiles", "SMILES", "smiles"],
    "inchi": ["INCHI", "InChI", "inchi"],
    "inchikey": ["InChIKey", "INCHIKEY", "inchikey"],
}


def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    existing = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().strip()
        if key in existing:
            return existing[key]
    return None


@st.cache_data(show_spinner=False)
def load_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep=";")
    elif name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(uploaded_file, sep="\t")
    else:
        return pd.read_csv(uploaded_file)


def has_valid_smiles(smiles) -> bool:
    return pd.notna(smiles) and str(smiles).strip() != ""


def smiles_to_html(smiles, canvas_id: str, width: int = 220, height: int = 180) -> str:
    if not has_valid_smiles(smiles):
        return '<div class="small-note">No valid structure</div>'

    smiles_clean = str(smiles).strip()
    smiles_str = json.dumps(smiles_clean)

    return f"""
    <div class="mol-canvas-wrap">
        <canvas id="{canvas_id}" width="{width}" height="{height}"></canvas>
    </div>
    <script>
    (function() {{
        const smiles = {smiles_str};
        const targetId = "{canvas_id}";

@st.cache_data(show_spinner=False)
def prepare_result_table(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    scan_col = find_first_existing_column(df, CANONICAL_NAMES["scan"])
    query_col = find_first_existing_column(df, CANONICAL_NAMES["query_validation"])
    name_col = find_first_existing_column(df, CANONICAL_NAMES["compound_name"])
    smiles_col = find_first_existing_column(df, CANONICAL_NAMES["smiles"])
    inchi_col = find_first_existing_column(df, CANONICAL_NAMES["inchi"])
    inchikey_col = find_first_existing_column(df, CANONICAL_NAMES["inchikey"])

    result = pd.DataFrame()

    if scan_col:
        result["#Scan#"] = df[scan_col]
    if query_col:
        result["query_validation"] = df[query_col]
    if name_col:
        result["Compound_Name"] = df[name_col]
    if smiles_col:
        result["Smiles"] = df[smiles_col].astype(str).str.strip()
        result.loc[result["Smiles"].isin(["", "nan", "None", "NaN", "null", "NULL"]), "Smiles"] = None
    else:
        result["Smiles"] = None
    if inchi_col:
        result["INCHI"] = df[inchi_col].astype(str).str.strip()
        result.loc[result["INCHI"].isin(["", "nan", "None", "NaN", "null", "NULL"]), "INCHI"] = None
    else:
        result["INCHI"] = None
    if inchikey_col:
        result["InChIKey"] = df[inchikey_col]

    result["Structure_Status"] = result.apply(
        lambda row: (
            "SMILES available"
            if has_valid_smiles(row.get("Smiles"))
            else ("InChI only" if pd.notna(row.get("INCHI")) and str(row.get("INCHI")).strip() else "Invalid / missing")
        ),
        axis=1,
    )

    return result


def build_single_molecule_html(smiles, width: int = 500, height: int = 350) -> str:
    canvas_id = "single_molecule_canvas"
    if not has_valid_smiles(smiles):
        return """
        <div style="padding:1rem; color:#666; font-size:14px;">
            No valid molecular structure could be generated for this row.
        </div>
        """

    smiles_clean = str(smiles).strip()
    smiles_str = json.dumps(smiles_clean)

    return f"""


def build_single_molecule_html(smiles, width: int = 500, height: int = 350) -> str:
    canvas_id = "single_molecule_canvas"
    if not has_valid_smiles(smiles):
        return """
        <div style="padding:1rem; color:#666; font-size:14px;">
            No valid molecular structure could be generated for this row.
        </div>
        """

    smiles_clean = str(smiles).strip()
    smiles_str = json.dumps(smiles_clean)
    const smiles = {smiles_str};

    return f"""
    <style>
    .single-mol-wrap {{
        width: 100%;
        min-height: {height}px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        background: white;
    }}
    </style>
    <script src="https://unpkg.com/smiles-drawer@2.1.7/dist/smiles-drawer.min.js"></script>
    <div class="single-mol-wrap">
        <canvas id="{canvas_id}" width="{width}" height="{height}"></canvas>
    </div>
    <script>
    (function() {{
        const smiles = {smiles_str};
        const targetId = "{canvas_id}";

        function drawMolecule() {{
            if (typeof SmilesDrawer === "undefined") {{
                const el = document.getElementById(targetId);
                if (el && el.parentElement) {{
                    el.parentElement.innerHTML = '<div style="color:#666;">Could not load SmilesDrawer</div>';
                }}
                return;
            }}

            SmilesDrawer.parse(smiles, function(tree) {{
                const drawer = new SmilesDrawer.Drawer({{
                    width: {width},
                    height: {height},
                    padding: 20
                }});
                drawer.draw(tree, targetId, "light", false);
            }}, function() {{
                const el = document.getElementById(targetId);
                if (el && el.parentElement) {{
                    el.parentElement.innerHTML = '<div style="color:#666;">Invalid SMILES</div>';
                }}
            }});
        }}

        if (typeof SmilesDrawer === "undefined") {{
            setTimeout(drawMolecule, 200);
        }} else {{
            drawMolecule();
        }}
    }})();
    </script>
    """


# ---------------------------------------------------------
# App
# ---------------------------------------------------------

st.title("MassQL Structure Viewer")
st.write(
    'Upload a CSV table exported after running MassQL queries, then inspect the recognized molecules as 2D chemical structures.'
)

with st.expander("Expected columns", expanded=False):
    st.markdown(
        """
        The app looks for these columns automatically:

        - `#Scan#`
        - `query_validation`
        - `Compound_Name`
        - `Smiles`
        - `INCHI` or `InChI`
        - `InChIKey`

        Structures are rendered from **SMILES**.  
        **InChI** and **InChIKey** are preserved as annotation columns.
        """
    )

uploaded_file = st.sidebar.file_uploader(
    "Upload MassQL result table",
    type=["csv", "tsv", "txt"],
    accept_multiple_files=False,
)

with st.sidebar.expander("Expected columns", expanded=False):
    st.markdown(
        """
Upload the CSV file exported from **"Table With Library Matches Only"** after running MassQL at https://massqlpostmn.gnps2.org/.  
This table contains the validated library matches (including SMILES, InChI, and InChIKey), which will be used here to render the molecules as 2D chemical structures.
        """
    )

st.sidebar.markdown("---")

st.sidebar.link_button(
    "Check also: massql-compendium-builder-from-csv app.py",
    "https://massql-builder-from-csv.streamlit.app/",
    use_container_width=True,
)

st.sidebar.link_button(
    "Check also: massql_compendium_runner app.py",
    "https://massql-compendium-runner.streamlit.app/",
    use_container_width=True,
)

st.sidebar.markdown("---")

if uploaded_file is not None:
    try:
        raw_df = load_table(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    st.subheader("Raw uploaded table")
    st.dataframe(raw_df, use_container_width=True)

    result_df = prepare_result_table(raw_df)

    st.subheader("Processed annotation table")
    st.dataframe(result_df, use_container_width=True)

    valid_count = (result_df["Structure_Status"] == "SMILES available").sum()
    invalid_count = (result_df["Structure_Status"] != "OK").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(result_df))
    c2.metric("Valid structures", int(valid_count))
    c3.metric("Invalid / missing", int(invalid_count))

    st.subheader("Structure table")
    max_rows = st.slider("Maximum rows to render as molecule table", 10, 500, 100, 10)
    html_table = build_html_table(result_df, max_rows=max_rows)
    st.components.v1.html(html_table, height=700, scrolling=True)

    st.subheader("Single-compound browser")
    if "Compound_Name" in result_df.columns:
        label_series = result_df["Compound_Name"].fillna("Unnamed")
    else:
        label_series = pd.Series([f"Row {i+1}" for i in range(len(result_df))])

    selected_idx = st.selectbox(
        "Select one row",
        options=list(range(len(result_df))),
        format_func=lambda i: f"{i + 1} — {label_series.iloc[i]}",
    )

    selected_row = result_df.iloc[selected_idx]

    col_a, col_b = st.columns([1, 1.3])
    with col_a:
        st.components.v1.html(
            build_single_molecule_html(selected_row.get("Smiles"), width=500, height=350),
            height=380,
            scrolling=False,
        )

    with col_b:
        details = selected_row.to_frame(name="Value")
        st.dataframe(details, use_container_width=True)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download processed table as CSV",
        data=csv_bytes,
        file_name="massql_structure_viewer_processed.csv",
        mime="text/csv",
    )

else:

    st.info("Upload a MassQL result table to begin.")
