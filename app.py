import base64
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw


st.set_page_config(page_title="MassQL Structure Viewer", layout="wide")

# -----------------------------
# LOGOs (optional)
# -----------------------------
STATIC_DIR = Path(__file__).parent / "static"
for logo_name in ["LAABio.png"]: #"logo_massQL.png", 
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


@st.cache_data(show_spinner=False)
def mol_from_identifiers(smiles, inchi):
    mol = None

    if pd.notna(smiles) and str(smiles).strip():
        try:
            mol = Chem.MolFromSmiles(str(smiles).strip())
        except Exception:
            mol = None

    if mol is None and pd.notna(inchi) and str(inchi).strip():
        try:
            mol = Chem.MolFromInchi(str(inchi).strip())
        except Exception:
            mol = None

    return mol


@st.cache_data(show_spinner=False)
def mol_to_png_bytes(smiles, inchi, size=(220, 180)):
    mol = mol_from_identifiers(smiles, inchi)
    if mol is None:
        return None

    try:
        Chem.rdDepictor.Compute2DCoords(mol)
    except Exception:
        pass

    img = Draw.MolToImage(mol, size=size)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


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
        result["Smiles"] = df[smiles_col]
    else:
        result["Smiles"] = None
    if inchi_col:
        result["INCHI"] = df[inchi_col]
    else:
        result["INCHI"] = None
    if inchikey_col:
        result["InChIKey"] = df[inchikey_col]

    result["Structure_Status"] = result.apply(
        lambda row: "OK" if mol_from_identifiers(row.get("Smiles"), row.get("INCHI")) is not None else "Invalid / missing",
        axis=1,
    )

    return result


@st.cache_data(show_spinner=False)
def build_html_table(result_df: pd.DataFrame, max_rows: int = 200) -> str:
    show_df = result_df.head(max_rows).copy()

    headers = [c for c in show_df.columns if c not in ["Smiles", "INCHI"]]
    html = []
    html.append(
        """
        <style>
        .mol-table {
            border-collapse: collapse;
            width: 100%;
            font-size: 14px;
        }
        .mol-table th, .mol-table td {
            border: 1px solid #d9d9d9;
            padding: 8px;
            text-align: left;
            vertical-align: middle;
        }
        .mol-table th {
            background-color: #f4f4f4;
            position: sticky;
            top: 0;
            z-index: 1;
        }
        .mol-cell {
            min-width: 240px;
            text-align: center;
        }
        .mol-img {
            max-width: 220px;
            height: auto;
        }
        .small-note {
            color: #666;
            font-size: 12px;
        }
        </style>
        """
    )
    html.append('<table class="mol-table">')
    html.append("<thead><tr>")
    html.append("<th>Structure</th>")
    for h in headers:
        html.append(f"<th>{h}</th>")
    html.append("</tr></thead><tbody>")

    for _, row in show_df.iterrows():
        png = mol_to_png_bytes(row.get("Smiles"), row.get("INCHI"))
        html.append("<tr>")

        if png is not None:
            b64 = base64.b64encode(png).decode("utf-8")
            img_html = f'<img class="mol-img" src="data:image/png;base64,{b64}"/>'
        else:
            img_html = '<div class="small-note">No valid structure</div>'

        html.append(f'<td class="mol-cell">{img_html}</td>')

        for h in headers:
            value = row.get(h, "")
            if pd.isna(value):
                value = ""
            html.append(f"<td>{value}</td>")

        html.append("</tr>")

    html.append("</tbody></table>")
    return "".join(html)


# ---------------------------------------------------------
# App
# ---------------------------------------------------------

st.title("MassQL Structure Viewer")
st.write(
    "Upload a CSV table exported after running MassQL queries, then inspect the recognized molecules as 2D chemical structures."
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

        The structure is drawn from **SMILES first**, and if SMILES is missing or invalid, it tries **InChI**.
        """
    )

uploaded_file = st.sidebar.file_uploader(
    "Upload MassQL result table",
    type=["csv", "tsv", "txt"],
    accept_multiple_files=False,
    #help=
)

with st.sidebar.expander("Expected columns", expanded=False):
    st.markdown(
        """
    Upload the CSV file exported from **"Table With Library Matches Only"** after running MassQL at https://massqlpostmn.gnps2.org/
. This table contains the validated library matches (including SMILES, InChI, and InChIKey), which will be used here to render the molecules as 2D chemical structures.
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

    valid_count = (result_df["Structure_Status"] == "OK").sum()
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
    png = mol_to_png_bytes(selected_row.get("Smiles"), selected_row.get("INCHI"), size=(500, 350))

    col_a, col_b = st.columns([1, 1.3])
    with col_a:
        if png is not None:
            st.image(png, caption=str(label_series.iloc[selected_idx]), use_container_width=True)
        else:
            st.warning("No valid molecular structure could be generated for this row.")

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
