# MassQL Structure Viewer App
# by Ricardo Moreira Borges (IPPN-UFRJ)

import html
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st


st.set_page_config(page_title="MassQL Structure Viewer", layout="wide")

# -----------------------------
# LOGOs (optional)
# -----------------------------
STATIC_DIR = Path(__file__).parent / "static"
for logo_name in ["LAABio.png"]:
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
    """Trim column names without changing their meaning."""
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
    existing = {str(c).lower().strip(): c for c in df.columns}
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

    if name.endswith(".tsv") or name.endswith(".txt"):
        return pd.read_csv(uploaded_file, sep="\t")

    return pd.read_csv(uploaded_file)


def clean_text_value(value):
    if pd.isna(value):
        return None

    text = str(value).strip()
    if text in {"", "nan", "None", "NaN", "null", "NULL"}:
        return None

    return text


def sanitize_smiles_for_url(smiles):
    smiles = clean_text_value(smiles)
    if smiles is None:
        return None

    s = smiles.strip()

    # remove surrounding quotes
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()

    # remove CXSMILES-like annotation block if present
    if " |" in s:
        s = s.split(" |", 1)[0].strip()

    # remove trailing whitespace fragments
    s = s.split()[0].strip()

    return s if s else None


def choose_structure_identifier(smiles, inchi, inchikey):
    smiles_clean = sanitize_smiles_for_url(smiles)
    inchi_clean = clean_text_value(inchi)
    inchikey_clean = clean_text_value(inchikey)

    if inchikey_clean:
        # NCI resolver accepts full Standard InChIKeys
        return f"InChIKey={inchikey_clean}"

    if inchi_clean:
        return inchi_clean

    if smiles_clean:
        return smiles_clean

    return None


def build_nci_image_url(smiles=None, inchi=None, inchikey=None, size=300) -> str | None:
    """
    Use NIH NCI/CADD Chemical Identifier Resolver image endpoint.
    Documentation pattern:
      /chemical/structure/"structure identifier"/image
    """
    identifier = choose_structure_identifier(smiles, inchi, inchikey)
    if identifier is None:
        return None

    # URL-encode identifier safely
    encoded_identifier = quote(identifier, safe="")
    # png option is supported by the resolver's image method/options
    return (
        f"https://cactus.nci.nih.gov/chemical/structure/"
        f"{encoded_identifier}/image?format=png&width={size}&height={size}"
    )


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
        result["Smiles"] = df[smiles_col].map(clean_text_value)
    else:
        result["Smiles"] = None

    if inchi_col:
        result["INCHI"] = df[inchi_col].map(clean_text_value)
    else:
        result["INCHI"] = None

    if inchikey_col:
        result["InChIKey"] = df[inchikey_col].map(clean_text_value)
    else:
        result["InChIKey"] = None

    result["Smiles_for_URL"] = result["Smiles"].map(sanitize_smiles_for_url)

    result["Structure_Source"] = result.apply(
        lambda row: (
            "InChIKey"
            if clean_text_value(row.get("InChIKey")) is not None
            else (
                "InChI"
                if clean_text_value(row.get("INCHI")) is not None
                else (
                    "SMILES"
                    if clean_text_value(row.get("Smiles")) is not None
                    else "Missing"
                )
            )
        ),
        axis=1,
    )

    result["Has_structure_identifier"] = result.apply(
        lambda row: choose_structure_identifier(
            row.get("Smiles"),
            row.get("INCHI"),
            row.get("InChIKey"),
        ) is not None,
        axis=1,
    )

    return result


def build_html_table(result_df: pd.DataFrame, max_rows: int = 200, image_size: int = 220) -> str:
    show_df = result_df.head(max_rows).copy()

    headers = [c for c in show_df.columns if c not in ["Smiles", "INCHI"]]
    html_parts = []

    html_parts.append(
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
            max-height: 220px;
            object-fit: contain;
            display: block;
            margin: 0 auto;
        }
        .small-note {
            color: #666;
            font-size: 12px;
        }
        </style>
        """
    )

    html_parts.append('<table class="mol-table">')
    html_parts.append("<thead><tr>")
    html_parts.append("<th>Structure</th>")

    for h in headers:
        html_parts.append(f"<th>{html.escape(str(h))}</th>")

    html_parts.append("</tr></thead><tbody>")

    for _, row in show_df.iterrows():
        image_url = build_nci_image_url(
            smiles=row.get("Smiles"),
            inchi=row.get("INCHI"),
            inchikey=row.get("InChIKey"),
            size=image_size,
        )

        html_parts.append("<tr>")

        if image_url is not None:
            img_html = f'<img class="mol-img" src="{html.escape(image_url)}" alt="structure"/>'
        else:
            img_html = '<div class="small-note">No usable identifier</div>'

        html_parts.append(f'<td class="mol-cell">{img_html}</td>')

        for h in headers:
            value = row.get(h, "")
            if pd.isna(value):
                value = ""
            html_parts.append(f"<td>{html.escape(str(value))}</td>")

        html_parts.append("</tr>")

    html_parts.append("</tbody></table>")
    return "".join(html_parts)


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

        Structure images are retrieved using the best available identifier in this order:

        1. `InChIKey`
        2. `INCHI`
        3. `Smiles`
        """
    )

uploaded_file = st.sidebar.file_uploader(
    "Upload MassQL result table",
    type=["csv", "tsv", "txt"],
    accept_multiple_files=False,
)


with st.sidebar.expander("Expected columns", expanded=False):
    st.markdown(
        """by Ricardo Moreira Borges (IPPN-UFRJ)""")
    
with st.sidebar.expander("Expected columns", expanded=False):
    st.markdown(
        """
Upload the CSV file exported from **"Table With Library Matches Only"** after running MassQL at https://massqlpostmn.gnps2.org/.  
This table contains the validated library matches (including SMILES, InChI, and InChIKey), which will be used here to display the structures.
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

    st.subheader("Identifier debug table")
    st.dataframe(
        result_df[
            [
                "Compound_Name",
                "Smiles",
                "Smiles_for_URL",
                "INCHI",
                "InChIKey",
                "Structure_Source",
                "Has_structure_identifier",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("Processed annotation table")
    st.dataframe(result_df, use_container_width=True)

    valid_count = result_df["Has_structure_identifier"].sum()
    missing_count = (~result_df["Has_structure_identifier"]).sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(result_df))
    c2.metric("Rows with structure identifier", int(valid_count))
    c3.metric("Missing identifier", int(missing_count))

    st.subheader("Structure table")
    max_rows = st.slider("Maximum rows to render as structure table", 10, 500, 100, 10)
    html_table = build_html_table(result_df, max_rows=max_rows, image_size=220)
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
    image_url = build_nci_image_url(
        smiles=selected_row.get("Smiles"),
        inchi=selected_row.get("INCHI"),
        inchikey=selected_row.get("InChIKey"),
        size=600,
    )

    col_a, col_b = st.columns([1, 1.3])

    with col_a:
        if image_url is not None:
            st.image(image_url, caption=str(label_series.iloc[selected_idx]), use_container_width=True)
        else:
            st.warning("No usable identifier was found for this row.")

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

