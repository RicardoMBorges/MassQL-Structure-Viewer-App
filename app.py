# MassQL Structure Viewer App
# by Ricardo Moreira Borges (IPPN-UFRJ)

import html
import json
from pathlib import Path

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

def sanitize_smiles_for_browser(smiles):
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


def has_valid_smiles(smiles) -> bool:
    return clean_text_value(smiles) is not None


def smiles_to_html(smiles, canvas_id: str, width: int = 220, height: int = 180) -> str:
    smiles_clean = sanitize_smiles_for_browser(smiles)
    if smiles_clean is None:
        return '<div class="small-note">No valid structure</div>'

    smiles_str = json.dumps(smiles_clean)

    return f"""
    <div class="mol-canvas-wrap">
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
                    el.parentElement.innerHTML = '<div class="small-note">Could not load SmilesDrawer</div>';
                }}
                return;
            }}

            SmilesDrawer.parse(
                smiles,
                function(tree) {{
                    const drawer = new SmilesDrawer.Drawer({{
                        width: {width},
                        height: {height},
                        padding: 10,
                        experimental: true
                    }});
                    drawer.draw(tree, targetId, "light", false);
                }},
                function() {{
                    const el = document.getElementById(targetId);
                    if (el && el.parentElement) {{
                        el.parentElement.innerHTML = '<div class="small-note">Invalid SMILES</div>';
                    }}
                }}
            );
        }}

        if (typeof SmilesDrawer === "undefined") {{
            setTimeout(drawMolecule, 200);
        }} else {{
            drawMolecule();
        }}
    }})();
    </script>
    """


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

    result["Structure_Status"] = result.apply(
        lambda row: (
            "SMILES available"
            if has_valid_smiles(row.get("Smiles"))
            else (
                "InChI only"
                if clean_text_value(row.get("INCHI")) is not None
                else "Invalid / missing"
            )
        ),
        axis=1,
    )

    return result


def build_html_table(result_df: pd.DataFrame, max_rows: int = 200) -> str:
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
        .mol-canvas-wrap {
            width: 220px;
            min-height: 180px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
        }
        .small-note {
            color: #666;
            font-size: 12px;
        }
        </style>
        <script src="https://unpkg.com/smiles-drawer@2.1.7/dist/smiles-drawer.min.js"></script>
        """
    )

    html_parts.append('<table class="mol-table">')
    html_parts.append("<thead><tr>")
    html_parts.append("<th>Structure</th>")

    for h in headers:
        html_parts.append(f"<th>{html.escape(str(h))}</th>")

    html_parts.append("</tr></thead><tbody>")

    for idx, row in show_df.iterrows():
        canvas_id = f"mol_canvas_{idx}"
        img_html = smiles_to_html(row.get("Smiles"), canvas_id=canvas_id)

        html_parts.append("<tr>")
        html_parts.append(f'<td class="mol-cell">{img_html}</td>')

        for h in headers:
            value = row.get(h, "")
            if pd.isna(value):
                value = ""
            html_parts.append(f"<td>{html.escape(str(value))}</td>")

        html_parts.append("</tr>")

    html_parts.append("</tbody></table>")
    return "".join(html_parts)


def build_single_molecule_html(smiles, width: int = 500, height: int = 350) -> str:
    canvas_id = "single_molecule_canvas"
    smiles_clean = sanitize_smiles_for_browser(smiles)

    if smiles_clean is None:
        return """
        <div style="padding:1rem; color:#666; font-size:14px;">
            No valid molecular structure could be generated for this row.
        </div>
        """

    smiles_str = json.dumps(smiles_clean)

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

            SmilesDrawer.parse(
                smiles,
                function(tree) {{
                    const drawer = new SmilesDrawer.Drawer({
                        width: {width},
                        height: {height},
                        padding: 10,
                        experimental: true
                        experimental: true
                    });
                    drawer.draw(tree, targetId, "light", false);
                }},
                function() {{
                    const el = document.getElementById(targetId);
                    if (el && el.parentElement) {{
                        el.parentElement.innerHTML = '<div style="color:#666;">Invalid SMILES</div>';
                    }}
                }}
            );
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
    result_df["Smiles_for_browser"] = result_df["Smiles"].map(sanitize_smiles_for_browser)
    st.dataframe(
        result_df[["Compound_Name", "Smiles", "Smiles_for_browser", "INCHI", "Structure_Status"]],
        use_container_width=True,
    )
    debug_df = result_df[["Compound_Name", "Smiles", "INCHI", "Structure_Status"]].copy()
    st.subheader("SMILES debug table")
    st.dataframe(debug_df, use_container_width=True)

    st.subheader("Processed annotation table")
    st.dataframe(result_df, use_container_width=True)

    valid_count = (result_df["Structure_Status"] == "SMILES available").sum()
    invalid_count = (result_df["Structure_Status"] == "Invalid / missing").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(result_df))
    c2.metric("Rows with SMILES", int(valid_count))
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



