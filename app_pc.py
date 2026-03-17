import base64
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw


st.set_page_config(page_title="MassQL Structure Viewer", layout="wide")


# -----------------------------
# Logos (optional)
# -----------------------------
STATIC_DIR = Path(__file__).parent / "static"
for logo_name in ["LAABio.png"]:
    p = STATIC_DIR / logo_name
    try:
        from PIL import Image
        st.sidebar.image(Image.open(p), use_container_width=True)
    except Exception:
        pass

st.sidebar.markdown("by Ricardo Moreira Borges (IPPN-UFRJ)")


# ---------------------------------------------------------
# Column aliases
# ---------------------------------------------------------
CANONICAL_NAMES = {
    "scan": ["#Scan#", "Scan", "scan", "scan_number"],
    "query_validation": ["query_validation", "Query_Validation", "validation"],
    "compound_name": ["Compound_Name", "compound_name", "Name", "name"],
    "smiles": ["Smiles", "SMILES", "smiles"],
    "inchi": ["INCHI", "InChI", "inchi"],
    "inchikey": ["InChIKey", "INCHIKEY", "inchikey"],
}


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def clean_colname(col) -> str:
    return str(col).strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_colname(c) for c in out.columns]
    return out


def find_first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    existing = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in existing:
            return existing[key]
    return None


def guess_smiles_column(df: pd.DataFrame) -> str | None:
    # 1) canonical aliases first
    found = find_first_existing_column(df, CANONICAL_NAMES["smiles"])
    if found is not None:
        return found

    # 2) anything containing 'smile'
    for c in df.columns:
        c_low = str(c).strip().lower()
        if "smile" in c_low:
            return c

    return None


def guess_inchi_column(df: pd.DataFrame) -> str | None:
    found = find_first_existing_column(df, CANONICAL_NAMES["inchi"])
    if found is not None:
        return found

    for c in df.columns:
        c_low = str(c).strip().lower()
        if c_low == "inchi" or "inchi" in c_low:
            return c

    return None


def guess_label_column(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> str | None:
    if exclude_cols is None:
        exclude_cols = []

    exclude_set = {str(c) for c in exclude_cols if c is not None}

    priority = [
        "Compound_Name", "compound_name", "Name", "name",
        "Title", "title",
        "Annotation", "annotation",
        "query_validation", "Query_Validation", "validation",
        "#Scan#", "Scan", "scan", "scan_number",
        "InChIKey", "inchikey", "INCHIKEY",
    ]

    for p in priority:
        for c in df.columns:
            if c in exclude_set:
                continue
            if str(c).strip().lower() == str(p).strip().lower():
                return c

    # fallback: first non-excluded column
    for c in df.columns:
        if c not in exclude_set:
            return c

    return None


@st.cache_data(show_spinner=False)
def load_table(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()

    # try common formats with fallbacks
    if name.endswith(".csv"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file)
        except Exception:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, sep=";")
            except Exception:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, sep="\t")

    elif name.endswith(".tsv") or name.endswith(".txt"):
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, sep="\t")
        except Exception:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, sep=";")
            except Exception:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file)

    # final generic fallback
    uploaded_file.seek(0)
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


def structure_status(smiles, inchi):
    return "OK" if mol_from_identifiers(smiles, inchi) is not None else "Invalid / missing"


@st.cache_data(show_spinner=False)
def prepare_massql_result_table(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_columns(df)

    scan_col = find_first_existing_column(df, CANONICAL_NAMES["scan"])
    query_col = find_first_existing_column(df, CANONICAL_NAMES["query_validation"])
    name_col = find_first_existing_column(df, CANONICAL_NAMES["compound_name"])
    smiles_col = guess_smiles_column(df)
    inchi_col = guess_inchi_column(df)
    inchikey_col = find_first_existing_column(df, CANONICAL_NAMES["inchikey"])

    result = pd.DataFrame()

    if scan_col:
        result["#Scan#"] = df[scan_col]
    if query_col:
        result["query_validation"] = df[query_col]
    if name_col:
        result["Compound_Name"] = df[name_col]

    result["Smiles"] = df[smiles_col] if smiles_col else None
    result["INCHI"] = df[inchi_col] if inchi_col else None

    if inchikey_col:
        result["InChIKey"] = df[inchikey_col]

    result["Structure_Status"] = result.apply(
        lambda row: structure_status(row.get("Smiles"), row.get("INCHI")),
        axis=1,
    )

    return result


@st.cache_data(show_spinner=False)
def prepare_generic_result_table(
    df: pd.DataFrame,
    smiles_col: str | None,
    selected_info_col: str | None,
    use_inchi_fallback: bool,
    inchi_col: str | None,
) -> pd.DataFrame:
    df = normalize_columns(df)

    result = pd.DataFrame()

    if selected_info_col and selected_info_col in df.columns:
        result["Selected_Column"] = df[selected_info_col]
    else:
        result["Selected_Column"] = [f"Row {i + 1}" for i in range(len(df))]

    result["Smiles"] = df[smiles_col] if smiles_col and smiles_col in df.columns else None

    if use_inchi_fallback and inchi_col and inchi_col in df.columns:
        result["INCHI"] = df[inchi_col]
    else:
        result["INCHI"] = None

    result["Structure_Status"] = result.apply(
        lambda row: structure_status(row.get("Smiles"), row.get("INCHI")),
        axis=1,
    )

    return result


@st.cache_data(show_spinner=False)
def build_html_table(
    result_df: pd.DataFrame,
    max_rows: int = 200,
    info_col_name: str = "Selected_Column",
) -> str:
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
        display_name = info_col_name if h == "Selected_Column" else h
        html.append(f"<th>{display_name}</th>")
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


def make_label_series(result_df: pd.DataFrame, preferred_col: str) -> pd.Series:
    if preferred_col in result_df.columns:
        return result_df[preferred_col].fillna("Unnamed").astype(str)
    return pd.Series([f"Row {i + 1}" for i in range(len(result_df))], index=result_df.index)


# ---------------------------------------------------------
# App header
# ---------------------------------------------------------
st.title("MassQL Structure Viewer")
st.write(
    "Upload a table and inspect the recognized molecules as 2D chemical structures."
)

generic_mode = st.sidebar.checkbox(
    "Generic CSV mode",
    value=False,
    help="When checked, the app accepts any table and lets you choose the SMILES column and another column to display."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload table",
    type=["csv", "tsv", "txt"],
    accept_multiple_files=False,
)

if generic_mode:
    with st.expander("Generic CSV mode", expanded=False):
        st.markdown(
            """
            In this mode, the app will:

            - accept any CSV / TSV / TXT
            - let you choose the **SMILES column**
            - let you choose **another column** to display beside the structures
            - optionally use **InChI as fallback**
            """
        )
else:
    with st.expander("Expected MassQL columns", expanded=False):
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

    with st.sidebar.expander("Expected columns", expanded=False):
        st.markdown(
            """
            Upload the CSV file exported from **"Table With Library Matches Only"** after running MassQL at  
            https://massqlpostmn.gnps2.org/

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


# ---------------------------------------------------------
# Main logic
# ---------------------------------------------------------
if uploaded_file is None:
    st.info("Upload a table to begin.")
    st.stop()

try:
    raw_df = load_table(uploaded_file)
except Exception as e:
    st.error(f"Could not read the file: {e}")
    st.stop()

raw_df = normalize_columns(raw_df)

with st.expander("Raw uploaded table"):
    st.dataframe(raw_df, use_container_width=True)

if len(raw_df.columns) == 0:
    st.error("The uploaded table has no columns.")
    st.stop()


# ---------------------------------------------------------
# Generic CSV mode
# ---------------------------------------------------------
if generic_mode:
    st.subheader("Generic CSV configuration")

    auto_smiles_col = guess_smiles_column(raw_df)
    auto_inchi_col = guess_inchi_column(raw_df)

    all_cols = list(raw_df.columns)

    col1, col2 = st.columns(2)

    with col1:
        if auto_smiles_col in all_cols:
            smiles_index = all_cols.index(auto_smiles_col)
        else:
            smiles_index = 0

        smiles_col = st.selectbox(
            "Select the column containing SMILES",
            options=all_cols,
            index=smiles_index,
            help="Choose the column that contains molecular structures in SMILES format.",
        )

    with col2:
        default_label_col = guess_label_column(raw_df, exclude_cols=[smiles_col])
        if default_label_col in all_cols:
            label_index = all_cols.index(default_label_col)
        else:
            label_index = 0

        selected_info_col = st.selectbox(
            "Select another column to display",
            options=all_cols,
            index=label_index,
            help="This column will be shown beside the structures in the molecule table.",
        )

    use_inchi_fallback = st.checkbox(
        "Use InChI fallback when SMILES is invalid or missing",
        value=auto_inchi_col is not None,
    )

    inchi_col = None
    if use_inchi_fallback:
        inchi_options = ["<None>"] + all_cols
        default_inchi_idx = 0
        if auto_inchi_col in all_cols:
            default_inchi_idx = inchi_options.index(auto_inchi_col)

        selected_inchi = st.selectbox(
            "Select the InChI column (optional)",
            options=inchi_options,
            index=default_inchi_idx,
        )
        inchi_col = None if selected_inchi == "<None>" else selected_inchi

    result_df = prepare_generic_result_table(
        raw_df,
        smiles_col=smiles_col,
        selected_info_col=selected_info_col,
        use_inchi_fallback=use_inchi_fallback,
        inchi_col=inchi_col,
    )

    st.subheader("Processed table")
    st.dataframe(result_df, use_container_width=True)

    valid_count = int((result_df["Structure_Status"] == "OK").sum())
    invalid_count = int((result_df["Structure_Status"] != "OK").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(result_df))
    c2.metric("Valid structures", valid_count)
    c3.metric("Invalid / missing", invalid_count)

    st.subheader("Structure table")

    show_only_valid = st.checkbox(
        "Show only valid structures",
        value=True,
    )

    df_to_show = result_df.copy()
    if show_only_valid:
        df_to_show = df_to_show[df_to_show["Structure_Status"] == "OK"]

    max_rows = st.slider(
        "Maximum rows to render as molecule table",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        key="generic_max_rows_structure_table",
    )

    html_table = build_html_table(
        df_to_show,
        max_rows=max_rows,
        info_col_name=selected_info_col,
    )
    st.components.v1.html(html_table, height=700, scrolling=True)

    st.subheader("Single-compound browser")
    label_series = make_label_series(result_df, "Selected_Column")

    selected_idx = st.selectbox(
        "Select one row",
        options=list(range(len(result_df))),
        format_func=lambda i: f"{i + 1} — {label_series.iloc[i]}",
        key="generic_single_compound_select",
    )

    selected_row = result_df.iloc[selected_idx]
    png = mol_to_png_bytes(
        selected_row.get("Smiles"),
        selected_row.get("INCHI"),
        size=(500, 350),
    )

    col_a, col_b = st.columns([1, 1.3])

    with col_a:
        if png is not None:
            st.image(
                png,
                caption=str(label_series.iloc[selected_idx]),
                use_container_width=True,
            )
        else:
            st.warning("No valid molecular structure could be generated for this row.")

    with col_b:
        details = selected_row.to_frame(name="Value")
        st.dataframe(details, use_container_width=True)

    csv_bytes = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download processed table as CSV",
        data=csv_bytes,
        file_name="generic_structure_viewer_processed.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# MassQL mode
# ---------------------------------------------------------
else:
    result_df = prepare_massql_result_table(raw_df)

    st.subheader("Processed annotation table")
    st.dataframe(result_df, use_container_width=True)

    valid_count = int((result_df["Structure_Status"] == "OK").sum())
    invalid_count = int((result_df["Structure_Status"] != "OK").sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", len(result_df))
    c2.metric("Valid structures", valid_count)
    c3.metric("Invalid / missing", invalid_count)

    st.subheader("Structure table")

    show_only_hits = st.checkbox(
        "Show only features that passed at least one query",
        value=True,
    )

    df_to_show = result_df.copy()

    if show_only_hits and "query_validation" in df_to_show.columns:
        df_to_show = df_to_show[
            df_to_show["query_validation"] != "Did not pass any selected query"
        ]

    max_rows = st.slider(
        "Maximum rows to render as molecule table",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
        key="massql_max_rows_structure_table",
    )

    html_table = build_html_table(
        df_to_show,
        max_rows=max_rows,
        info_col_name="Compound_Name",
    )
    st.components.v1.html(html_table, height=700, scrolling=True)

    st.subheader("Single-compound browser")

    if "Compound_Name" in result_df.columns:
        label_series = result_df["Compound_Name"].fillna("Unnamed").astype(str)
    else:
        label_series = pd.Series(
            [f"Row {i + 1}" for i in range(len(result_df))],
            index=result_df.index,
        )

    selected_idx = st.selectbox(
        "Select one row",
        options=list(range(len(result_df))),
        format_func=lambda i: f"{i + 1} — {label_series.iloc[i]}",
        key="massql_single_compound_select",
    )

    selected_row = result_df.iloc[selected_idx]
    png = mol_to_png_bytes(
        selected_row.get("Smiles"),
        selected_row.get("INCHI"),
        size=(500, 350),
    )

    col_a, col_b = st.columns([1, 1.3])

    with col_a:
        if png is not None:
            st.image(
                png,
                caption=str(label_series.iloc[selected_idx]),
                use_container_width=True,
            )
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
