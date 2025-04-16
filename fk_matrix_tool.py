import argparse
import os

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA

def load_fk_data(file_path):
    df = pd.read_excel(file_path)
    df.columns = [col.strip().lower() for col in df.columns]
    return df


def build_matrix(df):
    tables = sorted(set(df['from_table']) | set(df['to_table']))
    matrix = pd.DataFrame('', index=tables, columns=tables)

    for _, row in df.iterrows():
        from_table = row['from_table']
        to_table = row['to_table']
        matrix.at[from_table, to_table] = 'Y'

    return matrix


def reorder_matrix(matrix, algorithm="none", min_fks=1):
    if algorithm == "none":
        return matrix.sort_index().sort_index(axis=1)

    # Convert matrix to binary numeric form
    binary_matrix = matrix.replace({'Y': 1, '': 0}).astype(int)

    # Calculate total FK activity per table
    fk_activity = binary_matrix.sum(axis=1) + binary_matrix.sum(axis=0)
    active_tables = fk_activity[fk_activity >= min_fks].index.tolist()

    if len(active_tables) < 2:
        print("⚠️ Not enough connected tables for clustering. Falling back to alphabetical.")
        return matrix.sort_index().sort_index(axis=1)

    sub_matrix = binary_matrix.loc[active_tables, active_tables]

    try:
        if algorithm == "cosine":
            distance_matrix = pdist(sub_matrix.values, metric='cosine')
            if not np.isfinite(distance_matrix).all():
                raise ValueError(
                    "Non-finite values detected in distance matrix. Try a different algorithm or increase --min-fks.")
            linkage_matrix = linkage(distance_matrix, method='average')
            order = leaves_list(linkage_matrix)

        elif algorithm == "hierarchical":
            distance_matrix = pdist(sub_matrix.values, metric='jaccard')
            linkage_matrix = linkage(distance_matrix, method='average')
            order = leaves_list(linkage_matrix)

        elif algorithm == "pca":
            pca = PCA(n_components=2)
            coords = pca.fit_transform(sub_matrix.values)
            order = np.argsort(coords[:, 0])  # Sort by X-axis

        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        clustered_tables = sub_matrix.index[order].tolist()

    except Exception as e:
        print(f"⚠️ Clustering failed: {e}")
        print("🔁 Falling back to alphabetical.")
        return matrix.sort_index().sort_index(axis=1)

    # Append inactive tables after clustered ones
    inactive_tables = sorted(set(matrix.index) - set(clustered_tables))
    all_tables = clustered_tables + inactive_tables

    return matrix.loc[all_tables, all_tables]

def format_excel(file_path):
    wb = load_workbook(filename=file_path)
    ws = wb.active

    # Rotate column headers (skip top-left cell)
    for col in ws.iter_cols(min_row=1, max_row=1, min_col=2):
        for cell in col:
            cell.alignment = Alignment(textRotation=90)

    # Right-align first column
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row, min_col=1, max_col=1):
        for cell in row:
            cell.alignment = Alignment(horizontal='right')

    # Freeze panes at B2
    ws.freeze_panes = "B2"

    # Set first column (A) width based on actual content
    max_length = 0
    for cell in ws['A']:
        if cell.value:
            max_length = max(max_length, len(str(cell.value)))
    ws.column_dimensions['A'].width = max_length + 1

    # Set fixed width for all other columns (rotated headers)
    for col_cells in ws.iter_cols(min_row=1, max_row=1, min_col=2):
        col_letter = col_cells[0].column_letter
        ws.column_dimensions[col_letter].width = 2.67

    wb.save(filename=file_path)
    print(f"🎨 Excel formatted: {file_path}")


def main(input_file, algorithm, min_fks):
    print(f"🔍 Loading foreign keys from: {input_file}")
    df = load_fk_data(input_file)

    print("🧱 Building full FK matrix...")
    matrix = build_matrix(df)

    print(f"🔁 Applying algorithm: {algorithm}")
    reordered_matrix = reorder_matrix(matrix, algorithm)

    # Construct output file name from input
    input_base = os.path.splitext(os.path.basename(input_file))[0]
    output_file = f"{input_base}.{algorithm}.xlsx"

    print(f"💾 Saving output to: {output_file}")
    reordered_matrix.to_excel(output_file)

    format_excel(output_file)

    print("✅ Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foreign Key Matrix Reorder Tool")
    parser.add_argument("--input", type=str, default="foreign_keys.xlsx", help="Input Excel file with FK relationships")
    parser.add_argument("--algorithm", type=str, choices=["none", "hierarchical", "cosine", "pca"], default="none", help="Reordering algorithm")
    parser.add_argument("--min-fks", type=int, default=1, help="Minimum total FK activity (in+out) to include in clustering")
    args = parser.parse_args()

    main(args.input, args.algorithm, args.min_fks)