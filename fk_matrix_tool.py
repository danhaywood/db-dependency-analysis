import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from openpyxl import load_workbook
from openpyxl.styles import Alignment
import argparse


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


def reorder_matrix(matrix, algorithm="none"):
    if algorithm == "none":
        ordered_labels = sorted(matrix.index)
    else:
        binary_matrix = matrix.replace({'Y': 1, '': 0}).astype(int).values

        if algorithm == "cosine":
            distance_matrix = pdist(binary_matrix, metric='cosine')
        elif algorithm == "hierarchical":
            distance_matrix = pdist(binary_matrix, metric='jaccard')
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        linkage_matrix = linkage(distance_matrix, method='average')
        order = leaves_list(linkage_matrix)
        ordered_labels = matrix.index[order]

    return matrix.loc[ordered_labels, ordered_labels]


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

    # Autofit each column width
    for col_cells in ws.iter_cols(min_row=1, max_row=ws.max_row):
        max_length = 0
        for cell in col_cells:
            try:
                if cell.value:
                    max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        col_letter = col_cells[0].column_letter
        ws.column_dimensions[col_letter].width = max_length + 1

    wb.save(filename=file_path)
    print(f"🎨 Excel formatted: {file_path}")


def main(input_file, algorithm):
    print(f"🔍 Loading foreign keys from: {input_file}")
    df = load_fk_data(input_file)

    print("🧱 Building full FK matrix...")
    matrix = build_matrix(df)

    print(f"🔁 Applying algorithm: {algorithm}")
    reordered_matrix = reorder_matrix(matrix, algorithm)

    output_file = f"fk_matrix_clustered_{algorithm}.xlsx"
    print(f"💾 Saving output to: {output_file}")
    reordered_matrix.to_excel(output_file)

    format_excel(output_file)

    print("✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foreign Key Matrix Reorder Tool")
    parser.add_argument("--input", type=str, default="foreign_keys.xlsx", help="Input Excel file with FK relationships")
    parser.add_argument("--algorithm", type=str, choices=["none", "hierarchical", "cosine"], default="none", help="Reordering algorithm")
    args = parser.parse_args()

    main(args.input, args.algorithm)