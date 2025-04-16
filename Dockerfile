FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    gcc build-essential libxml2-dev libz-dev \
    libglpk-dev python3-dev curl git pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY fk_matrix_tool.py /app/fk_matrix_tool.py

RUN pip install --no-cache-dir \
    pandas numpy openpyxl scipy matplotlib \
    plotly networkx pyvis python-louvain scikit-learn

RUN pip install --no-cache-dir leidenalg

ENTRYPOINT ["python", "fk_matrix_tool.py"]