"""
download_sample.py
------------------
Copia uma amostra do dataset CBIS-DDSM ja extraido para a pasta do projeto.

Estrutura esperada em SOURCE_DIR:
  SOURCE_DIR/
    csv/   -> mass_case_description_train_set.csv, ...
    jpeg/  -> <UID>/1-263.jpg, <UID>/2-241.jpg, ...

Como usar:
  1. Configure SOURCE_DIR abaixo
  2. Execute: python copy_sample.py
"""

import os
import shutil
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# CONFIGURACAO
# ---------------------------------------------------------------------------

SOURCE_DIR       = r"F:\archive"
OUTPUT_DIR       = "./data/cnn-data"
SAMPLE_PER_CLASS = 50
RANDOM_SEED      = 42

# ---------------------------------------------------------------------------

CSV_SUBDIR = os.path.join(OUTPUT_DIR, "csv")
CSV_NAMES  = [
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv",
]


def find_subdir(root, name):
    direct = os.path.join(root, name)
    if os.path.isdir(direct):
        return direct
    for entry in os.listdir(root):
        c = os.path.join(root, entry, name)
        if os.path.isdir(c):
            return c
    return None


def copy_csvs(csv_dir):
    os.makedirs(CSV_SUBDIR, exist_ok=True)
    for name in CSV_NAMES:
        dest = os.path.join(CSV_SUBDIR, name)
        if not os.path.exists(dest):
            shutil.copy2(os.path.join(csv_dir, name), dest)
            print(f"  [ok] {name} copiado")
        else:
            print(f"  [ok] {name} ja existe")


def load_and_sample(csv_dir):
    def read(name):
        df = pd.read_csv(os.path.join(csv_dir, name))
        df.columns = df.columns.str.strip()
        df["label"] = df["pathology"].apply(
            lambda x: 1 if str(x).strip().upper() == "MALIGNANT" else 0
        )
        crop_col = next(c for c in df.columns if "cropped" in c.lower() and "file" in c.lower())
        df["crop_col"] = df[crop_col].astype(str).str.strip()
        return df

    def sample(df, n):
        n0 = min(n, (df["label"] == 0).sum())
        n1 = min(n, (df["label"] == 1).sum())
        return pd.concat([
            df[df["label"] == 0].sample(n0, random_state=RANDOM_SEED),
            df[df["label"] == 1].sample(n1, random_state=RANDOM_SEED),
        ]).reset_index(drop=True)

    train = sample(read(CSV_NAMES[0]), SAMPLE_PER_CLASS)
    test  = sample(read(CSV_NAMES[1]), SAMPLE_PER_CLASS)
    return train, test


def extract_uids(csv_path):
    """
    CSV path ex: Mass-Training_P_00001_LEFT_CC_1/1.3.6.../1.3.6.../000000.dcm
    Retorna lista de UIDs (segmentos que parecem UIDs DICOM: comecam com 1.)
    """
    parts = str(csv_path).replace("\\", "/").strip().split("/")
    return [p for p in parts if p.startswith("1.") and len(p) > 20]


def build_uid_index(jpeg_dir):
    """
    Mapeia: uid_folder_name -> primeiro jpg encontrado dentro dela
    Suporta estrutura plana (jpeg/<UID>/file.jpg)
    e estrutura aninhada (jpeg/<UID1>/<UID2>/file.jpg)
    """
    print("  Indexando pastas UID em jpeg/ ...")
    index = {}  # uid -> primeiro jpg path

    for uid_entry in os.listdir(jpeg_dir):
        uid_path = os.path.join(jpeg_dir, uid_entry)
        if not os.path.isdir(uid_path):
            continue

        # Coleta todos os jpgs nesta pasta e subpastas
        jpgs = []
        for root, _, files in os.walk(uid_path):
            for f in files:
                if f.lower().endswith(".jpg"):
                    jpgs.append(os.path.join(root, f))

        if jpgs:
            index[uid_entry] = jpgs

        # Tambem indexa subpastas com UID como chave
        for sub in os.listdir(uid_path):
            sub_path = os.path.join(uid_path, sub)
            if os.path.isdir(sub_path) and sub.startswith("1.") and len(sub) > 20:
                sub_jpgs = [
                    os.path.join(sub_path, f)
                    for f in os.listdir(sub_path)
                    if f.lower().endswith(".jpg")
                ]
                if sub_jpgs:
                    index[sub] = sub_jpgs

    print(f"  {len(index):,} pastas UID indexadas")
    return index


def find_jpg_for_row(csv_path, uid_index):
    """Tenta achar um JPG no index usando os UIDs presentes no path do CSV."""
    for uid in extract_uids(csv_path):
        if uid in uid_index:
            return uid_index[uid][0], uid   # (path_absoluto, uid_da_pasta)
    return None, None


def copy_images(train_df, test_df, uid_index, jpeg_dest_root):
    rows = []
    for df, split in [(train_df, "train"), (test_df, "test")]:
        for _, row in df.iterrows():
            rows.append((row["crop_col"], row["label"], split))

    total     = len(rows)
    copied    = 0
    skipped   = 0
    not_found = 0

    # Guarda mapeamento para atualizar CSVs depois
    path_map = {}   # csv_path -> dest_relative

    for i, (csv_path, label, split) in enumerate(rows, 1):
        if csv_path in ("nan", "", None):
            not_found += 1
            continue

        src, uid = find_jpg_for_row(csv_path, uid_index)
        if src is None:
            not_found += 1
        else:
            fname = os.path.basename(src)
            rel   = os.path.join(uid, fname)   # ex: 1.3.6.../1-263.jpg
            dest  = os.path.join(jpeg_dest_root, rel)

            path_map[csv_path] = os.path.join("jpeg", rel).replace("\\", "/")

            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                skipped += 1
            else:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(src, dest)
                copied += 1

        if i % 20 == 0 or i == total:
            print(
                f"  {i}/{total} | copiadas={copied} existiam={skipped} nao_encontradas={not_found}",
                end="\r",
            )

    print()
    return copied, skipped, not_found, path_map


def save_sample_csvs(train_df, test_df, path_map):
    """
    Salva versoes filtradas dos CSVs com apenas as linhas da amostra
    e uma coluna extra 'jpeg_path' apontando para os arquivos copiados.
    """
    for df, name in [(train_df, "train_sample.csv"), (test_df, "test_sample.csv")]:
        df = df.copy()
        df["jpeg_path"] = df["crop_col"].map(path_map)
        out = os.path.join(CSV_SUBDIR, name)
        df.to_csv(out, index=False)
        print(f"  Salvo: {out}")


def main():
    print("=" * 60)
    print("Amostragem CBIS-DDSM")
    print(f"Fonte   : {SOURCE_DIR}")
    print(f"Destino : {os.path.abspath(OUTPUT_DIR)}")
    print(f"Amostra : {SAMPLE_PER_CLASS} imagens por classe")
    print("=" * 60)

    if not os.path.isdir(SOURCE_DIR):
        print(f"ERRO: pasta nao encontrada: {SOURCE_DIR}")
        sys.exit(1)

    csv_dir  = find_subdir(SOURCE_DIR, "csv")
    jpeg_dir = find_subdir(SOURCE_DIR, "jpeg")

    if not csv_dir:
        print(f"ERRO: subpasta 'csv' nao encontrada em {SOURCE_DIR}")
        sys.exit(1)
    if not jpeg_dir:
        print(f"ERRO: subpasta 'jpeg' nao encontrada em {SOURCE_DIR}")
        sys.exit(1)

    print(f"\n  csv : {csv_dir}")
    print(f"  jpeg: {jpeg_dir}")

    print("\n[1/5] Copiando CSVs ...")
    copy_csvs(csv_dir)

    print("\n[2/5] Amostrando metadados ...")
    train_s, test_s = load_and_sample(csv_dir)
    print(f"  Treino : {len(train_s)} ({(train_s['label']==0).sum()} benignas + {(train_s['label']==1).sum()} malignas)")
    print(f"  Teste  : {len(test_s)} ({(test_s['label']==0).sum()} benignas + {(test_s['label']==1).sum()} malignas)")

    print("\n[3/5] Indexando imagens ...")
    uid_index = build_uid_index(jpeg_dir)

    # Diagnostico: exemplo de UID e arquivo
    sample_uid = next(iter(uid_index))
    print(f"  Exemplo: {sample_uid[:50]}... -> {os.path.basename(uid_index[sample_uid][0])}")
    sample_csv = train_s["crop_col"].iloc[0]
    uids_found = extract_uids(sample_csv)
    print(f"  Exemplo CSV path UIDs: {[u[:30]+'...' for u in uids_found]}")
    match = any(u in uid_index for u in uids_found)
    print(f"  Match de teste: {'SIM' if match else 'NAO — UIDs do CSV nao coincidem com pastas jpeg'}")

    print("\n[4/5] Copiando imagens ...")
    jpeg_dest = os.path.join(OUTPUT_DIR, "jpeg")
    os.makedirs(jpeg_dest, exist_ok=True)
    copied, skipped, not_found, path_map = copy_images(train_s, test_s, uid_index, jpeg_dest)

    print("\n[5/5] Salvando CSVs de amostra ...")
    save_sample_csvs(train_s, test_s, path_map)

    print("\n" + "=" * 60)
    print("Concluido!")
    print(f"  Copiadas       : {copied}")
    print(f"  Ja existiam    : {skipped}")
    print(f"  Nao encontradas: {not_found}")

    if not_found == len(train_s) + len(test_s):
        print("\n  ATENCAO: nenhuma imagem foi encontrada.")
        print("  O diagnostico acima ('Match de teste') indica o problema.")
        print("  Verifique se o CSV baixado corresponde ao dataset JPEG extraido.")

    print(f"\nDataset em: {os.path.abspath(OUTPUT_DIR)}")
    print("Execute: diagnostico_imagem_cnn.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()
