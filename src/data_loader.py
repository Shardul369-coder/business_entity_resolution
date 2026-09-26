import pandas as pd
from pathlib import Path

Required_columns = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

def load_tsv(file_path):
    '''
    Load a TSV file and validate its required columns.

    Parameters:
    file_path : str or Path
        Path to the TSV file.

    Returns:
    pandas.DataFrame
        Loaded dataset.
    '''
    
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found {file_path}"
        )

    if file_path.suffix.lower() != ".tsv":
        raise ValueError(
            f"Expected a .tsv file, got: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    missing_columns = [column for column in Required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            f"{file_path.name} is missing required columns: "
            f"{missing_columns}"
        )

    return df

def load_train_data(data_dir):
    """
    Load all training datasets.

    Expected structure:

    data_dir/
        train_source1.tsv
        train_source2.tsv
        train_source3.tsv
        train_ground_truth.tsv
    """

    data_dir = Path(data_dir)

    source1 = load_tsv(
        data_dir / "train_source1.tsv"
    )

    source2 = load_tsv(
        data_dir / "train_source2.tsv"
    )

    source3 = load_tsv(
        data_dir / "train_source3.tsv"
    )

    ground_truth_path = data_dir / "train_ground_truth.tsv"

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {ground_truth_path}"
        )

    ground_truth = pd.read_csv(
        ground_truth_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    required_ground_truth_columns = [
        "source1_entity_id",
        "matched_entity_ids",
    ]

    missing_columns = [
        column
        for column in required_ground_truth_columns
        if column not in ground_truth.columns
    ]

    if missing_columns:
        raise ValueError(
            "Ground truth is missing required columns: "
            f"{missing_columns}"
        )

    return source1, source2, source3, ground_truth

def load_test_data(data_dir):
    """
    Load all test datasets.

    Expected structure:

    data_dir/
        test_source1.tsv
        test_source2.tsv
        test_source3.tsv
    """

    data_dir = Path(data_dir)

    source1 = load_tsv(
        data_dir / "test_source1.tsv"
    )

    source2 = load_tsv(
        data_dir / "test_source2.tsv"
    )

    source3 = load_tsv(
        data_dir / "test_source3.tsv"
    )

    return source1, source2, source3