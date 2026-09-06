"""
Data & Exploratory Data Analysis (EDA) Route Handlers.
Endpoints:
- GET  /api/v1/data/summary: Full dataset descriptive statistics, missingness audit, and Hopkins statistic
- GET  /api/v1/data/correlations: Pearson and Spearman correlation matrices
- GET  /api/v1/data/distributions: Histogram bins and distribution stats for all numeric features
- GET  /api/v1/data/sample: Preview rows from active dataset
- POST /api/v1/data/upload: Upload CSV dataset or update active in-memory dataset
"""

import io
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, File, UploadFile, Query, HTTPException, status
import numpy as np
import pandas as pd

from ..schemas import (
    DatasetSummaryResponse,
    CorrelationResponse,
    DistributionsResponse,
    FeatureDistribution,
    HistogramBin,
    DataSampleResponse,
    DataUploadResponse,
    DescriptiveStat,
    MissingSummary,
    MissingColumnInfo,
)
from ..state import state
from crisp_dm.data_understanding import (
    compute_data_understanding_summary,
    compute_hopkins_statistic,
    NUMERIC_FEATURE_COLUMNS,
)

router = APIRouter(prefix="/data", tags=["Dataset & EDA"])


@router.get(
    "/summary",
    response_model=DatasetSummaryResponse,
    summary="Get comprehensive dataset summary statistics and Hopkins tendency",
)
def get_dataset_summary() -> DatasetSummaryResponse:
    """
    Computes and returns descriptive statistics, missing value audit,
    skewness ranking, Pearson/Spearman correlation matrices, and Hopkins clustering tendency.
    """
    df = state.get_dataset()
    summary = compute_data_understanding_summary(df)

    # Convert descriptive_statistics dict to schema format
    desc_stats: Dict[str, DescriptiveStat] = {}
    for col, s in summary["descriptive_statistics"].items():
        desc_stats[col] = DescriptiveStat(**s)

    # Convert missing summary
    missing_cols: Dict[str, MissingColumnInfo] = {}
    for col, info in summary["missing_summary"]["columns_with_missing"].items():
        missing_cols[col] = MissingColumnInfo(**info)

    missing_sum = MissingSummary(
        total_missing_cells=summary["missing_summary"]["total_missing_cells"],
        columns_with_missing=missing_cols,
    )

    return DatasetSummaryResponse(
        n_samples=summary["n_samples"],
        n_columns=summary["n_columns"],
        columns=summary["columns"],
        numeric_columns=summary["numeric_columns"],
        descriptive_statistics=desc_stats,
        missing_summary=missing_sum,
        skewness_ranking=[[feat, skew] for feat, skew in summary["skewness_ranking"]],
        correlation_pearson=summary["correlation_pearson"],
        correlation_spearman=summary["correlation_spearman"],
        hopkins_statistic=summary["hopkins_statistic"],
    )


@router.get(
    "/correlations",
    response_model=CorrelationResponse,
    summary="Get Pearson and Spearman correlation matrices",
)
def get_correlations() -> CorrelationResponse:
    """
    Returns pairwise Pearson and Spearman correlation matrices for numeric features.
    """
    df = state.get_dataset()
    numeric_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in df.columns]
    num_df = df[numeric_cols].astype(float)

    pearson = num_df.corr(method="pearson").fillna(0.0).round(4).to_dict()
    spearman = num_df.corr(method="spearman").fillna(0.0).round(4).to_dict()

    return CorrelationResponse(
        columns=numeric_cols,
        pearson=pearson,
        spearman=spearman,
    )


@router.get(
    "/distributions",
    response_model=DistributionsResponse,
    summary="Get histogram bins and distribution statistics per feature",
)
def get_distributions(
    n_bins: int = Query(default=20, ge=5, le=100, description="Number of histogram bins")
) -> DistributionsResponse:
    """
    Calculates histogram frequency bins, densities, and summary statistics
    for every numeric feature in the active dataset.
    """
    df = state.get_dataset()
    numeric_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in df.columns]

    features_dict: Dict[str, FeatureDistribution] = {}

    for col in numeric_cols:
        series = df[col].dropna().astype(float)
        if len(series) == 0:
            continue

        counts, bin_edges = np.histogram(series, bins=n_bins)
        total = float(len(series))
        bin_width = bin_edges[1] - bin_edges[0] if len(bin_edges) > 1 else 1.0

        bins: List[HistogramBin] = []
        for i in range(len(counts)):
            density = float(counts[i] / (total * bin_width)) if bin_width > 0 else 0.0
            bins.append(
                HistogramBin(
                    bin_start=round(float(bin_edges[i]), 4),
                    bin_end=round(float(bin_edges[i + 1]), 4),
                    count=int(counts[i]),
                    density=round(density, 6),
                )
            )

        skew_val = float(series.skew()) if len(series) > 2 else 0.0

        features_dict[col] = FeatureDistribution(
            feature=col,
            count=int(len(series)),
            mean=round(float(series.mean()), 4),
            std=round(float(series.std()), 4),
            median=round(float(series.median()), 4),
            min=round(float(series.min()), 4),
            max=round(float(series.max()), 4),
            skewness=round(skew_val, 4),
            bins=bins,
        )

    return DistributionsResponse(features=features_dict)


@router.get(
    "/sample",
    response_model=DataSampleResponse,
    summary="Get raw record preview sample from active dataset",
)
def get_data_sample(
    limit: int = Query(default=100, ge=1, le=1000, description="Number of sample rows to retrieve")
) -> DataSampleResponse:
    """
    Returns a preview sample of records from the active dataset.
    """
    df = state.get_dataset()
    sample_df = df.head(limit)
    records = sample_df.replace({np.nan: None}).to_dict(orient="records")

    return DataSampleResponse(
        total_records=int(len(df)),
        sample_size=int(len(sample_df)),
        columns=list(df.columns),
        records=records,
    )


@router.post(
    "/upload",
    response_model=DataUploadResponse,
    summary="Upload a new CSV dataset to replace active data in memory",
)
async def upload_dataset(
    file: UploadFile = File(..., description="CSV file containing customer segmentation data")
) -> DataUploadResponse:
    """
    Uploads a CSV file, parses and validates columns, updates the in-memory dataset,
    and returns initial ingestion confirmation with Hopkins clustering tendency.
    """
    if not file.filename.endswith((".csv", ".txt")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a CSV (.csv or .txt).",
        )

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse CSV file: {str(ex)}",
        )

    if len(df) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Dataset must contain at least 10 records.",
        )

    # Set as active dataset in singleton state
    state.set_dataset(df)

    # Compute Hopkins statistic
    numeric_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in df.columns]
    if len(numeric_cols) > 0:
        num_matrix = df[numeric_cols].fillna(df[numeric_cols].median()).values
        std_devs = np.std(num_matrix, axis=0)
        std_devs = np.where(std_devs == 0, 1.0, std_devs)
        std_matrix = (num_matrix - np.mean(num_matrix, axis=0)) / std_devs
        hopkins = compute_hopkins_statistic(std_matrix, random_state=42)
    else:
        hopkins = 0.5

    return DataUploadResponse(
        success=True,
        message=f"Successfully loaded {len(df)} records with {len(df.columns)} columns.",
        n_samples=int(len(df)),
        n_columns=int(len(df.columns)),
        columns=list(df.columns),
        hopkins_statistic=round(hopkins, 4),
    )
