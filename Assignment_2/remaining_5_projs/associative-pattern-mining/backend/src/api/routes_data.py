"""FastAPI routes for dataset inspection, synthetic generation, and upload."""

from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, Any, List
import io
import pandas as pd
from ..core.state import app_state
from ..core.dataset import DatasetLoader, SyntheticTransactionGenerator
from .schemas import DatasetSummaryResponse, SyntheticGenerateRequest

router = APIRouter(prefix="/data", tags=["Dataset"])


@router.get("/summary", response_model=DatasetSummaryResponse)
def get_dataset_summary():
    """Returns descriptive statistics, sparsity, basket distributions, and top catalog items."""
    return app_state.dataset.get_summary_stats()


@router.get("/sample")
def get_dataset_sample(limit: int = 25) -> Dict[str, Any]:
    """Returns a sample of transaction baskets for inspection."""
    txs = [sorted(list(t)) for t in app_state.dataset.transactions[:limit]]
    return {
        "total_transactions": app_state.dataset.num_transactions,
        "sample_count": len(txs),
        "transactions": txs
    }


@router.post("/generate", response_model=DatasetSummaryResponse)
def generate_synthetic_data(req: SyntheticGenerateRequest):
    """Generates synthetic transactions with Zipfian frequency and planted affinity clusters."""
    synth_dataset = SyntheticTransactionGenerator.generate(
        num_transactions=req.num_transactions,
        avg_basket_size=req.avg_basket_size,
        noise_level=req.noise_level,
        seed=req.seed
    )
    app_state.set_dataset(synth_dataset)
    return app_state.dataset.get_summary_stats()


@router.post("/load-preset/{preset_name}", response_model=DatasetSummaryResponse)
def load_preset_dataset(preset_name: str):
    """Switches active dataset to 'retail' or 'synthetic'."""
    if preset_name == "retail":
        path = "data/online_retail_sample.csv"
    elif preset_name == "synthetic":
        path = "data/synthetic_sample.csv"
    else:
        raise HTTPException(status_code=400, detail="Unknown preset name. Choose 'retail' or 'synthetic'.")

    try:
        new_ds = DatasetLoader.load_from_csv(path)
        app_state.set_dataset(new_ds)
        return app_state.dataset.get_summary_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load preset: {str(e)}")


@router.post("/upload", response_model=DatasetSummaryResponse)
async def upload_csv_dataset(file: UploadFile = File(...)):
    """Uploads and parses a custom transaction CSV dataset."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        new_ds = DatasetLoader.parse_dataframe(df, source_name=file.filename)
        app_state.set_dataset(new_ds)
        return app_state.dataset.get_summary_stats()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV file: {str(e)}")
