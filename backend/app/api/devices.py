"""
Device catalog API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
import asyncio
import logging

from app.models.database import ManualDocument, DocumentStatus
from app.models.schemas import DeviceListResponse, DeviceInfo
from app.core.auth import get_current_active_user

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/devices", response_model=DeviceListResponse)
async def list_devices():
    """
    Get list of supported devices with brands and models.

    Dynamically aggregates device_type / brand / model from the ManualDocument
    collection (indexed documents only), so the dropdown is always in sync with
    what has actually been indexed — no separate catalog sync required.
    """
    try:
        # Include INDEXED docs AND partially-uploaded FAILED docs
        # (a failed doc may still have many chunks in Qdrant)
        docs = await ManualDocument.find(
            {
                "$or": [
                    {"status": DocumentStatus.INDEXED},
                    {"status": DocumentStatus.FAILED, "chunks_count": {"$gt": 0}},
                ]
            }
        ).to_list()

        # Aggregate into  { device_type -> { brand -> {model, ...} } }
        catalog: dict = {}
        for doc in docs:
            dt = (doc.device_type or "").strip()
            br = (doc.brand or "").strip()
            mo = (doc.model or "").strip()
            if not dt or not br:
                continue
            if dt not in catalog:
                catalog[dt] = {}
            if br not in catalog[dt]:
                catalog[dt][br] = set()
            if mo:
                catalog[dt][br].add(mo)

        # Build response
        devices = [
            DeviceInfo(
                device_type=dt,
                brands=sorted(catalog[dt].keys()),
                models={br: sorted(catalog[dt][br]) for br in catalog[dt]},
            )
            for dt in sorted(catalog.keys())
        ]

        logger.info(f"Returning {len(devices)} device type(s) from indexed documents")
        return DeviceListResponse(devices=devices, total_count=len(devices))

    except Exception as e:
        logger.error(f"Error listing devices: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving device list",
        )


@router.get("/devices/{device_type}")
async def get_device_info(device_type: str):
    """
    Get information about a specific device type.

    Args:
        device_type: Type of device (case-sensitive, must match stored value)

    Returns:
        Device information with brands and models
    """
    docs = await ManualDocument.find(
        ManualDocument.device_type == device_type,
        ManualDocument.status == DocumentStatus.INDEXED,
    ).to_list()

    if not docs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Device type '{device_type}' not found or has no indexed documents",
        )

    brands_models: dict = {}
    for doc in docs:
        br = (doc.brand or "").strip()
        mo = (doc.model or "").strip()
        if not br:
            continue
        if br not in brands_models:
            brands_models[br] = set()
        if mo:
            brands_models[br].add(mo)

    return DeviceInfo(
        device_type=device_type,
        brands=sorted(brands_models.keys()),
        models={br: sorted(brands_models[br]) for br in brands_models},
    )
