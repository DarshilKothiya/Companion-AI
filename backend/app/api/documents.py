"""
Document upload and management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from typing import List
import os
import logging
from datetime import datetime, timezone

from app.models.database import User, ManualDocument, DocumentStatus, DeviceCategory
from app.models.schemas import DocumentUploadResponse, DocumentListResponse, DocumentMetadata
from app.core.auth import get_current_active_user
from app.core.config import settings
from app.services.document_processor import process_document_task

logger = logging.getLogger(__name__)
router = APIRouter()


def validate_file_extension(filename: str) -> bool:
    """Check if file extension is allowed."""
    ext = filename.split('.')[-1].lower()
    return ext in settings.allowed_extensions_list


def validate_file_size(file_size: int) -> bool:
    """Check if file size is within limit."""
    return file_size <= settings.max_upload_size_bytes


async def update_device_catalog(device_type: str, brand: str, model: str = None):
    """
    Update the device catalog with new device/brand/model information.
    
    Args:
        device_type: Type of device
        brand: Device brand
        model: Optional device model
    """
    try:
        # Find or create device category
        category = await DeviceCategory.find_one(DeviceCategory.name == device_type)
        
        if not category:
            # Create new category
            category = DeviceCategory(
                name=device_type,
                brands=[brand],
                models={brand: [model] if model else []}
            )
            await category.insert()
            logger.info(f"Created new device category: {device_type}")
        else:
            # Update existing category
            updated = False
            
            # Add brand if not exists
            if brand not in category.brands:
                category.brands.append(brand)
                updated = True
            
            # Add model if provided
            if model:
                if brand not in category.models:
                    category.models[brand] = [model]
                    updated = True
                elif model not in category.models[brand]:
                    category.models[brand].append(model)
                    updated = True
            
            if updated:
                category.updated_at = datetime.now(timezone.utc)
                await category.save()
                logger.info(f"Updated device category: {device_type}")
    
    except Exception as e:
        logger.error(f"Error updating device catalog: {e}")
        # Don't fail the upload if catalog update fails



@router.post("/upload-manual", response_model=DocumentUploadResponse)
async def upload_manual(
    file: UploadFile = File(...),
    device_type: str = Form(...),
    brand: str = Form(...),
    model: str = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload a device manual for processing.
    
    Args:
        file: PDF/text file to upload
        device_type: Type of device (e.g., Refrigerator, Washing Machine)
        brand: Device brand (e.g., Samsung, LG)
        model: Optional device model
        current_user: Authenticated user
        
    Returns:
        Upload confirmation with document ID
    """
    try:
        # Normalize inputs (trim whitespace, standardize)
        device_type = device_type.strip() if device_type else ""
        brand = brand.strip() if brand else ""
        model = model.strip() if model and model.strip() else None
        
        # Validate inputs
        if not device_type or not brand:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device type and brand are required"
            )
        
        # Validate file extension
        if not validate_file_extension(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type not allowed. Allowed types: {', '.join(settings.allowed_extensions_list)}"
            )
        
        # Read file
        contents = await file.read()
        file_size = len(contents)
        
        # Validate file size
        if not validate_file_size(file_size):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB"
            )
        
        # Create upload directory if it doesn't exist
        os.makedirs(settings.upload_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{device_type}_{brand}_{timestamp}_{file.filename}"
        file_path = os.path.join(settings.upload_dir, safe_filename)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Create document record
        document = ManualDocument(
            filename=file.filename,
            device_type=device_type,
            brand=brand,
            model=model,
            file_path=file_path,
            file_size=file_size,
            status=DocumentStatus.PENDING,
            uploaded_by=current_user
        )
        await document.insert()
        
        # Queue document for processing (async task)
        # This would typically use Celery, but for now we'll process synchronously
        try:
            from app.services.document_processor import DocumentProcessor
            processor = DocumentProcessor()
            success = await processor.process_document(document.id)
            
            # Update device catalog if processing was successful
            if success:
                await update_device_catalog(device_type, brand, model)
        except Exception as e:
            logger.error(f"Error queuing document for processing: {e}")
            # Document will remain in PENDING status
        
        logger.info(f"Document uploaded: {document.document_id}")
        
        return DocumentUploadResponse(
            document_id=document.document_id,
            filename=file.filename,
            device_type=device_type,
            brand=brand,
            model=model,
            status=document.status.value,
            message="Document uploaded successfully and queued for processing"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


@router.get("/documents", response_model=List[DocumentListResponse])
async def list_documents(
    current_user: User = Depends(get_current_active_user),
    device_type: str = None,
    brand: str = None,
    status_filter: str = None,
    limit: int = 50,
    skip: int = 0
):
    """
    List uploaded documents.
    
    Args:
        current_user: Authenticated user
        device_type: Optional filter by device type
        brand: Optional filter by brand
        status_filter: Optional filter by status
        limit: Maximum number of documents to return
        skip: Number of documents to skip
        
    Returns:
        List of documents
    """
    # Build query
    query = ManualDocument.uploaded_by.id == current_user.id
    
    if device_type:
        query = query & (ManualDocument.device_type == device_type)
    if brand:
        query = query & (ManualDocument.brand == brand)
    if status_filter:
        try:
            status_enum = DocumentStatus(status_filter)
            query = query & (ManualDocument.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Valid values: {[s.value for s in DocumentStatus]}"
            )
    
    # Execute query
    documents = await ManualDocument.find(query).sort("-uploaded_at").skip(skip).limit(limit).to_list()
    
    # Format response
    return [
        DocumentListResponse(
            document_id=doc.document_id,
            filename=doc.filename,
            device_type=doc.device_type,
            brand=doc.brand,
            model=doc.model,
            status=doc.status.value,
            chunks_count=doc.chunks_count,
            uploaded_at=doc.uploaded_at,
            processed_at=doc.processed_at
        )
        for doc in documents
    ]


@router.get("/documents/{document_id}", response_model=DocumentListResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Get document details.
    
    Args:
        document_id: Document ID
        current_user: Authenticated user
        
    Returns:
        Document details
    """
    document = await ManualDocument.find_one(
        ManualDocument.document_id == document_id
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Verify ownership
    await document.fetch_link(ManualDocument.uploaded_by)
    if document.uploaded_by.id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this document"
        )
    
    return DocumentListResponse(
        document_id=document.document_id,
        filename=document.filename,
        device_type=document.device_type,
        brand=document.brand,
        model=document.model,
        status=document.status.value,
        chunks_count=document.chunks_count,
        uploaded_at=document.uploaded_at,
        processed_at=document.processed_at
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete a document.
    
    Args:
        document_id: Document ID
        current_user: Authenticated user
        
    Returns:
        Success message
    """
    document = await ManualDocument.find_one(
        ManualDocument.document_id == document_id
    )
    
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    
    # Verify ownership
    await document.fetch_link(ManualDocument.uploaded_by)
    if document.uploaded_by.id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this document"
        )
    
    # Delete file from disk
    try:
        if os.path.exists(document.file_path):
            os.remove(document.file_path)
    except Exception as e:
        logger.warning(f"Error deleting file: {e}")
    
    # Delete from vector store
    try:
        from app.services.rag_service import rag_service
        rag_service.delete_document(document_id)
    except Exception as e:
        logger.warning(f"Error removing from vector store: {e}")
    
    # Delete document record
    await document.delete()
    
    logger.info(f"Document deleted: {document_id}")
    
    return {"message": "Document deleted successfully"}


@router.post("/documents/{document_id}/reindex")
async def reindex_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """
    Re-process a FAILED document without re-uploading the file.

    Useful when a document partially uploaded (e.g. last batch timed out).
    Resets the status to PENDING and re-runs the full embedding + indexing
    pipeline from the already-saved PDF on disk.
    """
    document = await ManualDocument.find_one(
        ManualDocument.document_id == document_id
    )

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    # Verify ownership
    await document.fetch_link(ManualDocument.uploaded_by)
    if document.uploaded_by.id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to reindex this document",
        )

    if not document.file_path or not os.path.exists(document.file_path):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Original file not found on disk. Please re-upload the document.",
        )

    # Reset status so the processor will run again
    document.status = DocumentStatus.PENDING
    document.error_message = None
    document.chunks_count = 0
    await document.save()

    # Re-process (same pipeline as upload)
    try:
        from app.services.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        success = await processor.process_document(document.id)

        if success:
            return {
                "message": "Document re-indexed successfully",
                "document_id": document_id,
                "status": "indexed",
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Re-indexing failed. Check server logs for details.",
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error re-indexing document {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Re-indexing error: {str(e)}",
        )

