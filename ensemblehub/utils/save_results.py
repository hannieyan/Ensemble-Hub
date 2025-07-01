"""
Result saving utilities for Ensemble-Hub
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ResultSaver:
    """Save API results to disk for debugging and analysis"""
    
    def __init__(self, base_dir: str = "saves/logs"):
        """
        Initialize the result saver.
        
        Args:
            base_dir: Base directory to save results (default: saves/logs)
        """
        self.base_dir = base_dir
        self._ensure_directory_exists()
    
    def _ensure_directory_exists(self):
        """Create the save directory if it doesn't exist"""
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"Result saver initialized: {self.base_dir}")
    
    def save_request_result(
        self, 
        request_data: Dict[str, Any], 
        response_data: Dict[str, Any],
        endpoint: str,
        request_id: Optional[str] = None
    ) -> str:
        """
        Save a complete request-response pair to disk.
        
        Args:
            request_data: The original HTTP request data
            response_data: The API response data
            endpoint: The API endpoint (e.g., "/v1/completions")
            request_id: Optional request ID, will generate one if not provided
            
        Returns:
            Path to the saved file
        """
        if request_id is None:
            request_id = response_data.get("id", f"req_{int(time.time() * 1000000)}")
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Clean endpoint name for filename
        endpoint_name = endpoint.replace("/", "_").replace("v1_", "")
        
        # Create filename
        filename = f"{timestamp}_{endpoint_name}_{request_id}.json"
        filepath = os.path.join(self.base_dir, filename)
        
        # Prepare complete log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "endpoint": endpoint,
            "request": request_data,
            "response": response_data,
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "file_version": "1.0"
            }
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved result to: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Failed to save result to {filepath}: {e}")
            return None


# Global result saver instance
_result_saver = None

def get_result_saver() -> ResultSaver:
    """Get the global result saver instance"""
    global _result_saver
    if _result_saver is None:
        _result_saver = ResultSaver()
    return _result_saver

def save_api_result(
    request_data: Dict[str, Any], 
    response_data: Dict[str, Any],
    endpoint: str,
    request_id: Optional[str] = None
) -> str:
    """
    Convenience function to save API results.
    
    Args:
        request_data: The original HTTP request data
        response_data: The API response data  
        endpoint: The API endpoint
        request_id: Optional request ID
        
    Returns:
        Path to the saved file
    """
    saver = get_result_saver()
    return saver.save_request_result(request_data, response_data, endpoint, request_id)