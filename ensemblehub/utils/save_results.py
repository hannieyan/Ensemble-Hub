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
        
        # Create single log file for the entire API session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.base_dir, f"api_session_{timestamp}.jsonl")
        
        # Initialize the log file with session start info
        session_start = {
            "session_start": datetime.now().isoformat(),
            "log_format": "jsonlines",
            "description": "Ensemble-Hub API session log"
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(session_start, f, ensure_ascii=False)
            f.write('\n')
    
    def _ensure_directory_exists(self):
        """Create the save directory if it doesn't exist"""
        os.makedirs(self.base_dir, exist_ok=True)
        logger.info(f"💾 Result saver initialized: {self.base_dir}")
        # Add debug info
        abs_path = os.path.abspath(self.base_dir)
        logger.info(f"💾 Absolute save path: {abs_path}")
        logger.info(f"💾 Directory exists: {os.path.exists(abs_path)}")
    
    def save_request_result(
        self, 
        request_data: Dict[str, Any], 
        response_data: Dict[str, Any],
        endpoint: str,
        request_id: Optional[str] = None
    ) -> str:
        """
        Append a complete request-response pair to the session log file.
        
        Args:
            request_data: The original HTTP request data
            response_data: The API response data
            endpoint: The API endpoint (e.g., "/v1/completions")
            request_id: Optional request ID, will generate one if not provided
            
        Returns:
            Path to the log file
        """
        if request_id is None:
            request_id = response_data.get("id", f"req_{int(time.time() * 1000000)}")
        
        # Prepare log entry for this request
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "endpoint": endpoint,
            "request": request_data,
            "response": response_data
        }
        
        try:
            # Append to the session log file (JSONL format)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, ensure_ascii=False)
                f.write('\n')
            
            logger.info(f"💾 Appended result to session log: {self.log_file}")
            return self.log_file
            
        except Exception as e:
            logger.error(f"Failed to append result to {self.log_file}: {e}")
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