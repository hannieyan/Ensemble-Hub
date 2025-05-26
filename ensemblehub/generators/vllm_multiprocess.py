"""
Multi-process vLLM runner for multi-GPU support
"""
import multiprocessing
import os
import queue
import time
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class VLLMWorker(multiprocessing.Process):
    """Worker process for running vLLM on a specific GPU"""
    
    def __init__(self, model_path: str, device_id: int, request_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
        super().__init__()
        self.model_path = model_path
        self.device_id = device_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.daemon = True
        
    def run(self):
        """Run in subprocess with isolated CUDA environment"""
        # Set CUDA device BEFORE any imports
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_id)
        
        # Now import vLLM (CUDA will only see the specified device)
        from vllm import LLM, SamplingParams
        
        logger.info(f"Starting vLLM worker for {self.model_path} on GPU {self.device_id}")
        
        # Initialize vLLM
        llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=4096,
            tensor_parallel_size=1,
            disable_log_stats=True,
            enforce_eager=True
        )
        
        # Process requests
        while True:
            try:
                request = self.request_queue.get(timeout=1)
                if request is None:  # Shutdown signal
                    break
                    
                request_id, prompts, sampling_params = request
                
                # Generate
                outputs = llm.generate(prompts, sampling_params)
                
                # Send response
                self.response_queue.put((request_id, outputs))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in vLLM worker: {e}")
                self.response_queue.put((request_id, e))


class MultiProcessVLLMPool:
    """Manages multiple vLLM instances across different GPUs"""
    
    def __init__(self):
        self.workers: Dict[Tuple[str, int], VLLMWorker] = {}
        self.request_queues: Dict[Tuple[str, int], multiprocessing.Queue] = {}
        self.response_queues: Dict[Tuple[str, int], multiprocessing.Queue] = {}
        self.request_counter = 0
        
    def get_or_create_worker(self, model_path: str, device_id: int) -> Tuple[multiprocessing.Queue, multiprocessing.Queue]:
        """Get or create a worker for the specified model and device"""
        key = (model_path, device_id)
        
        if key not in self.workers:
            # Create queues
            req_queue = multiprocessing.Queue()
            resp_queue = multiprocessing.Queue()
            
            # Create and start worker
            worker = VLLMWorker(model_path, device_id, req_queue, resp_queue)
            worker.start()
            
            # Store references
            self.workers[key] = worker
            self.request_queues[key] = req_queue
            self.response_queues[key] = resp_queue
            
            # Give worker time to initialize
            time.sleep(5)
            
        return self.request_queues[key], self.response_queues[key]
    
    def generate(self, model_path: str, device_id: int, prompts: List[str], **kwargs) -> List[str]:
        """Generate text using the specified model and device"""
        req_queue, resp_queue = self.get_or_create_worker(model_path, device_id)
        
        # Create sampling params
        from vllm import SamplingParams
        sampling_params = SamplingParams(
            max_tokens=kwargs.get('max_tokens', 256),
            temperature=kwargs.get('temperature', 0.95),
            top_p=kwargs.get('top_p', 0.7),
            stop=["\n"]
        )
        
        # Send request
        request_id = self.request_counter
        self.request_counter += 1
        req_queue.put((request_id, prompts, sampling_params))
        
        # Wait for response
        while True:
            try:
                resp_id, result = resp_queue.get(timeout=30)
                if resp_id == request_id:
                    if isinstance(result, Exception):
                        raise result
                    return [output.outputs[0].text for output in result]
            except queue.Empty:
                raise TimeoutError("vLLM worker timeout")
    
    def shutdown(self):
        """Shutdown all workers"""
        for key in self.workers:
            self.request_queues[key].put(None)
        
        for worker in self.workers.values():
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()


# Global pool instance
_vllm_pool = None

def get_vllm_pool() -> MultiProcessVLLMPool:
    """Get the global vLLM pool"""
    global _vllm_pool
    if _vllm_pool is None:
        _vllm_pool = MultiProcessVLLMPool()
    return _vllm_pool