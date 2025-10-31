"""
LLM-USE V2 - Universal LLM Router with Auto-Discovery
Sistema di routing intelligente con valutazione realistica dei modelli
Production Ready with Full Monitoring, Caching, and Rate Limiting
"""

import os
import json
import time
import hashlib
import re
import sqlite3
import threading
import logging
import pickle
import zlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import requests
from pathlib import Path
from collections import defaultdict, deque
from contextlib import contextmanager
from functools import lru_cache, wraps
from enum import Enum
import traceback
import uuid

# Ollama per modelli locali
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama not installed. Install with: pip install ollama")

# ====================
# PRODUCTION LOGGING
# ====================

class LogLevel(Enum):
    """Log levels for production logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ProductionLogger:
    """Enhanced production logger with structured logging and rotation"""
    
    def __init__(self, log_dir: str = None, app_name: str = "llm-use", 
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 max_files: int = 10):
        """
        Initialize production logger
        
        Args:
            log_dir: Directory for log files
            app_name: Application name for log files
            max_file_size: Maximum size per log file in bytes
            max_files: Maximum number of rotated files
        """
        self.app_name = app_name
        self.max_file_size = max_file_size
        self.max_files = max_files
        
        # Setup log directory
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path.home() / ".llm-use" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Current log file
        self.current_log_file = self.log_dir / f"{app_name}.log"
        self.log_lock = threading.Lock()
        
        # Setup Python logger
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler with rotation
        self._setup_handlers()
        
        # Structured log buffer for analysis
        self.log_buffer = deque(maxlen=1000)
        
        # Log session start
        self.session_id = str(uuid.uuid4())[:8]
        self.info(f"Logger initialized", extra={"session_id": self.session_id})
    
    def _setup_handlers(self):
        """Setup logging handlers with rotation"""
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.current_log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s | %(message)s | %(extra_data)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add custom filter for extra data
        file_handler.addFilter(self._add_extra_data)
        self.logger.addHandler(file_handler)
    
    def _add_extra_data(self, record):
        """Add extra data to log record"""
        extra_data = {}
        
        # Add custom fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName', 
                          'levelname', 'levelno', 'lineno', 'module', 'msecs', 
                          'pathname', 'process', 'processName', 'relativeCreated', 
                          'thread', 'threadName', 'exc_info', 'exc_text', 'stack_info']:
                extra_data[key] = value
        
        record.extra_data = json.dumps(extra_data) if extra_data else ''
        return True
    
    def _rotate_logs(self):
        """Rotate log files when size limit reached"""
        with self.log_lock:
            if not self.current_log_file.exists():
                return
            
            if self.current_log_file.stat().st_size < self.max_file_size:
                return
            
            # Rotate files
            for i in range(self.max_files - 1, 0, -1):
                old_file = self.log_dir / f"{self.app_name}.log.{i}"
                new_file = self.log_dir / f"{self.app_name}.log.{i + 1}"
                if old_file.exists():
                    if new_file.exists():
                        new_file.unlink()
                    old_file.rename(new_file)
            
            # Move current to .1
            first_backup = self.log_dir / f"{self.app_name}.log.1"
            if first_backup.exists():
                first_backup.unlink()
            self.current_log_file.rename(first_backup)
            
            # Recreate handlers
            self._setup_handlers()
    
    def _log(self, level: LogLevel, message: str, extra: Dict = None, 
             exc_info: bool = False):
        """Internal log method"""
        # Check rotation
        self._rotate_logs()
        
        # Prepare extra data
        log_extra = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            **(extra or {})
        }
        
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level.value,
            'message': message,
            'session_id': self.session_id,
            'extra': extra or {}
        }
        
        # Add to buffer
        self.log_buffer.append(log_entry)
        
        # Log based on level
        if level == LogLevel.DEBUG:
            self.logger.debug(message, extra=log_extra, exc_info=exc_info)
        elif level == LogLevel.INFO:
            self.logger.info(message, extra=log_extra, exc_info=exc_info)
        elif level == LogLevel.WARNING:
            self.logger.warning(message, extra=log_extra, exc_info=exc_info)
        elif level == LogLevel.ERROR:
            self.logger.error(message, extra=log_extra, exc_info=exc_info)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(message, extra=log_extra, exc_info=exc_info)
    
    def debug(self, message: str, extra: Dict = None):
        """Log debug message"""
        self._log(LogLevel.DEBUG, message, extra)
    
    def info(self, message: str, extra: Dict = None):
        """Log info message"""
        self._log(LogLevel.INFO, message, extra)
    
    def warning(self, message: str, extra: Dict = None):
        """Log warning message"""
        self._log(LogLevel.WARNING, message, extra)
    
    def error(self, message: str, extra: Dict = None, exc_info: bool = True):
        """Log error message"""
        self._log(LogLevel.ERROR, message, extra, exc_info)
    
    def critical(self, message: str, extra: Dict = None, exc_info: bool = True):
        """Log critical message"""
        self._log(LogLevel.CRITICAL, message, extra, exc_info)
    
    def log_api_call(self, provider: str, model: str, input_tokens: int, 
                     output_tokens: int, latency: float, success: bool, 
                     error: str = None):
        """Log API call details"""
        self.info("API call", extra={
            'provider': provider,
            'model': model,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'latency_ms': int(latency * 1000),
            'success': success,
            'error': error
        })
    
    def log_cache_event(self, event_type: str, key: str, hit: bool = None,
                       size_bytes: int = None):
        """Log cache events"""
        self.debug(f"Cache {event_type}", extra={
            'cache_key': key[:50],
            'cache_hit': hit,
            'size_bytes': size_bytes
        })
    
    def log_rate_limit(self, provider: str, remaining: int, reset_time: str):
        """Log rate limit events"""
        self.warning("Rate limit approaching", extra={
            'provider': provider,
            'remaining_calls': remaining,
            'reset_time': reset_time
        })
    
    def get_recent_errors(self, count: int = 10) -> List[Dict]:
        """Get recent error logs"""
        errors = [log for log in self.log_buffer 
                 if log['level'] in ['ERROR', 'CRITICAL']]
        return errors[-count:]
    
    def export_logs(self, start_time: datetime = None, 
                   end_time: datetime = None) -> List[Dict]:
        """Export logs for analysis"""
        logs = list(self.log_buffer)
        
        if start_time:
            logs = [log for log in logs 
                   if datetime.fromisoformat(log['timestamp']) >= start_time]
        
        if end_time:
            logs = [log for log in logs 
                   if datetime.fromisoformat(log['timestamp']) <= end_time]
        
        return logs

# ====================
# PRODUCTION DATABASE
# ====================

class ProductionDB:
    """SQLite database for production tracking and analytics"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize production database
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = Path.home() / ".llm-use" / "production.db"
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = None
        self.init_database()
    
    def init_database(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP,
                    total_messages INTEGER DEFAULT 0,
                    total_tokens_input INTEGER DEFAULT 0,
                    total_tokens_output INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0,
                    user_id TEXT,
                    metadata TEXT
                )
            """)
            
            # API calls table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_calls (
                    call_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_text TEXT,
                    output_text TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    latency_ms INTEGER,
                    cost REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    complexity_score INTEGER,
                    cache_hit BOOLEAN DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Model performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    avg_latency_ms REAL,
                    success_rate REAL,
                    total_calls INTEGER,
                    total_tokens INTEGER,
                    total_cost REAL,
                    quality_score REAL,
                    period_start TIMESTAMP,
                    period_end TIMESTAMP
                )
            """)
            
            # Cache performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    cache_hits INTEGER,
                    cache_misses INTEGER,
                    cache_size_bytes INTEGER,
                    evictions INTEGER,
                    avg_retrieval_ms REAL,
                    memory_saved_bytes INTEGER
                )
            """)
            
            # Rate limit events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rate_limit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    provider TEXT NOT NULL,
                    model TEXT,
                    limit_type TEXT,
                    remaining_calls INTEGER,
                    reset_time TIMESTAMP,
                    action_taken TEXT
                )
            """)
            
            # User feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    call_id TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    rating INTEGER,
                    feedback_text TEXT,
                    feedback_type TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY (call_id) REFERENCES api_calls(call_id)
                )
            """)
            
            # Indexes for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_session ON api_calls(session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_timestamp ON api_calls(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_api_calls_model ON api_calls(model)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_perf_timestamp ON model_performance(timestamp)")
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def create_session(self, user_id: str = None, metadata: Dict = None) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO sessions (session_id, user_id, metadata)
                VALUES (?, ?, ?)
            """, (session_id, user_id, json.dumps(metadata) if metadata else None))
            conn.commit()
        
        return session_id
    
    def update_session(self, session_id: str, messages: int = None, 
                      tokens_input: int = None, tokens_output: int = None,
                      cost: float = None, end_time: bool = False):
        """Update session statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            updates = []
            values = []
            
            if messages is not None:
                updates.append("total_messages = total_messages + ?")
                values.append(messages)
            
            if tokens_input is not None:
                updates.append("total_tokens_input = total_tokens_input + ?")
                values.append(tokens_input)
            
            if tokens_output is not None:
                updates.append("total_tokens_output = total_tokens_output + ?")
                values.append(tokens_output)
            
            if cost is not None:
                updates.append("total_cost = total_cost + ?")
                values.append(cost)
            
            if end_time:
                updates.append("end_time = CURRENT_TIMESTAMP")
            
            if updates:
                values.append(session_id)
                cursor.execute(f"""
                    UPDATE sessions 
                    SET {', '.join(updates)}
                    WHERE session_id = ?
                """, values)
                conn.commit()
    
    def log_api_call(self, session_id: str, provider: str, model: str,
                    input_text: str, output_text: str, input_tokens: int,
                    output_tokens: int, latency_ms: int, cost: float,
                    success: bool, error_message: str = None,
                    complexity_score: int = None, cache_hit: bool = False) -> str:
        """Log API call to database"""
        call_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO api_calls (
                    call_id, session_id, provider, model, input_text, 
                    output_text, input_tokens, output_tokens, latency_ms,
                    cost, success, error_message, complexity_score, cache_hit
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                call_id, session_id, provider, model, input_text[:1000],
                output_text[:1000] if output_text else None,
                input_tokens, output_tokens, latency_ms, cost,
                success, error_message, complexity_score, cache_hit
            ))
            conn.commit()
        
        return call_id
    
    def update_model_performance(self, model: str, provider: str,
                                period_minutes: int = 60):
        """Calculate and store model performance metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Calculate metrics for the period
            period_start = datetime.now() - timedelta(minutes=period_minutes)
            
            cursor.execute("""
                SELECT 
                    AVG(latency_ms) as avg_latency,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    COUNT(*) as total_calls,
                    SUM(input_tokens + output_tokens) as total_tokens,
                    SUM(cost) as total_cost
                FROM api_calls
                WHERE model = ? AND provider = ? AND timestamp >= ?
            """, (model, provider, period_start))
            
            row = cursor.fetchone()
            
            if row and row['total_calls'] > 0:
                cursor.execute("""
                    INSERT INTO model_performance (
                        model, provider, avg_latency_ms, success_rate,
                        total_calls, total_tokens, total_cost,
                        period_start, period_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    model, provider, row['avg_latency'], row['success_rate'],
                    row['total_calls'], row['total_tokens'] or 0, 
                    row['total_cost'] or 0, period_start, datetime.now()
                ))
                conn.commit()
    
    def log_cache_performance(self, hits: int, misses: int, size_bytes: int,
                            evictions: int, avg_retrieval_ms: float,
                            memory_saved: int):
        """Log cache performance metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO cache_performance (
                    cache_hits, cache_misses, cache_size_bytes,
                    evictions, avg_retrieval_ms, memory_saved_bytes
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (hits, misses, size_bytes, evictions, avg_retrieval_ms, memory_saved))
            conn.commit()
    
    def log_rate_limit_event(self, provider: str, model: str = None,
                           limit_type: str = None, remaining: int = None,
                           reset_time: datetime = None, action: str = None):
        """Log rate limit event"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO rate_limit_events (
                    provider, model, limit_type, remaining_calls,
                    reset_time, action_taken
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (provider, model, limit_type, remaining, reset_time, action))
            conn.commit()
    
    def add_user_feedback(self, session_id: str = None, call_id: str = None,
                        rating: int = None, feedback_text: str = None,
                        feedback_type: str = None):
        """Add user feedback"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_feedback (
                    session_id, call_id, rating, feedback_text, feedback_type
                ) VALUES (?, ?, ?, ?, ?)
            """, (session_id, call_id, rating, feedback_text, feedback_type))
            conn.commit()
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get session statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if row:
                return dict(row)
            return {}
    
    def get_model_analytics(self, hours: int = 24) -> List[Dict]:
        """Get model performance analytics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            since = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT 
                    model,
                    provider,
                    COUNT(*) as total_calls,
                    AVG(latency_ms) as avg_latency,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as success_rate,
                    SUM(cost) as total_cost,
                    AVG(complexity_score) as avg_complexity,
                    SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as cache_hit_rate
                FROM api_calls
                WHERE timestamp >= ?
                GROUP BY model, provider
                ORDER BY total_calls DESC
            """, (since,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_error_analysis(self, hours: int = 24) -> List[Dict]:
        """Analyze recent errors"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            since = datetime.now() - timedelta(hours=hours)
            
            cursor.execute("""
                SELECT 
                    model,
                    provider,
                    error_message,
                    COUNT(*) as error_count,
                    MAX(timestamp) as last_occurrence
                FROM api_calls
                WHERE success = 0 AND timestamp >= ?
                GROUP BY model, provider, error_message
                ORDER BY error_count DESC
            """, (since,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cutoff = datetime.now() - timedelta(days=days)
            
            # Delete old API calls
            cursor.execute("DELETE FROM api_calls WHERE timestamp < ?", (cutoff,))
            
            # Delete old sessions
            cursor.execute("DELETE FROM sessions WHERE start_time < ?", (cutoff,))
            
            # Delete old performance data
            cursor.execute("DELETE FROM model_performance WHERE timestamp < ?", (cutoff,))
            
            # Delete old cache performance
            cursor.execute("DELETE FROM cache_performance WHERE timestamp < ?", (cutoff,))
            
            conn.commit()

# ====================
# PRODUCTION CACHE
# ====================

class CacheEntry:
    """Single cache entry with metadata"""
    
    def __init__(self, key: str, value: Any, ttl: int = 3600):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.hits = 0
        self.last_accessed = time.time()
        self.size_bytes = len(pickle.dumps(value))
    
    def is_expired(self) -> bool:
        """Check if entry is expired"""
        return time.time() - self.created_at > self.ttl
    
    def access(self):
        """Update access statistics"""
        self.hits += 1
        self.last_accessed = time.time()

class ProductionCache:
    """Context-aware production cache with compression and eviction"""
    
    def __init__(self, max_size_mb: int = 100, ttl_seconds: int = 3600,
                 compression: bool = True, logger: ProductionLogger = None):
        """
        Initialize production cache
        
        Args:
            max_size_mb: Maximum cache size in MB
            ttl_seconds: Default TTL in seconds
            compression: Enable compression for large values
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = ttl_seconds
        self.compression = compression
        self.logger = logger
        
        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}
        self.cache_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'current_size': 0,
            'compression_savings': 0
        }
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
    
    def _generate_key(self, provider: str, model: str, messages: List[Dict],
                     temperature: float = None, max_tokens: int = None) -> str:
        """Generate cache key from request parameters"""
        # Create deterministic key
        key_data = {
            'provider': provider,
            'model': model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def _compress_value(self, value: Any) -> bytes:
        """Compress value if beneficial"""
        pickled = pickle.dumps(value)
        
        if self.compression and len(pickled) > 1024:  # Only compress if > 1KB
            compressed = zlib.compress(pickled, level=6)
            
            if len(compressed) < len(pickled) * 0.9:  # Only use if 10% smaller
                self.stats['compression_savings'] += len(pickled) - len(compressed)
                return compressed
        
        return pickled
    
    def _decompress_value(self, data: bytes) -> Any:
        """Decompress value if needed"""
        try:
            # Try decompression first
            if self.compression:
                try:
                    decompressed = zlib.decompress(data)
                    return pickle.loads(decompressed)
                except zlib.error:
                    pass
            
            # Fall back to direct unpickling
            return pickle.loads(data)
        except Exception as e:
            if self.logger:
                self.logger.error(f"Cache decompression error: {e}")
            return None
    
    def get(self, provider: str, model: str, messages: List[Dict],
            temperature: float = None, max_tokens: int = None) -> Optional[str]:
        """Get cached response if available"""
        key = self._generate_key(provider, model, messages, temperature, max_tokens)
        
        with self.cache_lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check expiration
                if entry.is_expired():
                    del self.cache[key]
                    self.stats['current_size'] -= entry.size_bytes
                    self.stats['misses'] += 1
                    
                    if self.logger:
                        self.logger.log_cache_event("expired", key, hit=False)
                    
                    return None
                
                # Update access stats
                entry.access()
                self.stats['hits'] += 1
                
                if self.logger:
                    self.logger.log_cache_event("hit", key, hit=True, 
                                               size_bytes=entry.size_bytes)
                
                # Decompress and return
                return self._decompress_value(entry.value)
            
            self.stats['misses'] += 1
            
            if self.logger:
                self.logger.log_cache_event("miss", key, hit=False)
            
            return None
    
    def set(self, provider: str, model: str, messages: List[Dict],
            response: str, temperature: float = None, max_tokens: int = None,
            ttl: int = None):
        """Cache a response"""
        key = self._generate_key(provider, model, messages, temperature, max_tokens)
        
        # Compress value
        compressed = self._compress_value(response)
        
        with self.cache_lock:
            # Check if we need to evict
            entry_size = len(compressed)
            self._evict_if_needed(entry_size)
            
            # Create and store entry
            entry = CacheEntry(key, compressed, ttl or self.default_ttl)
            self.cache[key] = entry
            self.stats['current_size'] += entry.size_bytes
            
            if self.logger:
                self.logger.log_cache_event("set", key, size_bytes=entry.size_bytes)
    
    def _evict_if_needed(self, new_size: int):
        """Evict entries if cache is full"""
        while self.stats['current_size'] + new_size > self.max_size_bytes:
            if not self.cache:
                break
            
            # LRU eviction
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].last_accessed)
            
            entry = self.cache[oldest_key]
            del self.cache[oldest_key]
            
            self.stats['current_size'] -= entry.size_bytes
            self.stats['evictions'] += 1
            
            if self.logger:
                self.logger.log_cache_event("evict", oldest_key, 
                                           size_bytes=entry.size_bytes)
    
    def _cleanup_worker(self):
        """Background thread to clean expired entries"""
        while True:
            time.sleep(60)  # Check every minute
            
            with self.cache_lock:
                expired_keys = []
                
                for key, entry in self.cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    entry = self.cache[key]
                    del self.cache[key]
                    self.stats['current_size'] -= entry.size_bytes
                    
                    if self.logger:
                        self.logger.log_cache_event("cleanup", key)
    
    def clear(self):
        """Clear entire cache"""
        with self.cache_lock:
            self.cache.clear()
            self.stats['current_size'] = 0
            
            if self.logger:
                self.logger.info("Cache cleared")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with self.cache_lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'hit_rate': hit_rate,
                'evictions': self.stats['evictions'],
                'current_size_mb': self.stats['current_size'] / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'entries': len(self.cache),
                'compression_savings_mb': self.stats['compression_savings'] / (1024 * 1024)
            }

# ====================
# PRODUCTION METRICS
# ====================

class MetricType(Enum):
    """Types of metrics to collect"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class Metric:
    """Single metric with history"""
    
    def __init__(self, name: str, metric_type: MetricType, 
                 window_size: int = 1000):
        self.name = name
        self.metric_type = metric_type
        self.values = deque(maxlen=window_size)
        self.timestamp = time.time()
        self.lock = threading.Lock()
    
    def add(self, value: float, timestamp: float = None):
        """Add value to metric"""
        with self.lock:
            ts = timestamp or time.time()
            self.values.append((ts, value))
            self.timestamp = ts
    
    def get_stats(self) -> Dict:
        """Get metric statistics"""
        with self.lock:
            if not self.values:
                return {'count': 0}
            
            values_only = [v[1] for v in self.values]
            
            if self.metric_type == MetricType.COUNTER:
                return {
                    'count': len(values_only),
                    'total': sum(values_only),
                    'rate': self._calculate_rate()
                }
            
            elif self.metric_type == MetricType.GAUGE:
                return {
                    'current': values_only[-1],
                    'min': min(values_only),
                    'max': max(values_only),
                    'avg': sum(values_only) / len(values_only)
                }
            
            elif self.metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
                sorted_values = sorted(values_only)
                count = len(sorted_values)
                
                return {
                    'count': count,
                    'min': sorted_values[0],
                    'max': sorted_values[-1],
                    'mean': sum(sorted_values) / count,
                    'p50': sorted_values[int(count * 0.5)],
                    'p95': sorted_values[int(count * 0.95)],
                    'p99': sorted_values[int(count * 0.99)]
                }
            
            return {}
    
    def _calculate_rate(self) -> float:
        """Calculate rate for counter metrics"""
        if len(self.values) < 2:
            return 0.0
        
        time_diff = self.values[-1][0] - self.values[0][0]
        if time_diff == 0:
            return 0.0
        
        return len(self.values) / time_diff

class ProductionMetrics:
    """Production metrics collection and reporting"""
    
    def __init__(self, export_interval: int = 60, logger: ProductionLogger = None):
        """
        Initialize metrics collector
        
        Args:
            export_interval: Interval in seconds to export metrics
        """
        self.metrics: Dict[str, Metric] = {}
        self.export_interval = export_interval
        self.logger = logger
        self.lock = threading.Lock()
        
        # Start export thread
        self.export_thread = threading.Thread(target=self._export_worker, daemon=True)
        self.export_thread.start()
        
        # Initialize standard metrics
        self._init_standard_metrics()
    
    def _init_standard_metrics(self):
        """Initialize standard metrics"""
        # API metrics
        self.create_metric("api_requests_total", MetricType.COUNTER)
        self.create_metric("api_errors_total", MetricType.COUNTER)
        self.create_metric("api_latency_ms", MetricType.TIMER)
        self.create_metric("api_tokens_input", MetricType.COUNTER)
        self.create_metric("api_tokens_output", MetricType.COUNTER)
        self.create_metric("api_cost_usd", MetricType.COUNTER)
        
        # Cache metrics
        self.create_metric("cache_hits", MetricType.COUNTER)
        self.create_metric("cache_misses", MetricType.COUNTER)
        self.create_metric("cache_size_bytes", MetricType.GAUGE)
        self.create_metric("cache_evictions", MetricType.COUNTER)
        
        # System metrics
        self.create_metric("active_sessions", MetricType.GAUGE)
        self.create_metric("memory_usage_mb", MetricType.GAUGE)
        self.create_metric("complexity_score", MetricType.HISTOGRAM)
        
        # Model routing metrics
        self.create_metric("model_switches", MetricType.COUNTER)
        self.create_metric("routing_decisions", MetricType.COUNTER)
    
    def create_metric(self, name: str, metric_type: MetricType) -> Metric:
        """Create or get a metric"""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = Metric(name, metric_type)
            return self.metrics[name]
    
    def increment(self, name: str, value: float = 1.0):
        """Increment a counter metric"""
        metric = self.create_metric(name, MetricType.COUNTER)
        metric.add(value)
    
    def gauge(self, name: str, value: float):
        """Set a gauge metric"""
        metric = self.create_metric(name, MetricType.GAUGE)
        metric.add(value)
    
    def histogram(self, name: str, value: float):
        """Add value to histogram"""
        metric = self.create_metric(name, MetricType.HISTOGRAM)
        metric.add(value)
    
    def timer(self, name: str, duration_ms: float):
        """Record timer duration"""
        metric = self.create_metric(name, MetricType.TIMER)
        metric.add(duration_ms)
    
    @contextmanager
    def time(self, name: str):
        """Context manager to time operations"""
        start = time.time()
        yield
        duration_ms = (time.time() - start) * 1000
        self.timer(name, duration_ms)
    
    def record_api_call(self, provider: str, model: str, success: bool,
                       latency_ms: float, input_tokens: int, output_tokens: int,
                       cost: float, complexity: int = None):
        """Record API call metrics"""
        # Base metrics
        self.increment("api_requests_total")
        self.timer("api_latency_ms", latency_ms)
        self.increment("api_tokens_input", input_tokens)
        self.increment("api_tokens_output", output_tokens)
        self.increment("api_cost_usd", cost)
        
        # Provider/model specific
        self.increment(f"api_requests_{provider}_total")
        self.increment(f"api_requests_{model}_total")
        self.timer(f"api_latency_{provider}_ms", latency_ms)
        
        if not success:
            self.increment("api_errors_total")
            self.increment(f"api_errors_{provider}_total")
        
        if complexity:
            self.histogram("complexity_score", complexity)
    
    def record_cache_event(self, event_type: str, size_bytes: int = None):
        """Record cache event"""
        if event_type == "hit":
            self.increment("cache_hits")
        elif event_type == "miss":
            self.increment("cache_misses")
        elif event_type == "eviction":
            self.increment("cache_evictions")
        
        if size_bytes:
            self.gauge("cache_size_bytes", size_bytes)
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        """Get all metrics with statistics"""
        with self.lock:
            return {
                name: metric.get_stats() 
                for name, metric in self.metrics.items()
            }
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        all_metrics = self.get_all_metrics()
        
        # Calculate key metrics
        total_requests = all_metrics.get('api_requests_total', {}).get('total', 0)
        total_errors = all_metrics.get('api_errors_total', {}).get('total', 0)
        
        return {
            'total_requests': total_requests,
            'error_rate': total_errors / total_requests if total_requests > 0 else 0,
            'avg_latency_ms': all_metrics.get('api_latency_ms', {}).get('mean', 0),
            'total_cost': all_metrics.get('api_cost_usd', {}).get('total', 0),
            'cache_hit_rate': self._calculate_cache_hit_rate(),
            'total_tokens': (
                all_metrics.get('api_tokens_input', {}).get('total', 0) +
                all_metrics.get('api_tokens_output', {}).get('total', 0)
            )
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        all_metrics = self.get_all_metrics()
        hits = all_metrics.get('cache_hits', {}).get('total', 0)
        misses = all_metrics.get('cache_misses', {}).get('total', 0)
        total = hits + misses
        
        return hits / total if total > 0 else 0
    
    def _export_worker(self):
        """Background thread to export metrics"""
        while True:
            time.sleep(self.export_interval)
            
            try:
                # Export to logger
                if self.logger:
                    summary = self.get_summary()
                    self.logger.info("Metrics export", extra={
                        'metrics_summary': summary,
                        'timestamp': datetime.now().isoformat()
                    })
                
                # Could also export to Prometheus, CloudWatch, etc.
                self._export_to_monitoring()
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Metrics export error: {e}")
    
    def _export_to_monitoring(self):
        """Export metrics to monitoring system (Prometheus, CloudWatch, etc.)"""
        # This would integrate with your monitoring system
        # Example: push to Prometheus pushgateway, CloudWatch, etc.
        pass

# ====================
# PRODUCTION RATE LIMITER
# ====================

class RateLimitWindow:
    """Sliding window for rate limiting"""
    
    def __init__(self, window_seconds: int, max_requests: int):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.requests = deque()
        self.lock = threading.Lock()
    
    def allow_request(self) -> Tuple[bool, int, float]:
        """
        Check if request is allowed
        
        Returns:
            (allowed, remaining_requests, reset_time)
        """
        with self.lock:
            now = time.time()
            
            # Remove old requests outside window
            cutoff = now - self.window_seconds
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            # Check if allowed
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                remaining = self.max_requests - len(self.requests)
                reset_time = self.requests[0] + self.window_seconds if self.requests else now
                return True, remaining, reset_time
            
            # Calculate when next request will be allowed
            reset_time = self.requests[0] + self.window_seconds
            return False, 0, reset_time

class ProductionRateLimiter:
    """Production rate limiter with per-provider and per-model limits"""
    
    def __init__(self, config: Dict = None, logger: ProductionLogger = None,
                 db: ProductionDB = None):
        """
        Initialize rate limiter
        
        Args:
            config: Rate limit configuration
        """
        self.logger = logger
        self.db = db
        
        # Default rate limits (requests per minute)
        default_config = {
            'global': {'window': 60, 'max_requests': 100},
            'providers': {
                'openai': {'window': 60, 'max_requests': 60},
                'anthropic': {'window': 60, 'max_requests': 50},
                'groq': {'window': 60, 'max_requests': 30},
                'google': {'window': 60, 'max_requests': 60},
                'deepseek': {'window': 60, 'max_requests': 100},
                'ollama': {'window': 60, 'max_requests': 1000}
            },
            'models': {
                'gpt-4o': {'window': 60, 'max_requests': 20},
                'claude-3-opus': {'window': 60, 'max_requests': 15}
            }
        }
        
        self.config = config or default_config
        self.windows: Dict[str, RateLimitWindow] = {}
        self.lock = threading.Lock()
        
        # Initialize windows
        self._init_windows()
        
        # Backoff strategy
        self.backoff_times: Dict[str, float] = {}
        self.max_backoff = 60  # Maximum backoff in seconds
    
    def _init_windows(self):
        """Initialize rate limit windows"""
        # Global window
        global_config = self.config.get('global', {})
        self.windows['global'] = RateLimitWindow(
            global_config.get('window', 60),
            global_config.get('max_requests', 100)
        )
        
        # Provider windows
        for provider, limits in self.config.get('providers', {}).items():
            self.windows[f'provider:{provider}'] = RateLimitWindow(
                limits.get('window', 60),
                limits.get('max_requests', 60)
            )
        
        # Model windows
        for model, limits in self.config.get('models', {}).items():
            self.windows[f'model:{model}'] = RateLimitWindow(
                limits.get('window', 60),
                limits.get('max_requests', 60)
            )
    
    def check_rate_limit(self, provider: str, model: str) -> Tuple[bool, Dict]:
        """
        Check if request is allowed under rate limits
        
        Returns:
            (allowed, limit_info)
        """
        with self.lock:
            # Check global limit
            global_allowed, global_remaining, global_reset = self.windows['global'].allow_request()
            
            if not global_allowed:
                self._handle_limit_exceeded('global', global_remaining, global_reset)
                return False, {
                    'limited_by': 'global',
                    'remaining': global_remaining,
                    'reset_time': global_reset,
                    'retry_after': global_reset - time.time()
                }
            
            # Check provider limit
            provider_key = f'provider:{provider}'
            if provider_key in self.windows:
                provider_allowed, provider_remaining, provider_reset = \
                    self.windows[provider_key].allow_request()
                
                if not provider_allowed:
                    self._handle_limit_exceeded(provider, provider_remaining, provider_reset)
                    return False, {
                        'limited_by': provider,
                        'remaining': provider_remaining,
                        'reset_time': provider_reset,
                        'retry_after': provider_reset - time.time()
                    }
            
            # Check model limit
            model_key = f'model:{model}'
            if model_key in self.windows:
                model_allowed, model_remaining, model_reset = \
                    self.windows[model_key].allow_request()
                
                if not model_allowed:
                    self._handle_limit_exceeded(model, model_remaining, model_reset)
                    return False, {
                        'limited_by': model,
                        'remaining': model_remaining,
                        'reset_time': model_reset,
                        'retry_after': model_reset - time.time()
                    }
            
            # All limits passed
            return True, {
                'allowed': True,
                'global_remaining': global_remaining
            }
    
    def _handle_limit_exceeded(self, limited_by: str, remaining: int, reset_time: float):
        """Handle rate limit exceeded"""
        # Log event
        if self.logger:
            self.logger.log_rate_limit(limited_by, remaining, 
                                      datetime.fromtimestamp(reset_time).isoformat())
        
        # Record in database
        if self.db:
            self.db.log_rate_limit_event(
                provider=limited_by if limited_by != 'global' else 'system',
                limit_type='rate_limit',
                remaining=remaining,
                reset_time=datetime.fromtimestamp(reset_time),
                action='blocked'
            )
    
    def wait_if_limited(self, provider: str, model: str, max_wait: float = 5.0) -> bool:
        """
        Wait if rate limited
        
        Returns:
            True if request can proceed, False if wait exceeded max_wait
        """
        allowed, info = self.check_rate_limit(provider, model)
        
        if allowed:
            return True
        
        retry_after = info.get('retry_after', 0)
        
        if retry_after > max_wait:
            return False
        
        if self.logger:
            self.logger.info(f"Rate limited, waiting {retry_after:.2f}s")
        
        time.sleep(retry_after)
        return True
    
    def apply_backoff(self, provider: str, error: Exception = None):
        """Apply exponential backoff for a provider"""
        key = f'backoff:{provider}'
        
        # Calculate next backoff time
        current_backoff = self.backoff_times.get(key, 0)
        
        if current_backoff == 0:
            next_backoff = 1  # Start with 1 second
        else:
            next_backoff = min(current_backoff * 2, self.max_backoff)
        
        self.backoff_times[key] = next_backoff
        
        if self.logger:
            self.logger.warning(f"Applying backoff for {provider}: {next_backoff}s", 
                              extra={'error': str(error) if error else None})
        
        time.sleep(next_backoff)
    
    def reset_backoff(self, provider: str):
        """Reset backoff for a provider after successful request"""
        key = f'backoff:{provider}'
        if key in self.backoff_times:
            del self.backoff_times[key]
    
    def get_limits_info(self) -> Dict:
        """Get current rate limit information"""
        info = {}
        
        for key, window in self.windows.items():
            with window.lock:
                now = time.time()
                cutoff = now - window.window_seconds
                
                # Clean old requests
                while window.requests and window.requests[0] < cutoff:
                    window.requests.popleft()
                
                current_usage = len(window.requests)
                
                info[key] = {
                    'current': current_usage,
                    'max': window.max_requests,
                    'window_seconds': window.window_seconds,
                    'usage_percentage': (current_usage / window.max_requests) * 100
                }
        
        return info
    
    def update_limits(self, provider: str = None, model: str = None, 
                     max_requests: int = None, window: int = None):
        """Dynamically update rate limits"""
        with self.lock:
            if provider:
                key = f'provider:{provider}'
                if key in self.windows:
                    if max_requests:
                        self.windows[key].max_requests = max_requests
                    if window:
                        self.windows[key].window_seconds = window
            
            if model:
                key = f'model:{model}'
                if key in self.windows:
                    if max_requests:
                        self.windows[key].max_requests = max_requests
                    if window:
                        self.windows[key].window_seconds = window
            
            if self.logger:
                self.logger.info("Rate limits updated", extra={
                    'provider': provider,
                    'model': model,
                    'max_requests': max_requests,
                    'window': window
                })

# ====================
# CONFIGURAZIONE BASE (original)
# ====================

@dataclass
class ModelConfig:
    """Configurazione completa modello - QUALITATIVA"""
    provider: str
    model_id: str
    display_name: str
    cost_per_1k_input: float
    cost_per_1k_output: float
    tier: str  # expert, professional, competent, basic, minimal
    speed: str  # ultra_fast, fast, medium, slow
    context_window: int
    best_for: List[str]  # Lista qualitativa di use cases
    avoid_for: List[str]  # Cosa evitare
    requires_key: bool
    size_gb: float = 0.0
    complexity_range: Tuple[int, int] = (1, 10)  # Range di complessità adatto
    personality: str = ""  # Stile di output

# [Original MODEL_CATALOG remains the same]
MODEL_CATALOG = {
    # OPENAI
    "gpt-4o": ModelConfig(
        provider="openai",
        model_id="gpt-4o",
        display_name="GPT-4 Optimized 🌟",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        tier="expert",
        speed="medium",
        context_window=128000,
        best_for=[
            "complex reasoning and logic problems",
            "advanced coding and debugging",
            "deep technical analysis",
            "research-level questions",
            "mathematical proofs"
        ],
        avoid_for=[
            "simple greetings",
            "basic arithmetic",
            "when speed is critical"
        ],
        complexity_range=(7, 10),
        personality="Thorough, analytical, and precise",
        requires_key=True,
        size_gb=100
    ),
    
    "gpt-4o-mini": ModelConfig(
        provider="openai",
        model_id="gpt-4o-mini",
        display_name="GPT-4 Mini ⭐",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        tier="professional",
        speed="fast",
        context_window=128000,
        best_for=[
            "general purpose queries",
            "cost-effective solutions",
            "moderate complexity tasks",
            "quick analysis",
            "standard coding tasks"
        ],
        avoid_for=[
            "PhD-level research",
            "extremely complex reasoning"
        ],
        complexity_range=(4, 8),
        personality="Balanced and efficient",
        requires_key=True,
        size_gb=20
    ),
    
    "gpt-3.5-turbo": ModelConfig(
        provider="openai",
        model_id="gpt-3.5-turbo",
        display_name="GPT-3.5 Turbo",
        cost_per_1k_input=0.0005,
        cost_per_1k_output=0.0015,
        tier="competent",
        speed="fast",
        context_window=16385,
        best_for=[
            "simple queries",
            "quick responses",
            "basic explanations",
            "light conversation",
            "simple coding"
        ],
        avoid_for=[
            "complex reasoning",
            "advanced analysis",
            "mathematical proofs"
        ],
        complexity_range=(1, 5),
        personality="Quick and straightforward",
        requires_key=True,
        size_gb=10
    ),
    
    # ANTHROPIC
    "claude-3-opus": ModelConfig(
        provider="anthropic",
        model_id="claude-3-opus-20240229",
        display_name="Claude 3 Opus 🌟",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        tier="expert",
        speed="slow",
        context_window=200000,
        best_for=[
            "complex analysis requiring nuance",
            "creative writing and storytelling",
            "academic research",
            "ethical reasoning",
            "long-form content generation"
        ],
        avoid_for=[
            "quick simple answers",
            "when speed matters",
            "basic calculations"
        ],
        complexity_range=(8, 10),
        personality="Thoughtful, nuanced, and articulate",
        requires_key=True,
        size_gb=150
    ),
    
    "claude-3-sonnet": ModelConfig(
        provider="anthropic",
        model_id="claude-3-5-sonnet-20241022",
        display_name="Claude 3.5 Sonnet ⭐",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        tier="professional",
        speed="medium",
        context_window=200000,
        best_for=[
            "balanced performance tasks",
            "coding with good explanations",
            "detailed reasoning",
            "technical documentation",
            "moderate creative tasks"
        ],
        avoid_for=[
            "ultra-fast responses needed",
            "simple yes/no questions"
        ],
        complexity_range=(5, 9),
        personality="Clear and comprehensive",
        requires_key=True,
        size_gb=70
    ),
    
    "claude-3-haiku": ModelConfig(
        provider="anthropic",
        model_id="claude-3-haiku-20240307",
        display_name="Claude 3 Haiku ✓",
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        tier="competent",
        speed="ultra_fast",
        context_window=200000,
        best_for=[
            "simple tasks needing speed",
            "quick summaries",
            "basic Q&A",
            "light conversation",
            "simple data processing"
        ],
        avoid_for=[
            "complex reasoning",
            "creative writing",
            "deep analysis"
        ],
        complexity_range=(1, 5),
        personality="Concise and efficient",
        requires_key=True,
        size_gb=10
    ),
    
    # GROQ
    "llama-3.3-70b": ModelConfig(
        provider="groq",
        model_id="llama-3.3-70b-versatile",
        display_name="Llama 3.3 70B (Groq) ⭐",
        cost_per_1k_input=0.00059,
        cost_per_1k_output=0.00079,
        tier="professional",
        speed="ultra_fast",
        context_window=128000,
        best_for=[
            "fast high-quality responses",
            "general purpose with speed",
            "real-time applications",
            "conversational AI",
            "quick analysis"
        ],
        avoid_for=[
            "extremely complex research",
            "tasks requiring slow deliberation"
        ],
        complexity_range=(4, 8),
        personality="Fast and reliable",
        requires_key=True,
        size_gb=70
    ),
    
    "mixtral-8x7b": ModelConfig(
        provider="groq",
        model_id="mixtral-8x7b-32768",
        display_name="Mixtral 8x7B (Groq) ✓",
        cost_per_1k_input=0.00024,
        cost_per_1k_output=0.00024,
        tier="competent",
        speed="ultra_fast",
        context_window=32768,
        best_for=[
            "ultra-fast responses",
            "general conversation",
            "basic coding",
            "simple explanations",
            "real-time chat"
        ],
        avoid_for=[
            "complex analysis",
            "creative writing",
            "deep technical topics"
        ],
        complexity_range=(2, 6),
        personality="Quick and practical",
        requires_key=True,
        size_gb=45
    ),
    
    # GOOGLE
    "gemini-2.0-flash": ModelConfig(
        provider="google",
        model_id="gemini-2.0-flash-exp",
        display_name="Gemini 2.0 Flash ⭐",
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        tier="professional",
        speed="ultra_fast",
        context_window=1048576,
        best_for=[
            "extremely long context tasks",
            "document analysis",
            "multimodal tasks",
            "free API usage",
            "bulk processing"
        ],
        avoid_for=[
            "tasks requiring specific formatting",
            "when consistency is critical"
        ],
        complexity_range=(3, 7),
        personality="Versatile and fast",
        requires_key=True,
        size_gb=20
    ),
    
    # DEEPSEEK
    "deepseek-chat": ModelConfig(
        provider="deepseek",
        model_id="deepseek-chat",
        display_name="DeepSeek Chat ⭐",
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        tier="professional",
        speed="fast",
        context_window=64000,
        best_for=[
            "coding and debugging",
            "mathematical problems",
            "Chinese language tasks",
            "technical explanations",
            "algorithmic thinking"
        ],
        avoid_for=[
            "creative writing",
            "casual conversation",
            "Western cultural references"
        ],
        complexity_range=(4, 8),
        personality="Technical and precise",
        requires_key=True,
        size_gb=20
    ),
    
    # O1 MODELS (se vuoi aggiungerli)
    "o1": ModelConfig(
        provider="openai",
        model_id="o1",
        display_name="GPT-o1 (Reasoning) 🌟🌟",
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.060,
        tier="expert",
        speed="slow",
        context_window=200000,
        best_for=[
            "complex multi-step reasoning",
            "mathematical proofs",
            "puzzle solving",
            "code debugging with deep analysis",
            "research-level problems"
        ],
        avoid_for=[
            "simple questions",
            "when quick response needed",
            "casual conversation"
        ],
        complexity_range=(8, 10),
        personality="Methodical reasoning with chain-of-thought",
        requires_key=True,
        size_gb=200
    ),
    
    "o1-mini": ModelConfig(
        provider="openai",
        model_id="o1-mini",
        display_name="GPT-o1 Mini (Fast Reasoning) ⭐",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.012,
        tier="professional",
        speed="medium",
        context_window=128000,
        best_for=[
            "moderate reasoning tasks",
            "coding problems",
            "math problems",
            "logical puzzles",
            "structured thinking"
        ],
        avoid_for=[
            "creative writing",
            "open-ended generation",
            "simple lookups"
        ],
        complexity_range=(5, 8),
        personality="Structured reasoning approach",
        requires_key=True,
        size_gb=50
    )
}

# [All original provider implementations remain the same]
# ====================
# PROVIDER BASE CLASSES
# ====================

class LLMProvider(ABC):
    """Classe base per tutti i provider LLM"""
    
    @abstractmethod
    def is_available(self) -> bool:
        """Controlla se il provider è disponibile"""
        pass
    
    @abstractmethod
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        """Invia messaggi e ricevi risposta"""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """Lista modelli disponibili"""
        pass

# ====================
# PROVIDER IMPLEMENTATIONS (original)
# ====================

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000)
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def list_models(self) -> List[str]:
        if not self.is_available():
            return []
        return ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        system = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        data = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", 2000),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        if system:
            data["system"] = system
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["content"][0]["text"]
    
    def list_models(self) -> List[str]:
        if not self.is_available():
            return []
        return ["claude-3-opus-20240229", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"]

class GroqProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000)
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def list_models(self) -> List[str]:
        if not self.is_available():
            return []
        return ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]

class GoogleProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.base_url = "https://generativelanguage.googleapis.com/v1beta"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        contents = []
        for msg in messages:
            contents.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["content"]}]
            })
        
        url = f"{self.base_url}/models/{model}:generateContent?key={self.api_key}"
        
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 2000)
            }
        }
        
        response = requests.post(url, json=data)
        response.raise_for_status()
        
        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
    
    def list_models(self) -> List[str]:
        if not self.is_available():
            return []
        return ["gemini-2.0-flash-exp", "gemini-1.5-pro"]

class DeepSeekProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 2000)
        }
        
        response = requests.post(self.base_url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["choices"][0]["message"]["content"]
    
    def list_models(self) -> List[str]:
        if not self.is_available():
            return []
        return ["deepseek-chat"]

class OllamaProvider(LLMProvider):
    def __init__(self):
        self.client = ollama if OLLAMA_AVAILABLE else None

    def is_ollama_model_name(self, name: str) -> bool:
        """Verifica se è un nome valido per Ollama"""
        return ":" in name or not name.startswith("ollama_")

    def fix_model_name(self, model: str) -> str:
        """Converte model_id in nome Ollama valido"""
        if model.startswith("ollama_"):
            # ollama_gemma3_12b → gemma3:12b
            fixed = model.replace("ollama_", "")
            
            # Trova dove mettere il ":"
            # Esempi: gemma3_12b → gemma3:12b
            #         gemma3_270m → gemma3:270m  
            parts = fixed.split("_", 1)
            if len(parts) == 2 and ":" not in fixed:
                return f"{parts[0]}:{parts[1]}"
            return fixed.replace("_", ":")
        return model
    
    def is_available(self) -> bool:
        if not OLLAMA_AVAILABLE:
            return False
        try:
            self.client.list()
            return True
        except:
            return False
    
    
    def chat(self, messages: List[Dict], model: str, **kwargs) -> str:
        if not self.client:
            raise Exception("Ollama not available")
        model = self.fix_model_name(model)
        response = self.client.chat(
            model=model,
            messages=messages,
            options={
                'temperature': kwargs.get("temperature", 0.7),
                'num_predict': kwargs.get("max_tokens", 2000)
            }
        )
        return response['message']['content']
    
    def list_models(self) -> List[str]:
        if not self.is_available():
            return []
        
        try:
            models = self.client.list()
            return [m['model'] for m in models['models']]
        except:
            return []

# [All original context, complexity evaluator, and auto-discovery code remains the same]
# ====================
# CONTEXT MANAGER (original)
# ====================

class ConversationContext:
    """Gestisce il contesto unificato della conversazione"""
    
    def __init__(self, max_tokens: int = 8000):
        self.messages: List[Dict] = []
        self.max_tokens = max_tokens
        self.model_history: List[str] = []
        self.token_count = 0
    
    def add_message(self, role: str, content: str, model_used: str = None):
        """Aggiunge messaggio mantenendo il contesto"""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "model_used": model_used
        }
        
        self.messages.append(msg)
        if model_used:
            self.model_history.append(model_used)
        
        # Stima tokens
        self.token_count += len(content.split()) * 1.3
        
        # Trim se necessario
        self._trim_context()
    
    def _trim_context(self):
        """Riduce il contesto se troppo lungo"""
        if self.token_count > self.max_tokens:
            system = [m for m in self.messages if m["role"] == "system"]
            recent = self.messages[-10:]
            
            self.messages = system + recent
            self.token_count = sum(len(m["content"].split()) * 1.3 for m in self.messages)
    
    def get_messages_for_api(self) -> List[Dict]:
        """Ritorna messaggi nel formato per API"""
        return [
            {"role": m["role"], "content": m["content"]} 
            for m in self.messages
        ]
    
    def get_summary(self) -> str:
        """Genera summary del contesto"""
        return f"Messages: {len(self.messages)}, Models used: {len(set(self.model_history))}, Tokens: ~{int(self.token_count)}"

# [All original auto-discovery and complexity evaluator code remains the same]
# ====================
# AUTO-DISCOVERY SYSTEM (original)
# ====================

class ModelAutoDiscovery:
    """Sistema di auto-discovery e benchmarking per modelli Ollama"""
    
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.test_suite = self._create_test_suite()
        self.benchmark_cache_file = Path.home() / ".llm-use" / "benchmarks.json"
        self.benchmark_cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.benchmarks = self._load_benchmarks()
        
        # INIZIALIZZA QUALITY EVALUATOR AGENTICO
        if OLLAMA_AVAILABLE:
            try:
                ollama_provider = OllamaProvider()
                if ollama_provider.is_available():
                    self.quality_evaluator = AgenticQualityEvaluator(
                        ollama_provider, 
                        verbose=verbose
                    )
                    if verbose:
                        print("   🤖 Using AGENTIC quality evaluator")
                else:
                    self.quality_evaluator = None
            except:
                self.quality_evaluator = None
        else:
            self.quality_evaluator = None
    
    def _create_test_suite(self) -> Dict[str, Dict]:
        """Suite di test COMPLETA per differenziare OGNI livello di capacità"""
        return {
            # ========== LEVEL 1: ULTRA BASIC (tutti dovrebbero passare) ==========
            "instant_response": {
                "prompt": "Hi",
                "category": "instant",
                "expected_quality": "any_response",
                "max_tokens": 10,
                "complexity": 1
            },
            "yes_no": {
                "prompt": "Is 5 greater than 3? Answer only yes or no.",
                "category": "binary",
                "expected_quality": "correct_binary",
                "correct_answer": "yes",
                "max_tokens": 5,
                "complexity": 1
            },
            
            # ========== LEVEL 2: SIMPLE (modelli basic dovrebbero gestire) ==========
            "simple_math": {
                "prompt": "Calculate: 8 + 7",
                "category": "arithmetic",
                "expected_quality": "exact_number",
                "correct_answer": "15",
                "max_tokens": 10,
                "complexity": 2
            },
            "basic_question": {
                "prompt": "What color is the sky on a clear day?",
                "category": "factual",
                "expected_quality": "correct_fact",
                "correct_answer": "blue",
                "max_tokens": 20,
                "complexity": 2
            },
            
            # ========== LEVEL 3-4: STANDARD (modelli competent iniziano qui) ==========
            "word_problem": {
                "prompt": "If John has 3 apples and Mary gives him 5 more, how many apples does John have?",
                "category": "word_math",
                "expected_quality": "correct_reasoning",
                "correct_answer": "8",
                "max_tokens": 50,
                "complexity": 3
            },
            "explain_concept": {
                "prompt": "Explain what recursion is in programming using simple words.",
                "category": "explanation",
                "expected_quality": "clear_explanation",
                "max_tokens": 100,
                "complexity": 4
            },
            "simple_code": {
                "prompt": "Write a Python function that returns True if a number is even, False otherwise. Only the function.",
                "category": "coding_basic",
                "expected_quality": "working_code",
                "max_tokens": 50,
                "complexity": 4
            },
            
            # ========== LEVEL 5-6: INTERMEDIATE (modelli professional iniziano qui) ==========
            "logic_puzzle": {
                "prompt": "All cats have tails. Fluffy is a cat. What can we conclude about Fluffy?",
                "category": "logic",
                "expected_quality": "valid_logic",
                "correct_answer": "has a tail",
                "max_tokens": 50,
                "complexity": 5
            },
            "algorithm": {
                "prompt": "Write a Python function to check if a string is a palindrome. Handle edge cases.",
                "category": "coding_medium",
                "expected_quality": "complete_solution",
                "max_tokens": 150,
                "complexity": 6
            },
            "multi_step_math": {
                "prompt": "If a train travels 60 km/h for 2.5 hours, then 80 km/h for 1.5 hours, what's the total distance?",
                "category": "multi_step",
                "expected_quality": "correct_calculation",
                "correct_answer": "270",
                "max_tokens": 100,
                "complexity": 5
            },
            
            # ========== LEVEL 7-8: ADVANCED (solo modelli forti) ==========
            "tricky_reasoning": {
                "prompt": "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much does the ball cost? Think step by step.",
                "category": "tricky_math",
                "expected_quality": "correct_tricky",
                "correct_answer": "0.05",
                "max_tokens": 200,
                "complexity": 7
            },
            "code_optimization": {
                "prompt": "This Python code is O(n²): def has_duplicate(lst): for i in range(len(lst)): for j in range(i+1, len(lst)): if lst[i]==lst[j]: return True; return False. Rewrite it to be O(n).",
                "category": "optimization",
                "expected_quality": "optimized_code",
                "max_tokens": 150,
                "complexity": 8
            },
            "complex_analysis": {
                "prompt": "Compare transformer architecture to LSTM for NLP tasks. Focus on attention mechanism vs sequential processing.",
                "category": "technical_deep",
                "expected_quality": "expert_analysis",
                "max_tokens": 200,
                "complexity": 8
            },
            
            # ========== LEVEL 9-10: EXPERT (solo modelli top-tier) ==========
            "advanced_math": {
                "prompt": "Prove that the square root of 2 is irrational.",
                "category": "proof",
                "expected_quality": "valid_proof",
                "max_tokens": 300,
                "complexity": 9
            },
            "system_design": {
                "prompt": "Design a distributed cache system that handles 1 million requests per second with sub-millisecond latency. List key components.",
                "category": "architecture",
                "expected_quality": "professional_design",
                "max_tokens": 300,
                "complexity": 9
            },
            "creative_complex": {
                "prompt": "Write a haiku about quantum entanglement that accurately reflects the physics.",
                "category": "creative_technical",
                "expected_quality": "creative_and_accurate",
                "max_tokens": 50,
                "complexity": 8
            },
            "research_question": {
                "prompt": "What are the main challenges in achieving artificial general intelligence and which approach (symbolic AI, connectionist, hybrid) is most promising?",
                "category": "research",
                "expected_quality": "research_level",
                "max_tokens": 400,
                "complexity": 10
            },
            
            # ========== SPEED TESTS (differenzia per velocità) ==========
            "speed_test_instant": {
                "prompt": "Reply with 'ok'",
                "category": "speed",
                "expected_quality": "instant",
                "correct_answer": "ok",
                "max_tokens": 5,
                "complexity": 1
            },
            "speed_test_quick": {
                "prompt": "Count from 1 to 5",
                "category": "speed",
                "expected_quality": "quick_list",
                "max_tokens": 20,
                "complexity": 1
            },
            
            # ========== STRESS TESTS (trova i limiti) ==========
            "long_context": {
                "prompt": "Summarize this in one sentence: " + "The weather is nice. " * 50,
                "category": "context_handling",
                "expected_quality": "coherent_summary",
                "max_tokens": 50,
                "complexity": 4
            },
            "error_recovery": {
                "prompt": "Fix this code: def func(x) return x * 2 if x > else x",
                "category": "error_handling",
                "expected_quality": "fixed_code",
                "max_tokens": 100,
                "complexity": 6
            },
            "nuanced_question": {
                "prompt": "Is AI dangerous? Provide a balanced view in 2 sentences.",
                "category": "nuance",
                "expected_quality": "balanced_response",
                "max_tokens": 100,
                "complexity": 7
            },
            
            # ========== LANGUAGE UNDERSTANDING (test vero understanding) ==========
            "ambiguity": {
                "prompt": "The chicken is ready to eat. Who is eating?",
                "category": "language_understanding",
                "expected_quality": "ambiguity_recognition",
                "max_tokens": 50,
                "complexity": 6
            },
            "instruction_following": {
                "prompt": "Write exactly 3 words about space. No more, no less.",
                "category": "instruction",
                "expected_quality": "exact_following",
                "max_tokens": 10,
                "complexity": 3
            },
            "context_awareness": {
                "prompt": "I have 3 apples. I eat one. I buy 5 more. My friend takes 2. How many do I have?",
                "category": "context_tracking",
                "expected_quality": "correct_tracking",
                "correct_answer": "5",
                "max_tokens": 50,
                "complexity": 5
            }
        }
    
    def _load_benchmarks(self) -> Dict:
        """Carica benchmark salvati"""
        if self.benchmark_cache_file.exists():
            try:
                with open(self.benchmark_cache_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {}
    
    def _save_benchmarks(self):
        """Salva benchmark su file"""
        with open(self.benchmark_cache_file, 'w') as f:
            json.dump(self.benchmarks, f, indent=2)
    
    def discover_and_benchmark_ollama_models(self, force_benchmark: bool = False) -> Dict[str, ModelConfig]:
        """Scopre e testa TUTTI i modelli Ollama installati"""
        discovered_models = {}
        
        if not OLLAMA_AVAILABLE:
            if self.verbose:
                print("⚠️ Ollama not available for discovery")
            return discovered_models
        
        try:
            # Ottieni lista modelli installati
            ollama_models = ollama.list()
            
            if self.verbose:
                print(f"\n🔍 Discovering Ollama models...")
                print(f"   Found {len(ollama_models['models'])} models installed\n")
            
            for model_info in ollama_models['models']:
                model_name = model_info['model']
                model_size_bytes = model_info.get('size', 0)
                model_size_gb = model_size_bytes / (1024**3) if model_size_bytes > 0 else 0
                
                model_base = model_name.split(':')[0]
                model_id = f"ollama_{model_base}_{model_name.replace(':', '_')}"
                
                # Controlla cache benchmark
                use_cache = False
                if not force_benchmark and model_id in self.benchmarks:
                    benchmark_age_hours = (time.time() - self.benchmarks[model_id].get('timestamp', 0)) / 3600
                    if benchmark_age_hours < 24:
                        use_cache = True
                        if self.verbose:
                            print(f"✓ {model_name}: Using cached profile (age: {benchmark_age_hours:.1f}h)")
                
                if use_cache:
                    # Usa benchmark dalla cache
                    benchmark_results = self.benchmarks[model_id]
                    config = self._create_config_from_benchmark(
                        model_name, 
                        benchmark_results,
                        model_size_gb
                    )
                    discovered_models[model_id] = config
                    continue
                
                # Esegui benchmark con profiling qualitativo
                if self.verbose:
                    print(f"\n🧪 Testing {model_name} ({model_size_gb:.1f}GB)")
                
                benchmark_results = self._benchmark_model(model_name)
                benchmark_results['size_gb'] = model_size_gb
                
                # Mostra risultati
                if self.verbose:
                    print(f"\n   📊 Profile Summary:")
                    print(f"     Tier: {benchmark_results.get('tier', 'unknown')}")
                    print(f"     Speed: {benchmark_results.get('speed_profile', 'unknown')}")
                    print(f"     Complexity range: {benchmark_results.get('complexity_min', 1)}-{benchmark_results.get('complexity_max', 10)}")
                    print(f"     Avg response time: {benchmark_results.get('avg_response_time', 0):.1f}s")
                
                # Salva risultati
                self.benchmarks[model_id] = benchmark_results
                self._save_benchmarks()
                
                # Crea configurazione
                config = self._create_config_from_benchmark(model_name, benchmark_results, model_size_gb)
                discovered_models[model_id] = config
                
                if self.verbose:
                    self._print_model_summary(model_name, config, benchmark_results)
                    print()
            
            # 🔥 FIX CRITICO: AGGIUNGI I MODELLI A available_models!
            if discovered_models:
                if self.verbose:
                    print(f"\n✅ Registering {len(discovered_models)} models...")
                
                # Inizializza available_models se non esiste
                if not hasattr(self, 'available_models'):
                    self.available_models = {}
                
                # Aggiungi tutti i modelli scoperti
                for model_id, config in discovered_models.items():
                    self.available_models[model_id] = config
                    if self.verbose:
                        print(f"   ✓ Registered: {config.display_name}")
                
                if self.verbose:
                    print(f"\n   📊 Total models available: {len(self.available_models)}")
            
            # SUMMARY FINALE
            if self.verbose and discovered_models:
                self._print_discovery_summary(discovered_models)
            
            return discovered_models
            
        except Exception as e:
            if self.verbose:
                print(f"\n⚠️ Error discovering Ollama models: {e}")
                import traceback
                traceback.print_exc()
            
            # Anche in caso di errore, ritorna quello che hai scoperto
            if discovered_models:
                # Prova ad aggiungere comunque i modelli scoperti
                if not hasattr(self, 'available_models'):
                    self.available_models = {}
                self.available_models.update(discovered_models)
            
            return discovered_models
    
    def _print_model_summary(self, model_name: str, config: ModelConfig, benchmark: Dict):
        """Stampa summary del modello - QUALITATIVO"""
        print(f"   ✓ {config.display_name}")
        print(f"     Tier: {config.tier.upper()} | Speed: {config.speed}")
        print(f"     Complexity range: {config.complexity_range[0]}-{config.complexity_range[1]}")
        
        best_for = ', '.join(config.best_for[:3])
        print(f"     Best for: {best_for}")
        
        if config.personality:
            print(f"     Style: {config.personality[:80]}")

    def _print_discovery_summary(self, discovered_models: Dict[str, ModelConfig]):
        """Stampa summary finale della discovery"""
        print(f"\n{'='*70}")
        print(f"📊 DISCOVERY SUMMARY - {len(discovered_models)} MODELS PROFILED")
        print(f"{'='*70}\n")
        
        # Raggruppa per tier
        by_tier = {}
        for model_id, config in discovered_models.items():
            # Prendi il tier direttamente dalla config
            tier = config.tier if hasattr(config, 'tier') else "unknown"
            
            if tier not in by_tier:
                by_tier[tier] = []
            
            # Prepara info del modello
            complexity_range = getattr(config, 'complexity_range', (1, 10))
            model_info = {
                'name': config.display_name,
                'speed': config.speed,
                'complexity': f"{complexity_range[0]}-{complexity_range[1]}",
                'best_for': config.best_for[:2] if hasattr(config, 'best_for') else [],
                'size': config.size_gb
            }
            by_tier[tier].append(model_info)
        
        # Ordine dei tier
        tier_order = ["expert", "professional", "competent", "basic", "minimal", "unknown"]
        tier_descriptions = {
            "expert": "🌟🌟 EXPERT - Exceptional models for complex tasks",
            "professional": "⭐ PROFESSIONAL - Solid and reliable for most work",
            "competent": "✓ COMPETENT - Good for standard tasks",
            "basic": "• BASIC - Limited to simple tasks",
            "minimal": "· MINIMAL - Very basic capabilities",
            "unknown": "? UNKNOWN - Not yet profiled"
        }
        
        # Mostra per tier
        for tier in tier_order:
            if tier in by_tier:
                print(f"{tier_descriptions.get(tier, tier.upper())}:")
                print("-" * 70)
                
                # FIX: Ordina per dimensione usando la chiave del dizionario
                sorted_models = sorted(by_tier[tier], key=lambda x: x['size'], reverse=True)
                
                # FIX: Itera sui dizionari, non tuple
                for model in sorted_models:
                    # Prima riga: nome, speed, complexity range
                    print(f"  📦 {model['name']:<35} Speed: {model['speed']:<12} Complex: {model['complexity']:<5}")
                    
                    # Seconda riga: best for (se disponibile)
                    if model['best_for']:
                        best_for_str = ', '.join(model['best_for'][:2])
                        if len(best_for_str) > 60:
                            best_for_str = best_for_str[:57] + "..."
                        print(f"     Best: {best_for_str}")
                print()
        
        # Summary statistics
        print("="*70)
        print("📈 SUMMARY STATISTICS:")
        print("-"*70)
        
        # Conta modelli per tier
        tier_counts = {}
        for tier in by_tier:
            tier_counts[tier] = len(by_tier[tier])
        
        print("Distribution by tier:")
        for tier in tier_order:
            if tier in tier_counts:
                bar = "█" * tier_counts[tier]
                print(f"  {tier:<15} {bar} ({tier_counts[tier]})")
        
        # Speed distribution
        speed_counts = {}
        for model_id, config in discovered_models.items():
            speed = config.speed
            speed_counts[speed] = speed_counts.get(speed, 0) + 1
        
        print("\nDistribution by speed:")
        for speed in ["ultra_fast", "fast", "medium", "slow"]:
            if speed in speed_counts:
                bar = "▓" * speed_counts[speed]
                print(f"  {speed:<15} {bar} ({speed_counts[speed]})")
        
        # Complexity coverage
        print("\nComplexity coverage:")
        complexity_coverage = [False] * 10
        for model_id, config in discovered_models.items():
            if hasattr(config, 'complexity_range'):
                min_c, max_c = config.complexity_range
                for i in range(min_c-1, min(max_c, 10)):
                    complexity_coverage[i] = True
        
        coverage_str = ""
        for i in range(10):
            if complexity_coverage[i]:
                coverage_str += "█"
            else:
                coverage_str += "░"
        print(f"  1 2 3 4 5 6 7 8 9 10")
        print(f"  {' '.join(coverage_str)}")
        
        # Final message
        print("\n" + "="*70)
        if len(discovered_models) > 0:
            expert_count = tier_counts.get("expert", 0)
            prof_count = tier_counts.get("professional", 0)
            
            if expert_count > 0:
                print("✅ Excellent model diversity! You have expert models for complex tasks.")
            elif prof_count > 0:
                print("✅ Good model selection! Professional models available for solid work.")
            else:
                print("ℹ️ Basic model set. Consider adding larger models for complex tasks.")
        else:
            print("⚠️ No models discovered.")
        
        print("🎯 Ready for intelligent qualitative routing!")
        print("="*70 + "\n")
        
        # Mostra per tier
        tier_order = ["expert", "professional", "competent", "basic", "minimal", "unknown"]
        tier_emoji = {
            "expert": "🌟",
            "professional": "⭐",
            "competent": "✓",
            "basic": "○",
            "minimal": "·"
        }
        
        for tier in tier_order:
            if tier in by_tier:
                print(f"{tier_descriptions.get(tier, tier.upper())}:")
                print("-" * 70)
                
                sorted_models = sorted(by_tier[tier], key=lambda x: x['size'], reverse=True)
                
                for model in sorted_models:
                    # USA LE CHIAVI DEL DIZIONARIO!
                    print(f"  📦 {model['name']:<35} Speed: {model['speed']:<12} Complex: {model['complexity']:<5}")
                    
                    # Mostra best_for se disponibile
                    if model['best_for']:
                        best_for_str = ', '.join(model['best_for'][:2])
                        if len(best_for_str) > 60:
                            best_for_str = best_for_str[:57] + "..."
                        print(f"     Best: {best_for_str}")
                print()
        
        print(f"{'='*70}")
        print(f"✅ All models profiled and ready for intelligent routing!")
        print(f"{'='*70}\n")
    
    def _benchmark_model(self, model_name: str) -> Dict:
        """Crea un profilo basato su QUALITÀ REALE delle risposte"""
        
        if self.verbose:
            print(f"\n🤖 Testing QUALITY of {model_name}")
        
        test_responses = {}
        quality_scores = {}
        
        for test_name, test_config in self.test_suite.items():
            try:
                start = time.time()
                response = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": test_config["prompt"]}],
                    stream=False,
                    options={'temperature': 0.3}
                )
                elapsed = time.time() - start
                
                response_text = response['message']['content']
                
                # 🔥 VALUTA LA QUALITÀ DELLA RISPOSTA
                quality = self._evaluate_response_quality(
                    test_config["prompt"],
                    response_text,
                    test_config["category"],
                    elapsed
                )
                
                test_responses[test_name] = {
                    "prompt": test_config["prompt"],
                    "response": response_text,
                    "category": test_config["category"],
                    "time": elapsed,
                    "quality_score": quality  # SALVA IL PUNTEGGIO
                }
                
                quality_scores[test_config["category"]] = quality
                
                if self.verbose:
                    print(f"     {test_name}: Quality {quality:.1f}/10, Time {elapsed:.1f}s")
                    
            except Exception as e:
                if self.verbose:
                    print(f"     {test_name}: ❌ Failed - {e}")
                quality_scores[test_config["category"]] = 0
        
        # 🔥 DETERMINA CAPACITÀ BASATE SU QUALITÀ REALE
        profile = self._create_qualitative_profile(model_name, test_responses)
            # Aggiungi metriche aggregate
        avg_time = sum(r['time'] for r in test_responses.values() if r['time'] < 999) / len(test_responses)
        avg_quality = sum(r.get('quality', 0) for r in test_responses.values()) / len(test_responses)
        
        profile['avg_response_time'] = avg_time
        profile['avg_quality_score'] = avg_quality
        profile['timestamp'] = time.time()
        
        # Aggiungi speed profile basato su tempo
        if avg_time < 1:
            profile['speed_profile'] = "ultra_fast"
        elif avg_time < 3:
            profile['speed_profile'] = "fast"
        elif avg_time < 10:
            profile['speed_profile'] = "medium"
        else:
            profile['speed_profile'] = "slow"
        
        return profile
    
    def _evaluate_response_quality(self, prompt: str, response: str, category: str, response_time: float) -> float:
        """🤖 USA UN AGENT PER VALUTARE LA QUALITÀ"""
        
        evaluation_prompt = f"""<role>
    You are a Response Quality Evaluator. Score this response quality from 1-10.
    </role>

    <test>
    CATEGORY: {category}
    QUESTION: {prompt}
    RESPONSE: {response}
    RESPONSE TIME: {response_time:.1f}s
    </test>

    <evaluation_criteria>
    1. CORRECTNESS: Is the answer right?
    2. COMPLETENESS: Does it fully address the question?
    3. CLARITY: Is it well-expressed?
    4. APPROPRIATENESS: Is it suitable for the question type?

    For "{category}" tasks, focus on:
    - simple: Can it respond appropriately to greetings?
    - math: Is the calculation correct?
    - reasoning: Is the logic sound?
    - coding: Does the code work?
    - creativity: Is it actually creative?
    - technical: Is it accurate and detailed?
    </evaluation_criteria>

    Evaluate and output ONLY a number 1-10:"""

        try:
            eval_response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": evaluation_prompt}],
                stream=False,
                options={'temperature': 0.1}
            )
            
            # Estrai score
            import re
            text = eval_response['message']['content']
            numbers = re.findall(r'\b([1-9]|10)\b', text)
            if numbers:
                return float(numbers[-1])
            return 5.0
            
        except:
            return 5.0
    
    def _create_qualitative_profile(self, model_name: str, test_responses: Dict) -> Dict:
        """🎯 Profilazione DETERMINISTICA basata SOLO sui dati reali dei test"""
        
        if self.verbose:
            print(f"\n   📊 Analyzing test results for {model_name}...")
        
        # ANALISI DIRETTA DEI TEST - NO LLM AGENT
        good_complexities = []
        test_evidence = []
        all_tests = []
        
        for test_name, data in test_responses.items():
            complexity = self.test_suite.get(test_name, {}).get('complexity', 5)
            quality = data.get('quality', data.get('quality_score', 0))
            
            all_tests.append((test_name, complexity, quality))
            
            if quality >= 7:
                good_complexities.append(complexity)
                test_evidence.append((test_name, complexity, quality))
        
        # CALCOLA STATISTICHE
        avg_time = sum(r['time'] for r in test_responses.values()) / len(test_responses)
        avg_quality = sum(r.get('quality', 5) for r in test_responses.values()) / len(test_responses)
        
        # DETERMINA RANGE E TIER BASANDOSI SUI DATI
        if good_complexities:
            min_good = min(good_complexities)
            max_good = max(good_complexities)
            
            if self.verbose:
                print(f"   ✅ Tests passed (≥7/10): {len(good_complexities)}")
                print(f"      Complexity range: {min_good}-{max_good}")
                # Mostra top 5
                top_tests = sorted(test_evidence, key=lambda x: x[1], reverse=True)[:5]
                for test, comp, qual in top_tests:
                    print(f"      ✓ {test}: complexity {comp}, score {qual}/10")
            
            # ASSEGNA RANGE basato sul MASSIMO complexity passato
            if max_good >= 9:
                complexity_min, complexity_max = 8, 10
                tier = "expert"
                best_for = [
                    "Complex analysis and research",
                    "Expert-level problem solving",
                    "Advanced technical discussions",
                    "Challenging reasoning tasks"
                ]
                avoid_for = [
                    "Simple queries (use smaller model for efficiency)",
                    "Real-time applications if response time >5s"
                ]
                personality = "Sophisticated, thorough, and capable of handling complex topics"
                
            elif max_good >= 6:
                complexity_min, complexity_max = 5, 8
                tier = "professional"
                best_for = [
                    "Professional analysis and reasoning",
                    "Technical explanations and documentation",
                    "Moderate to complex problem solving",
                    "Detailed Q&A on various topics"
                ]
                avoid_for = [
                    "Cutting-edge research (needs expert model)",
                    "Simple queries (use smaller model for efficiency)"
                ]
                personality = "Professional and comprehensive in responses"
                
            elif max_good >= 4:
                complexity_min, complexity_max = 3, 5
                tier = "basic"
                best_for = [
                    "General Q&A and assistance",
                    "Standard explanations and summaries",
                    "Everyday problem solving",
                    "Basic technical queries"
                ]
                avoid_for = [
                    "Complex technical analysis",
                    "Expert-level reasoning",
                    "Advanced coding tasks"
                ]
                personality = "Helpful and clear for general tasks"
                
            else:
                complexity_min, complexity_max = 1, 3
                tier = "minimal"
                best_for = [
                    "Simple queries and quick responses",
                    "Basic factual questions",
                    "Simple calculations"
                ]
                avoid_for = [
                    "Complex reasoning",
                    "Technical analysis",
                    "Multi-step problems"
                ]
                personality = "Simple, direct, and fast"
            
            # PERSONALIZZA basandosi sui test specifici
            categories_passed = {}
            for test_name, complexity, quality in test_evidence:
                category = test_responses[test_name].get('category', 'general')
                if category not in categories_passed:
                    categories_passed[category] = []
                categories_passed[category].append((complexity, quality))
            
            # Aggiungi specializzazioni
            if 'coding' in categories_passed or 'coding_basic' in categories_passed or 'coding_medium' in categories_passed:
                if tier in ["expert", "professional"]:
                    best_for.insert(1, "Code generation and debugging")
            
            if 'math' in categories_passed or 'arithmetic' in categories_passed or 'multi_step' in categories_passed:
                if any(c[0] >= 5 for c in categories_passed.get('math', [])):
                    best_for.append("Mathematical problem solving")
            
            if 'creative' in categories_passed or 'creative_technical' in categories_passed:
                best_for.append("Creative writing tasks")
            
            # CALCOLA QUALITY SCORE
            avg_quality_passed = sum(t[2] for t in test_evidence) / len(test_evidence)
            quality_score = int(avg_quality_passed)
            
            # CALCOLA CONSISTENCY SCORE (bassa varianza = alta consistenza)
            quality_scores = [t[2] for t in test_evidence]
            variance = sum((q - avg_quality_passed)**2 for q in quality_scores) / len(quality_scores)
            consistency_score = max(1, min(10, int(10 - variance)))
            
        else:
            # Nessun test passato
            complexity_min, complexity_max = 1, 3
            tier = "minimal"
            best_for = ["Very basic queries only"]
            avoid_for = ["Most tasks requiring quality responses"]
            personality = "Limited capabilities"
            quality_score = 3
            consistency_score = 5
            
            if self.verbose:
                print(f"   ⚠️ No tests passed with score ≥7!")
        
        # DETERMINA SPEED
        if avg_time < 1:
            speed = "ultra_fast"
        elif avg_time < 3:
            speed = "fast"
        elif avg_time < 10:
            speed = "medium"
        else:
            speed = "slow"
        
        # STIMA SIZE basandosi su performance e tempo
        if tier == "expert" and avg_time > 5:
            estimated_size = "10-20GB"
        elif tier == "professional":
            estimated_size = "5-15GB"
        elif tier == "basic":
            estimated_size = "2-8GB"
        else:
            estimated_size = "<3GB"
        
        # CREA PROFILO FINALE
        profile = {
            "model_name": model_name,
            "tier": tier,
            "complexity_min": complexity_min,
            "complexity_max": complexity_max,
            "strengths": best_for[:3],  # Top 3
            "weaknesses": avoid_for,
            "best_for": best_for,
            "avoid_for": avoid_for,
            "speed_profile": speed,
            "personality": personality,
            "analysis": f"Handles complexity {min_good if good_complexities else 0}-{max_good if good_complexities else 0} successfully. {len(good_complexities)} tests passed.",
            "inferred_size": estimated_size,
            "quality_score": quality_score,
            "consistency_score": consistency_score,
            "confidence": "high" if len(test_evidence) >= 5 else "medium",
            "special_notes": f"Tested on {len(test_responses)} tasks, passed {len(good_complexities)}",
            "timestamp": time.time(),
            "tests_passed": len(good_complexities),
            "highest_complexity_passed": max_good if good_complexities else 0,
            "avg_response_time": avg_time,
            "avg_quality_score": avg_quality
        }
        
        if self.verbose:
            print(f"\n   🎯 Final Assessment:")
            print(f"      Tier: {tier.upper()}")
            print(f"      Range: {complexity_min}-{complexity_max}")
            print(f"      Quality: {quality_score}/10")
            print(f"      Speed: {speed}")
        
        return profile

    def _build_test_transcript(self, model_name: str, test_responses: Dict) -> str:
        """Costruisce transcript ULTRA-CHIARO e IMPOSSIBILE DA FRAINTENDERE per gli agent"""
        
        # Prima calcola TUTTI i dati chiave
        passed_tests = []
        failed_tests = []
        
        for test_name, data in test_responses.items():
            complexity = self.test_suite.get(test_name, {}).get('complexity', 5)
            quality = data.get('quality', 0)
            
            if quality >= 7:
                passed_tests.append((test_name, complexity, quality))
            else:
                failed_tests.append((test_name, complexity, quality))
        
        # Calcola statistiche chiave
        max_complexity_passed = max(p[1] for p in passed_tests) if passed_tests else 0
        min_complexity_passed = min(p[1] for p in passed_tests) if passed_tests else 0
        avg_time = sum(r['time'] for r in test_responses.values()) / len(test_responses)
        avg_quality = sum(r.get('quality', 5) for r in test_responses.values()) / len(test_responses)
        
        # Determina il range CORRETTO
        if max_complexity_passed >= 9:
            correct_range = "8-10"
            tier = "EXPERT"
        elif max_complexity_passed >= 6:
            correct_range = "5-8"
            tier = "PROFESSIONAL"
        elif max_complexity_passed >= 4:
            correct_range = "3-5"
            tier = "BASIC"
        else:
            correct_range = "1-3"
            tier = "MINIMAL"
        
        # INIZIA CON UN RIEPILOGO IMPOSSIBILE DA IGNORARE
        transcript = f"""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                         CRITICAL SUMMARY                                ║
    ║                    READ THIS FIRST - DO NOT SKIP!                      ║
    ╠════════════════════════════════════════════════════════════════════════╣
    ║ MODEL: {model_name:<63} ║
    ║ HIGHEST COMPLEXITY PASSED (≥7/10): {max_complexity_passed:<35} ║
    ║                                                                        ║
    ║ 🎯 MANDATORY RANGE ASSIGNMENT: {correct_range} ({tier}){' '*(39-len(correct_range)-len(tier)-3)}║
    ║                                                                        ║
    ║ ⚠️  DO NOT ASSIGN ANY OTHER RANGE! THIS IS BASED ON ACTUAL DATA!     ║
    ╚════════════════════════════════════════════════════════════════════════╝

    🚨 EVIDENCE SUMMARY:
    - Tests Passed (≥7/10): {len(passed_tests)} out of {len(test_responses)}
    - Complexity Range Handled Well: {min_complexity_passed}-{max_complexity_passed}
    - Average Response Time: {avg_time:.1f}s
    - Average Quality Score: {avg_quality:.1f}/10

    📊 KEY FINDING: This model handles complexity up to level {max_complexity_passed} successfully.
    Therefore, it MUST be classified as {correct_range} ({tier}).

    ════════════════════════════════════════════════════════════════════════

    🎯 DETAILED EVIDENCE - ALL TESTS PASSED (score ≥ 7/10):
    """
        
        # Mostra i test passati IN ORDINE DI COMPLESSITÀ DECRESCENTE
        if passed_tests:
            for test_name, complexity, quality in sorted(passed_tests, key=lambda x: (x[1], x[2]), reverse=True):
                transcript += f"   ✅ Level {complexity}: {test_name:<30} → {quality}/10"
                if complexity >= 7:
                    transcript += " ⭐ HIGH COMPLEXITY"
                transcript += "\n"
        else:
            transcript += "   ❌ No tests passed with score ≥ 7/10\n"
        
        # Sezione di interpretazione
        transcript += f"""
    ════════════════════════════════════════════════════════════════════════

    📋 INTERPRETATION GUIDE:
    """
        
        if max_complexity_passed >= 9:
            transcript += """
    This model successfully handles EXPERT-LEVEL tasks (complexity 9-10).
    → It should be used for the most complex, challenging queries.
    → Classification: 8-10 (EXPERT)
    """
        elif max_complexity_passed >= 6:
            transcript += """
    This model successfully handles PROFESSIONAL-LEVEL tasks (complexity 6-8).
    → It's suitable for advanced reasoning, analysis, and technical work.
    → Classification: 5-8 (PROFESSIONAL)
    """
        elif max_complexity_passed >= 4:
            transcript += """
    This model successfully handles STANDARD tasks (complexity 4-5).
    → It's good for everyday queries and basic problem-solving.
    → Classification: 3-5 (BASIC)
    """
        else:
            transcript += """
    This model handles only SIMPLE tasks (complexity 1-3).
    → Best for basic queries and simple responses.
    → Classification: 1-3 (MINIMAL)
    """
        
        # Test falliti (per completezza)
        transcript += """
    ════════════════════════════════════════════════════════════════════════

    ❌ TESTS FAILED (score < 7/10):
    """
        
        if failed_tests:
            # Mostra solo i primi 10 test falliti per brevità
            for test_name, complexity, quality in sorted(failed_tests, key=lambda x: x[2], reverse=True)[:10]:
                transcript += f"   ❌ Level {complexity}: {test_name:<30} → {quality}/10\n"
            if len(failed_tests) > 10:
                transcript += f"   ... and {len(failed_tests) - 10} more failed tests\n"
        else:
            transcript += "   ✅ No failed tests!\n"
        
        # Dettagli completi per complessità
        transcript += """
    ════════════════════════════════════════════════════════════════════════

    📊 COMPLETE TEST RESULTS BY COMPLEXITY LEVEL:
    """
        
        # Organizza per complessità
        tests_by_complexity = {}
        for test_name, data in test_responses.items():
            complexity = self.test_suite.get(test_name, {}).get('complexity', 5)
            if complexity not in tests_by_complexity:
                tests_by_complexity[complexity] = []
            tests_by_complexity[complexity].append((test_name, data))
        
        # Mostra per ogni livello di complessità
        for complexity in sorted(tests_by_complexity.keys(), reverse=True):
            transcript += f"\n{'─'*70}\n"
            transcript += f"COMPLEXITY LEVEL {complexity}/10:\n"
            transcript += f"{'─'*70}\n"
            
            tests_at_level = tests_by_complexity[complexity]
            passed_at_level = sum(1 for _, data in tests_at_level if data.get('quality', 0) >= 7)
            
            transcript += f"Tests at this level: {len(tests_at_level)} | Passed: {passed_at_level}\n\n"
            
            for test_name, data in sorted(tests_at_level, key=lambda x: x[1].get('quality', 0), reverse=True):
                quality = data.get('quality', 5)
                status = "✅ PASS" if quality >= 7 else "❌ FAIL"
                
                transcript += f"🔹 {test_name.upper()}\n"
                transcript += f"   Status: {status} | Score: {quality}/10 | Time: {data['time']:.1f}s\n"
                transcript += f"   Category: {data['category']}\n"
                transcript += f"   Question: {data['prompt'][:150]}...\n"
                
                # Per test passati, mostra parte della risposta
                if quality >= 7 and len(data['response']) > 0:
                    transcript += f"   Good Response: {data['response'][:200]}...\n"
                
                transcript += "\n"
        
        # Statistiche finali
        transcript += f"""
    ════════════════════════════════════════════════════════════════════════

    📈 FINAL STATISTICS:
    - Total Tests Run: {len(test_responses)}
    - Tests Passed (≥7/10): {len(passed_tests)} ({len(passed_tests)/len(test_responses)*100:.1f}%)
    - Average Response Time: {avg_time:.2f}s
    - Average Quality Score: {avg_quality:.1f}/10
    - Complexity Range Handled: {min_complexity_passed if passed_tests else 0}-{max_complexity_passed}

    🎯 FINAL CLASSIFICATION: {correct_range} ({tier})

    ⚠️ REMINDER: Base your assessment on the HIGHEST COMPLEXITY PASSED, not averages!
    ════════════════════════════════════════════════════════════════════════
    """
        
        return transcript

    

    

    
    def _agent_validate_profile(self, profile: Dict, test_responses: Dict) -> Dict:
        """🔍 AGENT VALIDATOR: Validazione e correzione completa del profilo basata sui dati reali"""
        
        # DEBUG: Mostra cosa stiamo ricevendo
        if self.verbose:
            print(f"\n   🔍 Validator receiving {len(test_responses)} test results")
        
        # Trova TUTTI i test dove il modello ha score ≥7
        good_complexities = []
        test_evidence = []
        all_tests_debug = []  # Per debug
        
        for test_name, data in test_responses.items():
            complexity = self.test_suite.get(test_name, {}).get('complexity', 5)
            
            # Gestisci diversi possibili nomi per il campo quality
            quality = data.get('quality', data.get('quality_score', data.get('score', 0)))
            
            # Raccogli tutti i test per debug
            all_tests_debug.append((test_name, complexity, quality))
            
            if quality >= 7:  # Test passato bene!
                good_complexities.append(complexity)
                test_evidence.append((test_name, complexity, quality))
        
        # DEBUG: Mostra i top test per quality
        if self.verbose and len(all_tests_debug) > 0:
            print(f"   📋 All test scores:")
            for name, comp, qual in sorted(all_tests_debug, key=lambda x: x[2], reverse=True)[:10]:
                print(f"      {name}: complexity {comp}, score {qual}/10")
        
        # Analizza i risultati
        if good_complexities:
            min_good = min(good_complexities)
            max_good = max(good_complexities)
            
            if self.verbose:
                print(f"\n   ✅ Tests passed (≥7/10): {len(good_complexities)}")
                print(f"      Complexity range: {min_good}-{max_good}")
                # Mostra evidenza dei test più complessi passati
                top_tests = sorted(test_evidence, key=lambda x: x[1], reverse=True)[:5]
                for test, comp, qual in top_tests:
                    print(f"      ✓ {test}: complexity {comp}, score {qual}/10")
            
            # DETERMINA IL RANGE CORRETTO basato sul MASSIMO complexity passato
            if max_good >= 9:
                correct_range = (8, 10)
                tier = "expert"
            elif max_good >= 6:
                correct_range = (5, 8)
                tier = "professional"
            elif max_good >= 4:
                correct_range = (3, 5)
                tier = "basic"
            else:
                correct_range = (1, 3)
                tier = "minimal"
            
            # Determina best_for basandosi sui test effettivamente passati
            best_for = []
            avoid_for = []
            
            # Analizza quali categorie sono state gestite bene
            categories_passed = {}
            for test_name, complexity, quality in test_evidence:
                category = test_responses[test_name].get('category', 'general')
                if category not in categories_passed:
                    categories_passed[category] = []
                categories_passed[category].append((complexity, quality))
            
            # Crea best_for basato sulle categorie passate
            if tier == "expert":
                best_for = [
                    "Complex analysis and research",
                    "Expert-level problem solving",
                    "Advanced technical discussions",
                    "Challenging reasoning tasks"
                ]
                avoid_for = [
                    "Simple queries (use smaller model)",
                    "Real-time applications if response time >5s"
                ]
                personality = "Sophisticated, thorough, and capable of handling complex topics"
                
            elif tier == "professional":
                best_for = [
                    "Professional analysis and reasoning",
                    "Technical explanations and documentation",
                    "Moderate to complex problem solving",
                    "Detailed Q&A on various topics"
                ]
                avoid_for = [
                    "Cutting-edge research questions",
                    "Simple queries (use smaller model)"
                ]
                personality = "Professional and comprehensive in responses"
                
            elif tier == "basic":
                best_for = [
                    "General Q&A and assistance",
                    "Standard explanations and summaries",
                    "Everyday problem solving",
                    "Basic technical queries"
                ]
                avoid_for = [
                    "Complex technical analysis",
                    "Expert-level reasoning",
                    "Advanced coding tasks"
                ]
                personality = "Helpful and clear for general tasks"
                
            else:  # minimal
                best_for = [
                    "Simple queries and quick responses",
                    "Basic factual questions",
                    "Simple calculations",
                    "Quick information lookup"
                ]
                avoid_for = [
                    "Complex reasoning",
                    "Technical analysis",
                    "Creative tasks",
                    "Multi-step problems"
                ]
                personality = "Simple, direct, and fast"
            
            # Personalizza basandosi sui test specifici passati
            if 'coding' in categories_passed or 'coding_basic' in categories_passed:
                if tier in ["expert", "professional"]:
                    best_for.append("Code generation and debugging")
            
            if 'math' in categories_passed or 'arithmetic' in categories_passed:
                if any(c[0] >= 5 for c in categories_passed.get('math', [])):
                    best_for.append("Mathematical problem solving")
            
            if 'creative' in categories_passed:
                best_for.append("Creative writing tasks")
                
        else:
            # Nessun test passato bene
            correct_range = (1, 3)
            tier = "minimal"
            best_for = ["Very basic queries only"]
            avoid_for = ["Most tasks requiring quality responses"]
            personality = "Limited capabilities"
            
            if self.verbose:
                print(f"\n   ⚠️ WARNING: No tests passed with score ≥7!")
                print(f"      This model may have very limited capabilities")
        
        # APPLICA LE CORREZIONI se necessario
        needs_correction = False
        
        # Controlla se il range deve essere corretto
        if (profile.get('complexity_min', 0), profile.get('complexity_max', 0)) != correct_range:
            needs_correction = True
            if self.verbose:
                print(f"\n   🔧 Correcting range from {profile.get('complexity_min', 0)}-{profile.get('complexity_max', 0)} to {correct_range[0]}-{correct_range[1]}")
        
        # Controlla se il tier deve essere corretto
        if profile.get('tier', '').lower() != tier:
            needs_correction = True
            if self.verbose:
                print(f"   🔧 Correcting tier from {profile.get('tier', 'unknown')} to {tier}")
        
        # Applica tutte le correzioni
        if needs_correction:
            profile['complexity_min'] = correct_range[0]
            profile['complexity_max'] = correct_range[1]
            profile['tier'] = tier
            profile['best_for'] = best_for
            profile['avoid_for'] = avoid_for
            profile['personality'] = personality
            
            # Aggiungi metadati sulla correzione
            profile['correction_applied'] = True
            profile['correction_reason'] = f"Based on actual test performance (max complexity passed: {max_good if good_complexities else 0})"
            profile['tests_passed'] = len(good_complexities)
            profile['highest_complexity_passed'] = max_good if good_complexities else 0
            
            # Aggiorna anche il quality score basandosi sui test
            if test_evidence:
                avg_quality_of_passed = sum(t[2] for t in test_evidence) / len(test_evidence)
                profile['quality_score'] = int(avg_quality_of_passed)
        
        # Aggiungi sempre informazioni di validazione
        profile['validation_timestamp'] = time.time()
        profile['validation_confidence'] = "high" if len(test_evidence) >= 5 else "medium" if len(test_evidence) >= 3 else "low"
        
        if self.verbose and needs_correction:
            print(f"\n   ✅ Profile corrected successfully")
            print(f"      Final assessment: {tier} ({correct_range[0]}-{correct_range[1]})")
        
        return profile


    def _agent_evaluate_conversation(self, question: str, model_response: str, 
                                    category: str, response_time: float) -> Dict:
        """🤖 EVALUATOR AGENT analizza la risposta attraverso CONVERSAZIONE"""
        
        # L'evaluator "parla" con mistral per analizzare la risposta
        evaluator_prompt = f"""<role>
    You are an Expert Model Evaluator. You just witnessed a conversation between a user and an LLM model.
    Your job is to evaluate the quality of the model's response through critical analysis.
    </role>

    <conversation>
    USER ASKED ({category} task):
    "{question}"

    MODEL RESPONDED (in {response_time:.1f} seconds):
    "{model_response}"
    </conversation>

    <your_evaluation_task>
    Analyze this response as if you were reviewing a colleague's work. Ask yourself:

    1. CORRECTNESS: Is this answer actually correct/appropriate?
    - Would you trust this response?
    - Are there any errors or misconceptions?
    - For code: would it work? For math: is it right?

    2. COMPLETENESS: Did the model fully answer the question?
    - Is anything important missing?
    - Is it too brief or too verbose?

    3. QUALITY: How well is this written?
    - Is it clear and coherent?
    - Is the explanation good?
    - Would a user be satisfied?

    4. INTELLIGENCE: Does this response show real understanding?
    - Or is it just pattern matching?
    - Does it show reasoning ability?
    - Is there depth or just surface level?
    </your_evaluation_task>

    <output_format>
    Provide your honest evaluation:

    QUALITY_SCORE: [0.0 to 1.0]
    CORRECTNESS: [correct/partially-correct/incorrect]
    COMPLETENESS: [complete/partial/incomplete]
    CLARITY: [excellent/good/fair/poor]
    INTELLIGENCE_LEVEL: [high/medium/low]

    STRENGTHS: [what impressed you]
    WEAKNESSES: [what concerned you]

    OVERALL_ASSESSMENT: [2-3 sentences of your honest opinion]

    RECOMMENDED_USE: [what tasks is this model suitable for based on this response]
    </output_format>

    Be honest and critical. This helps us route tasks to the right models.

    Evaluate now:"""

        try:
            # L'evaluator (mistral) analizza
            eval_response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": evaluator_prompt}],
                stream=False,
                options={'temperature': 0.2}
            )
            
            eval_text = eval_response['message']['content']
            
            # Estrai valutazioni
            import re
            
            def extract_score(text):
                pattern = r"QUALITY_SCORE:\s*([0-9]*\.?[0-9]+)"
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        return max(0.0, min(1.0, score))
                    except:
                        pass
                return 0.5
            
            def extract_field(text, field):
                pattern = rf"{field}:\s*(.+?)(?:\n|$)"
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
                return ""
            
            def extract_assessment(text):
                pattern = r"OVERALL_ASSESSMENT:\s*(.+?)(?:\n\n|RECOMMENDED|$)"
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(1).strip()
                return ""
            
            quality = extract_score(eval_text)
            correctness = extract_field(eval_text, "CORRECTNESS")
            completeness = extract_field(eval_text, "COMPLETENESS")
            clarity = extract_field(eval_text, "CLARITY")
            intelligence = extract_field(eval_text, "INTELLIGENCE_LEVEL")
            strengths = extract_field(eval_text, "STRENGTHS")
            weaknesses = extract_field(eval_text, "WEAKNESSES")
            assessment = extract_assessment(eval_text)
            recommended = extract_field(eval_text, "RECOMMENDED_USE")
            
            # Converti valutazioni in score numerico
            score = quality
            
            # Aggiusta basandosi su correttezza
            if "incorrect" in correctness.lower():
                score *= 0.5
            elif "partially" in correctness.lower():
                score *= 0.8
            
            return {
                "score": score,
                "quality": quality,
                "correctness": correctness,
                "completeness": completeness,
                "clarity": clarity,
                "intelligence": intelligence,
                "strengths": [strengths] if strengths else [],
                "weaknesses": [weaknesses] if weaknesses else [],
                "analysis": eval_text,
                "assessment": assessment,
                "notes": f"Evaluator assessment: {assessment[:200]}...",
                "recommended_use": recommended
            }
            
        except Exception as e:
            if self.verbose:
                print(f"     ⚠️ Evaluator conversation failed: {e}")
            
            return {
                "score": 0.5,
                "quality": 0.5,
                "analysis": f"Evaluation error: {str(e)}",
                "strengths": [],
                "weaknesses": ["Evaluation failed"],
                "notes": "Evaluation error"
            }
    
    def _evaluate_response_with_agent(self, response: str, test_config: Dict, 
                                  response_time: float) -> Dict:
        """🤖 USA UN LLM PER VALUTARE LA QUALITÀ DELLA RISPOSTA"""
        
        category = test_config.get("category", "general")
        prompt_text = test_config.get("prompt", "")
        
        evaluation_prompt = f"""<role>
    You are a Response Quality Evaluator Agent. Your job is to assess the quality of LLM responses.
    </role>

    <task>
    A model was given this task:
    CATEGORY: {category}
    PROMPT: "{prompt_text}"

    The model responded with:
    "{response[:1000]}"

    Response time: {response_time:.1f} seconds
    Response length: {len(response.split())} words
    </task>

    <evaluation_criteria>
    Evaluate the response on these dimensions (0.0 to 1.0 scale):

    1. CORRECTNESS: Is the answer correct/appropriate?
    2. COMPLETENESS: Does it fully answer the question?
    3. COHERENCE: Is the response well-structured and clear?
    4. RELEVANCE: Does it address what was asked?
    5. DEPTH: For the category, is the response sufficiently detailed?
    </evaluation_criteria>

    <output_format>
    Provide your evaluation in this EXACT format:

    CORRECTNESS: [0.0-1.0]
    COMPLETENESS: [0.0-1.0]
    COHERENCE: [0.0-1.0]
    RELEVANCE: [0.0-1.0]
    DEPTH: [0.0-1.0]
    OVERALL_QUALITY: [0.0-1.0]
    STRENGTHS: [comma separated list]
    WEAKNESSES: [comma separated list]
    REASONING: [brief explanation]
    </output_format>

    Evaluate now:"""

        try:
            # Usa un modello veloce per valutare
            eval_response = ollama.chat(
                model="mistral",
                messages=[{"role": "user", "content": evaluation_prompt}],
                stream=False,
                options={'temperature': 0.1}
            )
            
            eval_text = eval_response['message']['content']
            
            # Estrai le valutazioni - USA RAW STRINGS
            import re
            
            def extract_score(text, key):
                pattern = rf"{key}:\s*([0-9]*\.?[0-9]+)"  # RAW STRING!
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except:
                        return 0.5
                return 0.5
            
            def extract_list(text, key):
                pattern = rf"{key}:\s*(.+?)(?:\n|$)"  # RAW STRING!
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    items = match.group(1).strip()
                    return [item.strip() for item in items.split(',') if item.strip()]
                return []
            
            def extract_reasoning(text):
                pattern = r"REASONING:\s*(.+?)(?:\n\n|$)"
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(1).strip()
                return ""
            
            correctness = extract_score(eval_text, "CORRECTNESS")
            completeness = extract_score(eval_text, "COMPLETENESS")
            coherence = extract_score(eval_text, "COHERENCE")
            relevance = extract_score(eval_text, "RELEVANCE")
            depth = extract_score(eval_text, "DEPTH")
            overall = extract_score(eval_text, "OVERALL_QUALITY")
            
            strengths = extract_list(eval_text, "STRENGTHS")
            weaknesses = extract_list(eval_text, "WEAKNESSES")
            reasoning = extract_reasoning(eval_text)
            
            # Se overall non è stato trovato, calcolalo
            if overall == 0.5 and correctness != 0.5:
                overall = (correctness * 0.3 + completeness * 0.2 + 
                        coherence * 0.2 + relevance * 0.2 + depth * 0.1)
            
            return {
                "score": correctness,
                "quality": overall,
                "correctness": correctness,
                "completeness": completeness,
                "coherence": coherence,
                "relevance": relevance,
                "depth": depth,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "reasoning": reasoning
            }
            
        except Exception as e:
            if self.verbose:
                print(f"     ⚠️ Agent evaluation failed: {e}")
            
            # Fallback minimo
            return {
                "score": 0.5,
                "quality": 0.5,
                "correctness": 0.5,
                "completeness": 0.5,
                "coherence": 0.5,
                "relevance": 0.5,
                "depth": 0.5,
                "strengths": [],
                "weaknesses": ["Evaluation failed"],
                "reasoning": "Agent evaluation error"
            }
    
    def _evaluate_response(self, response: str, test_config: Dict) -> float:
        """Valuta qualità della risposta (0-1)"""
        score = 0.0
        
        if not response or len(response.strip()) < 5:
            return 0.0
        
        category = test_config["category"]
        
        if category == "simple":
            score = 0.8 if len(response) > 10 else 0.4
        
        elif category == "math":
            if "correct_answer" in test_config:
                score = 1.0 if test_config["correct_answer"] in response else 0.0
            else:
                score = 0.5
        
        elif category == "reasoning":
            logic_words = ["because", "therefore", "however", "if", "then", "conclude"]
            matches = sum(1 for word in logic_words if word.lower() in response.lower())
            score = min(1.0, matches * 0.25)
        
        elif category == "coding":
            code_indicators = ["def ", "return", ":", "(", ")"]
            matches = sum(1 for indicator in code_indicators if indicator in response)
            score = min(1.0, matches * 0.2)
        
        elif category == "creative":
            score = 0.8 if len(response) > 20 else 0.4
        
        elif category == "technical":
            tech_terms = ["supervised", "unsupervised", "learning", "data", "model", "training"]
            matches = sum(1 for term in tech_terms if term.lower() in response.lower())
            score = min(1.0, matches * 0.2)
        
        else:
            score = 0.5
        
        return score
    
    def _calculate_quality_score(self, model_name: str, benchmark_results: Dict, 
                            size_gb: float) -> int:
        """
        Calcola quality score usando AGENT se disponibile
        """
        # USA L'AGENT SE DISPONIBILE
        if self.quality_evaluator:
            try:
                return self.quality_evaluator.evaluate_model_quality(
                    model_name, benchmark_results, size_gb
                )
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Agentic evaluation failed: {e}, using fallback")
        
        # FALLBACK: Calcolo basato su dimensione
        if size_gb < 1:
            return 3
        elif size_gb < 3:
            return 4
        elif size_gb < 5:
            return 5
        elif size_gb < 8:
            return 6
        elif size_gb < 12:
            return 7
        elif size_gb < 15:
            return 8
        elif size_gb < 20:
            return 9
        else:
            return 10
    
    def _calculate_speed(self, avg_response_time: float) -> str:
        """Determina velocità basata SOLO su tempo risposta"""
        if avg_response_time < 0.5:
            return "ultra_fast"
        elif avg_response_time < 1.5:
            return "fast"
        elif avg_response_time < 5.0:
            return "medium"
        else:
            return "slow"
    
    def _create_config_from_benchmark(self, model_name: str, benchmark: Dict, size_gb: float) -> ModelConfig:
        """Crea config dal profilo agentico - CON FIX per model naming"""
        
        # ID interno del sistema (univoco)
        model_id = f"ollama_{model_name.replace(':', '_')}"
        
        # Estrai valori dal benchmark
        tier = benchmark.get('tier', 'competent')
        speed = benchmark.get('speed_profile', 'medium')
        complexity_min = benchmark.get('complexity_min', 3)
        complexity_max = benchmark.get('complexity_max', 7)
        
        # Crea la config con i parametri ESATTI che ModelConfig richiede
        config = ModelConfig(
            provider="ollama",
            model_id=model_id,
            display_name=f"{model_name} ({size_gb:.1f}GB)",
            
            # Costi - Ollama è gratuito
            cost_per_1k_input=0.0,
            cost_per_1k_output=0.0,
            
            # Parametri qualitativi dal benchmark
            tier=tier,
            speed=speed,
            
            # Context window - stima basata su size
            context_window=8192 if size_gb > 10 else 4096 if size_gb > 5 else 2048,
            
            # Liste qualitative
            best_for=benchmark.get('best_for', ['general tasks']),
            avoid_for=benchmark.get('avoid_for', []),
            
            # Ollama non richiede API key
            requires_key=False,
            
            # Dimensione
            size_gb=size_gb,
            
            # Range di complessità dal profiling agentico
            complexity_range=(complexity_min, complexity_max),
            
            # Personalità dal profiling
            personality=benchmark.get('personality', 'Standard assistant')
        )
        
        # 🔥 FIX CRITICO: Salva il nome ORIGINALE del modello per Ollama
        if hasattr(config, '__dict__'):
            # IMPORTANTE: questo è il nome che Ollama riconosce!
            config.model_name = model_name  # Es: "gemma3:12b"
            config.ollama_model_name = model_name  # Backup esplicito
            
            # Metadati dal profiling agentico
            config.quality_score = benchmark.get('quality_score', 5)
            config.avg_quality = benchmark.get('avg_quality_score', 5)
            config.consistency_score = benchmark.get('consistency_score', 5)
            config.confidence_level = benchmark.get('confidence', 'medium')
            config.special_notes = benchmark.get('special_notes', '')
            
            # Capacità dedotte
            config.capabilities = {
                'reasoning': benchmark.get('avg_quality_score', 5) >= 7,
                'coding': 'code' in ' '.join(benchmark.get('strengths', [])).lower() if benchmark.get('strengths') else False,
                'math': 'math' in ' '.join(benchmark.get('strengths', [])).lower() if benchmark.get('strengths') else False,
                'creative': 'creative' in ' '.join(benchmark.get('strengths', [])).lower() if benchmark.get('strengths') else False,
                'analysis': benchmark.get('complexity_max', 5) >= 7,
                'conversation': True,
                'multilingual': False
            }
            
            # Informazioni base
            config.base_model = model_name.split(':')[0]
            config.quantization = model_name.split(':')[1] if ':' in model_name else 'default'
            
            # Tempi di risposta
            config.avg_response_time = benchmark.get('avg_response_time', 5.0)
            
            # Strengths e weaknesses dal profiling
            config.strengths = benchmark.get('strengths', [])
            config.weaknesses = benchmark.get('weaknesses', [])
            
            # Report completi degli agent (se vuoi conservarli per debug)
            if benchmark.get('full_profile'):
                config.full_agent_profile = benchmark.get('full_profile', '')
        
        return config

    def _estimate_context_window(self, model_base: str) -> int:
        """Stima context window basata su modello"""
        known_contexts = {
            "llama3": 128000,
            "llama3.1": 128000,
            "llama3.2": 128000,
            "mistral": 32768,
            "qwen": 32768,
            "qwen2": 32768,
            "qwen2.5": 32768,
            "deepseek": 32768,
            "deepseek-r1": 32768,
            "gemma": 8192,
            "gemma2": 8192,
            "gemma3": 8192,
            "gpt": 32768,
            "mixtral": 32768
        }
        
        for key, context in known_contexts.items():
            if key in model_base.lower():
                return context
        
        return 8192
    
    def _generate_display_name(self, model_base: str, size_gb: float, quality_score: int) -> str:
        """Genera nome user-friendly"""
        name_parts = model_base.split('-')
        formatted_base = ' '.join(part.capitalize() for part in name_parts)
        
        if size_gb > 0:
            formatted_base += f" {size_gb:.0f}GB"
        
        if quality_score >= 9:
            quality_tag = "⭐"
        elif quality_score >= 7:
            quality_tag = "✓"
        else:
            quality_tag = ""
        
        return f"{formatted_base} (Local) {quality_tag}".strip()
    
    def _print_benchmark_summary(self, model_name: str, config: ModelConfig):
        """Stampa summary del benchmark"""
        print(f"   ✓ {config.display_name}")
        print(f"     Quality: {config.quality_score}/10 | Speed: {config.speed}")
        print(f"     Best for: {', '.join(config.best_for[:3])}")

# ====================
# COMPLEXITY EVALUATORS (original)
# ====================

class ComplexityEvaluator(ABC):
    """Base class per valutazione complessità"""
    
    @abstractmethod
    def evaluate(self, user_input: str) -> int:
        pass

class MistralComplexityEvaluator(ComplexityEvaluator):
    """Valutazione agentica della complessità con reasoning chain"""
    
    def __init__(self, ollama_provider, verbose=True):
        self.provider = ollama_provider
        self.verbose = verbose  # AGGIUNGI QUESTO!
    
    def evaluate(self, user_input: str) -> int:
        prompt = f"""You are a Task Complexity Analyzer Agent. You must understand the TRUE complexity of any request through reasoning.

    <user_message>
    {user_input}
    </user_message>

    <reasoning_framework>
    To determine complexity, consider these dimensions:

    1. COGNITIVE LOAD
    - How much thinking/reasoning is required?
    - Is it recall of facts or generation of new ideas?
    - Does it require understanding of multiple concepts?

    2. EXECUTION DEPTH  
    - Is it a single-step or multi-step process?
    - Does it require planning and architecture?
    - How many subsystems need to work together?

    3. EXPERTISE REQUIRED
    - Can a child answer this?
    - Does it need domain knowledge?
    - Does it require specialized skills?

    4. OUTPUT SOPHISTICATION
    - Is the expected output simple or complex?
    - Does it need to be precise and detailed?
    - Are there quality/performance requirements?

    5. CONTEXTUAL AMBITION
    - What is the user trying to achieve ultimately?
    - Is this question part of a larger goal?
    - What level of solution do they expect?
    </reasoning_framework>

    <complexity_scale>
    Score each dimension 1-10, then synthesize:
    - All dimensions low (1-2) → Final score 1-3
    - Mixed low/medium → Final score 4-6  
    - Mostly high → Final score 7-8
    - All dimensions high → Final score 9-10

    The final score should reflect the HIGHEST demanding dimension, not the average.
    </complexity_scale>

    Reason through the dimensions and provide a final complexity score from 1 to 10."""

        try:
            response = self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model="mistral",
                temperature=0.1,
                max_tokens=500
            )
            
            # 🔥 PARSING ROBUSTO - trova il numero in QUALSIASI formato
            import re
            
            # Pattern per trovare score in vari formati
            patterns = [
                r'score[:\s]+(\d+)',                    # score: 5, score 5
                r'score\s+(?:is|of|=)\s+(\d+)',        # score is 5, score of 5
                r'final\s+score[:\s]+(\d+)',           # final score: 5
                r'final\s+score\s+(?:is|of|=)\s+(\d+)', # final score is 5
                r'complexity[:\s]+(\d+)',              # complexity: 5
                r'complexity\s+(?:is|of|=)\s+(\d+)',   # complexity is 5
                r'score.*?(\d+)',                      # score ... 5 (qualsiasi cosa in mezzo)
                r'\b(\d+)\s*(?:/10)?$',                # 5 o 5/10 alla fine
                r'(?:^|\n)\s*(\d+)\s*$',               # numero solo su una riga
            ]
            
            response_lower = response.lower()
            
            # Prova ogni pattern
            for pattern in patterns:
                matches = re.findall(pattern, response_lower, re.MULTILINE | re.IGNORECASE)
                if matches:
                    # Prendi l'ultimo match (probabilmente il finale)
                    for match in reversed(matches):
                        try:
                            score = int(match)
                            if 1 <= score <= 10:
                                if self.verbose:
                                    print(f"   → Complexity: {score}/10 (found with pattern: {pattern[:20]}...)")
                                return score
                        except:
                            continue
            
            # 🔥 FALLBACK INTELLIGENTE basato su parole chiave
            response_lower = response.lower()
            
            # Se parla di "simple", "greeting", "low", usa score basso
            if any(word in response_lower for word in ['simple', 'greeting', 'trivial', 'basic', 'low']):
                if self.verbose:
                    print(f"   → Complexity: 2/10 (fallback: detected simple task)")
                return 2
            
            # Se parla di "complex", "high", "difficult", usa score alto
            if any(word in response_lower for word in ['complex', 'difficult', 'high', 'sophisticated']):
                if self.verbose:
                    print(f"   → Complexity: 7/10 (fallback: detected complex task)")
                return 7
            
            # Default medio
            if self.verbose:
                print(f"   → Complexity: 4/10 (fallback: no clear indication)")
            return 4
            
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Complexity evaluation failed: {e}")
            return 4
        
class HeuristicComplexityEvaluator(ComplexityEvaluator):
    """Advanced heuristic complexity evaluation with multi-dimensional analysis"""
    
    def __init__(self):
        self.patterns = {
            'error_indicators': {
                'patterns': [
                    r'error\s*:', r'exception', r'invalid\s+format', r'missing\s+[\"\']?\w+[\"\']?',
                    r'failed\s+to', r'cannot\s+\w+', r'unable\s+to', r'unexpected\s+\w+',
                    r'traceback', r'stack\s+trace', r'assertion\s+failed', r'undefined',
                    r'null\s+pointer', r'segmentation\s+fault', r'syntax\s+error'
                ],
                'weight': 4.0,
                'min_complexity': 7
            },
            
            'ai_output_patterns': {
                'patterns': [
                    r'thought\s*:', r'action\s*:', r'observation\s*:', r'reflection\s*:',
                    r'step\s+\d+\s*:', r'```[\w]*\n', r'here\'s\s+the\s+corrected',
                    r'apologies\s+for', r'let\s+me\s+clarify', r'to\s+summarize',
                    r'invalid\s+format.*missing', r'confused\s+about'
                ],
                'weight': 3.5,
                'min_complexity': 6
            },
            
            'code_patterns': {
                'patterns': [
                    r'def\s+\w+\s*KATEX_INLINE_OPEN', r'class\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import',
                    r'if\s+__name__\s*==', r'async\s+def', r'return\s+\w+', r'yield\s+',
                    r'lambda\s+\w+:', r'@\w+', r'try\s*:', r'except\s+\w+',
                    r'\w+\s*=\s*\w+\s*KATEX_INLINE_OPEN', r'for\s+\w+\s+in\s+', r'while\s+\w+\s*:'
                ],
                'weight': 2.5,
                'min_complexity': 5
            },
            
            'architecture_patterns': {
                'patterns': [
                    r'microservice', r'distributed\s+system', r'load\s+balanc', r'scalab',
                    r'kubernetes', r'docker', r'orchestrat', r'message\s+queue', r'event\s+driven',
                    r'design\s+pattern', r'architect', r'infrastructure', r'deployment',
                    r'high\s+availability', r'fault\s+toleran', r'consensus', r'replication'
                ],
                'weight': 4.5,
                'min_complexity': 8
            },
            
            'algorithm_patterns': {
                'patterns': [
                    r'algorithm', r'time\s+complexity', r'space\s+complexity', r'big\s*-?\s*o',
                    r'dynamic\s+programming', r'graph\s+traversal', r'tree\s+\w+', r'sort\s+\w+',
                    r'binary\s+search', r'hash\s+table', r'recursion', r'optimization',
                    r'np\s*-?\s*complete', r'heuristic', r'greedy\s+approach'
                ],
                'weight': 3.8,
                'min_complexity': 7
            },
            
            'simple_patterns': {
                'patterns': [
                    r'^hi\s*[!?]?$', r'^hello\s*[!?]?$', r'^hey\s*[!?]?$', r'^thanks?\s*[!?]?$',
                    r'^\d+\s*[\+\-\*/]\s*\d+\s*[?]?$', r'^what\s+is\s+\d+', r'^yes\s*[?]?$',
                    r'^no\s*[?]?$', r'^ok\s*[?]?$', r'^good\s*[?]?$'
                ],
                'weight': -3.0,
                'max_complexity': 2
            }
        }
        
        self.linguistic_features = {
            'subordinate_clauses': [r'\b(which|that|who|whom|whose|where|when|while|although|because|since|if|unless)\b'],
            'technical_terms': [r'\b(api|sdk|framework|library|protocol|interface|implementation|abstraction|encapsulation)\b'],
            'quantifiers': [r'\b(all|some|many|few|several|various|multiple|numerous)\b'],
            'modal_verbs': [r'\b(should|could|would|might|must|shall|may|can|will)\b'],
            'comparison': [r'\b(more|less|better|worse|faster|slower|higher|lower|greater|smaller)\s+than\b']
        }
    
    def evaluate(self, user_input: str) -> int:
        """Multi-dimensional complexity evaluation - CONSERVATIVE"""
        
        input_lower = user_input.lower()
        word_count = len(user_input.split())
        
        # Start with LOW base score
        base_score = 3
        
        # SIMPLE PATTERNS - if matched, return immediately
        simple_patterns = [
            (r'^(hi|hello|hey|ciao|salve)\b', 1),
            (r'^(how are you|come stai|what\'s up)', 1),
            (r'^(what|cosa|che cos|qual è) (is|è|sono) \w+\??$', 2),
            (r'^\d+\s*[\+\-\*/]\s*\d+', 2),
            (r'^(explain|spiega|cos\'è|what is) \w+$', 3),
            (r'^(list|elenca|dammi|give me) \w+', 3),
            (r'^(translate|traduci)', 3),
            (r'^(write|scrivi) (a|an|una?) (short|breve)', 4),
        ]
        
        for pattern, score in simple_patterns:
            if re.search(pattern, input_lower):
                print(f"[Complexity] Matched simple pattern: {pattern[:20]}... -> {score}")
                return score
        
        # Check length
        if word_count < 10:
            base_score = 2
        elif word_count < 30:
            base_score = 3
        elif word_count < 100:
            base_score = 4
        else:
            base_score = 5
        
        complexity_modifiers = 0
        
        # ERROR PATTERNS (+3 only if actual debugging needed)
        if re.search(r'(error|exception|traceback|failed)', input_lower):
            if re.search(r'(fix|debug|solve|why|perché)', input_lower):
                complexity_modifiers += 3
            else:
                complexity_modifiers += 1
        
        # CODE PATTERNS (+1 to +2 max)
        code_indicators = [r'```', r'def ', r'class ', r'import ', r'function']
        if any(re.search(p, user_input) for p in code_indicators):
            if 'debug' in input_lower or 'fix' in input_lower:
                complexity_modifiers += 2
            else:
                complexity_modifiers += 1
        
        # ANALYSIS KEYWORDS (+1)
        analysis_words = ['analyze', 'compare', 'contrast', 'evaluate', 'analizza']
        if any(word in input_lower for word in analysis_words):
            complexity_modifiers += 1
        
        # EXPERT KEYWORDS (+2)
        expert_words = ['architecture', 'microservice', 'distributed', 'algorithm', 
                       'optimization', 'machine learning', 'neural network']
        if any(word in input_lower for word in expert_words):
            complexity_modifiers += 2
        
        # QUESTION WORDS (reduce complexity)
        if re.search(r'^(what|when|where|who|why|how|qual|quando|dove|chi|perché|come)\b', input_lower):
            if word_count < 20:
                complexity_modifiers -= 1
        
        # Calculate final score
        final_score = base_score + complexity_modifiers
        
        # CAP SCORES based on input length
        if word_count < 20:
            final_score = min(final_score, 5)
        elif word_count < 50:
            final_score = min(final_score, 6)
        
        # Never go below 1 or above 10
        final_score = max(1, min(10, final_score))
        
        print(f"[Complexity] Words: {word_count}, Base: {base_score}, "
              f"Modifiers: {complexity_modifiers}, Final: {final_score}")
        
        return final_score
    
    def _calculate_base_complexity(self, words: int, lines: int, chars: int) -> float:
        """Sophisticated base complexity calculation"""
        word_score = 3 + (words ** 0.7) / 10
        line_bonus = min(2, (lines - 1) * 0.3) if lines > 1 else 0
        avg_word_length = chars / max(words, 1)
        density_factor = min(1.5, avg_word_length / 6)
        
        return word_score + line_bonus * density_factor
    
    def _analyze_linguistic_complexity(self, text: str) -> float:
        """Analyze linguistic features for complexity"""
        score = 0
        
        for feature, patterns in self.linguistic_features.items():
            matches = sum(len(re.findall(pattern, text, re.IGNORECASE)) 
                         for pattern in patterns)
            
            if feature == 'subordinate_clauses':
                score += matches * 0.8
            elif feature == 'technical_terms':
                score += matches * 0.6
            elif feature == 'quantifiers':
                score += matches * 0.3
            elif feature == 'modal_verbs':
                score += matches * 0.4
            elif feature == 'comparison':
                score += matches * 0.5
        
        return min(5, score)
    
    def _analyze_structural_complexity(self, text: str) -> float:
        """Analyze structural patterns"""
        score = 0
        
        if any(pattern in text for pattern in ['```', '{', '[', '<', '|']):
            score += 1.5
        
        if re.search(r'^\s*[\d\-\*\•]\s+', text, re.MULTILINE):
            score += 1.0
        
        if text.count('"') >= 2 or text.count("'") >= 2:
            score += 0.5
        
        if re.search(r'[A-Z]{2,}', text):
            score += 0.5
        
        nesting_depth = self._calculate_nesting_depth(text)
        score += min(2, nesting_depth * 0.5)
        
        return score
    
    def _calculate_nesting_depth(self, text: str) -> int:
        """Calculate maximum nesting depth of brackets/parentheses"""
        max_depth = 0
        current_depth = 0
        
        for char in text:
            if char in '([{':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char in ')]}':
                current_depth = max(0, current_depth - 1)
        
        return max_depth
    
    def _calculate_information_density(self, text: str) -> float:
        """Calculate information density of the text"""
        words = text.split()
        if not words:
            return 0
        
        unique_ratio = len(set(words)) / len(words)
        avg_length = sum(len(w) for w in words) / len(words)
        punct_count = sum(1 for c in text if c in '.,;:!?-()[]{}')
        punct_density = punct_count / len(text) if text else 0
        
        density = (unique_ratio * 2 + avg_length / 5 + punct_density * 10)
        
        return min(3, density)
    
# Aggiungi DOPO la classe HeuristicComplexityEvaluator

class AgenticQualityEvaluator:
    """Valutazione agentica della qualità dei modelli"""
    
    def __init__(self, ollama_provider, verbose=True):
        self.provider = ollama_provider
        self.verbose = verbose
    
    def evaluate_model_quality(self, model_name: str, benchmark_results: Dict, 
                          size_gb: float) -> int:
        """🤖 Decisione finale basata sulle conversazioni dell'evaluator"""
        
        avg_quality = benchmark_results.get("avg_quality_score", 0.5)
        test_results = benchmark_results.get("test_results", {})
        
        # Costruisci report
        performance_report = self._build_performance_report(
            model_name, size_gb, test_results, 
            benchmark_results.get("capabilities", []),
            benchmark_results.get("avg_response_time", 0),
            benchmark_results.get("overall_score", 0)
        )
        
        # Prompt per decisione finale
        final_prompt = f"""<role>
    You are reviewing an evaluation report from an expert evaluator who tested an LLM model.
    </role>

    <report>
    {performance_report}
    </report>

    <your_task>
    The evaluator had actual conversations with the model "{model_name}" and assessed response quality.

    Based on the evaluator's assessment of {avg_quality:.2f}/1.0 quality from real conversations:

    Assign a final quality score 1-10:
    - 0.80+ quality → 8-10
    - 0.65-0.79 quality → 6-7
    - 0.50-0.64 quality → 4-5
    - Below 0.50 quality → 2-3

    Look at the "EVALUATOR'S RECOMMENDATION" section.
    </your_task>

    <critical_instruction>
    Your LAST LINE must be ONLY a single number from 1 to 10.
    Example: if you decide score is 7, just write "7" on the last line.
    </critical_instruction>

    Provide your reasoning and final score:"""

        try:
            response = self.provider.chat(
                messages=[{"role": "user", "content": final_prompt}],
                model="mistral",
                temperature=0.1,
                max_tokens=400
            )
            
            if self.verbose:
                print(f"\n🤖 [Final Quality Decision for {model_name}]")
                print(f"   Evaluator quality from conversations: {avg_quality:.2f}")
            
            # Estrai score
            import re
            lines = response.strip().split('\n')
            
            for line in reversed(lines):
                line = line.strip()
                
                if line.isdigit() and 1 <= int(line) <= 10:
                    score = int(line)
                    if self.verbose:
                        print(f"   ✅ Final Score: {score}/10")
                    return score
                
                numbers = re.findall(r'\b([1-9]|10)\b', line)
                if numbers:
                    score = int(numbers[-1])
                    if self.verbose:
                        print(f"   ✅ Final Score: {score}/10")
                    return score
            
            # Fallback BASATO SU CONVERSAZIONI
            if avg_quality >= 0.80:
                fallback = 9
            elif avg_quality >= 0.65:
                fallback = 7
            elif avg_quality >= 0.50:
                fallback = 5
            else:
                fallback = 3
            
            if self.verbose:
                print(f"   Using conversation-based score: {fallback}/10")
            
            return fallback
            
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Error: {e}")
            return self._size_based_fallback(size_gb)
    
    def _build_performance_report(self, model_name: str, size_gb: float,
                             test_results: Dict, capabilities: List[str],
                             avg_time: float, overall_score: float) -> str:
        """Report basato su conversazioni reali con evaluator agent"""
        
        report = f"""MODEL EVALUATION REPORT - CONVERSATIONAL ANALYSIS
    {'='*70}
    Model: {model_name}
    Size: {size_gb:.1f} GB
    {'='*70}

    EVALUATOR AGENT ASSESSMENT:
    (An expert evaluator had conversations with this model and analyzed responses)
    """
        
        quality_scores = []
        
        for test_name, result in test_results.items():
            quality = result.get('quality_score', 0.5)
            quality_scores.append(quality)
            
            report += f"\n\n{test_name.upper()}:"
            report += f"\n{'─'*70}"
            
            # Mostra estratto della conversazione
            response_preview = result.get('response', '')[:200]
            report += f"\nModel's response: \"{response_preview}...\""
            
            # Valutazione dell'evaluator
            report += f"\n\nEvaluator's assessment:"
            report += f"\n  Quality Score: {quality:.2f}/1.0"
            
            correctness = result.get('correctness', 'unknown')
            clarity = result.get('clarity', 'unknown')
            intelligence = result.get('intelligence', 'unknown')
            
            if correctness:
                report += f"\n  Correctness: {correctness}"
            if clarity:
                report += f"\n  Clarity: {clarity}"
            if intelligence:
                report += f"\n  Intelligence Level: {intelligence}"
            
            # Note dell'evaluator
            notes = result.get('evaluator_notes', '')
            if notes:
                report += f"\n  Notes: {notes[:150]}..."
            
            strengths = result.get('strengths', [])
            weaknesses = result.get('weaknesses', [])
            
            if strengths and strengths[0]:
                report += f"\n  ✓ Strengths: {strengths[0][:100]}"
            if weaknesses and weaknesses[0]:
                report += f"\n  ✗ Weaknesses: {weaknesses[0][:100]}"
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        report += f"\n\n{'='*70}"
        report += f"\nEVALUATOR'S OVERALL ASSESSMENT:"
        report += f"\n{'='*70}"
        report += f"\nAverage Quality (from conversation analysis): {avg_quality:.2f}/1.0"
        report += f"\nCapabilities demonstrated: {len(capabilities)}/{len(test_results)}"
        report += f"\nAverage response time: {avg_time:.1f}s"
        
        # Valutazione finale dell'evaluator
        report += f"\n\nEVALUATOR'S RECOMMENDATION:"
        
        if avg_quality >= 0.80:
            report += "\n🌟 EXCELLENT - Evaluator observed high-quality responses consistently"
            report += "\n   This model showed strong understanding and good execution"
            report += "\n   Recommended score: 8-10"
        elif avg_quality >= 0.65:
            report += "\n✅ GOOD - Evaluator found solid performance across tasks"
            report += "\n   This model handled most questions well"
            report += "\n   Recommended score: 6-7"
        elif avg_quality >= 0.50:
            report += "\n✓ ADEQUATE - Evaluator noted acceptable but not impressive responses"
            report += "\n   This model can handle basic tasks"
            report += "\n   Recommended score: 4-5"
        else:
            report += "\n⚠️ LIMITED - Evaluator observed significant issues"
            report += "\n   This model struggled with most tasks"
            report += "\n   Recommended score: 2-3"
        
        report += f"\n\n{'='*70}"
        
        return report
    
    def _size_based_fallback(self, size_gb: float) -> int:
        """Fallback basato solo su dimensione"""
        if size_gb < 1:
            return 3
        elif size_gb < 3:
            return 4
        elif size_gb < 5:
            return 5
        elif size_gb < 8:
            return 6
        elif size_gb < 12:
            return 7
        elif size_gb < 15:
            return 8
        elif size_gb < 20:
            return 9
        else:
            return 10

# ====================
# LLM-USE CORE WITH PRODUCTION FEATURES
# ====================

class LlmUse:
    """Sistema di routing universale con context preservation e production features"""
    
    def __init__(self, verbose: bool = True, enable_production: bool = True):
        self.verbose = verbose
        self.enable_production = enable_production
        
        # Initialize production components
        if enable_production:
            self.logger = ProductionLogger()
            self.db = ProductionDB()
            self.cache = ProductionCache(logger=self.logger)
            self.metrics = ProductionMetrics(logger=self.logger)
            self.rate_limiter = ProductionRateLimiter(logger=self.logger, db=self.db)
            
            # Create session
            self.session_id = self.db.create_session()
            self.logger.info(f"Production session created: {self.session_id}")
        else:
            self.logger = None
            self.db = None
            self.cache = None
            self.metrics = None
            self.rate_limiter = None
            self.session_id = None
        
        # Original initialization
        self.providers = self._initialize_providers()
        self.available_models = self._detect_available_models()
        self.context = ConversationContext()
        self.current_model = None
        self.stats = {
            "total_messages": 0,
            "total_cost": 0.0,
            "models_used": {},
            "provider_distribution": {}
        }
        
        self.router = self._setup_router()
        
        if verbose:
            self._print_initialization()
    
    def _initialize_providers(self) -> Dict[str, LLMProvider]:
        """Inizializza tutti i provider"""
        return {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "groq": GroqProvider(),
            "google": GoogleProvider(),
            "deepseek": DeepSeekProvider(),
            "ollama": OllamaProvider()
        }
    
    def _detect_available_models(self) -> Dict[str, ModelConfig]:
        """Rileva quali modelli sono realmente disponibili"""
        available = {}
        
        for model_name, config in MODEL_CATALOG.items():
            provider = self.providers.get(config.provider)
            
            if provider and provider.is_available():
                available[model_name] = config
        
        return available
    
    def _setup_router(self):
        """Setup del router per valutazione complessità"""
        if self.providers["ollama"].is_available():
            models = self.providers["ollama"].list_models()
            for model in models:
                if any(name in model.lower() for name in ["mistral", "gemma", "phi"]):
                    if self.verbose:
                        print(f"   Using {model} for complexity evaluation")
                    # PASSA verbose AL COSTRUTTORE
                    return MistralComplexityEvaluator(
                        self.providers["ollama"], 
                        verbose=self.verbose  # AGGIUNGI QUESTO!
                    )
        
        if self.verbose:
            print("   Using heuristic complexity evaluator")
        return HeuristicComplexityEvaluator()
    
    def _print_initialization(self):
        """Stampa stato inizializzazione"""
        print("\n" + "="*70)
        print("🌐 LLM-USE - Universal LLM Router")
        if self.enable_production:
            print("   🚀 Production Mode: ENABLED")
            print(f"   📊 Session ID: {self.session_id}")
        print("="*70)
        
        available_providers = [name for name, p in self.providers.items() if p.is_available()]
        print(f"\n✅ Active providers: {', '.join(available_providers)}")
        
        print(f"\n📦 Available models ({len(self.available_models)} total):")
        
        by_provider = {}
        for model_name, config in self.available_models.items():
            if config.provider not in by_provider:
                by_provider[config.provider] = []
            by_provider[config.provider].append(model_name)
        
        for provider, models in sorted(by_provider.items()):
            print(f"\n  {provider.upper()}:")
            for model in models[:3]:
                config = self.available_models[model]
                cost = "FREE" if config.cost_per_1k_input == 0 else f"${config.cost_per_1k_input}/1K"
                print(f"    • {config.display_name} ({cost})")
            if len(models) > 3:
                print(f"    ... and {len(models)-3} more")
        
        print("\n" + "="*70 + "\n")
    
    def evaluate_and_route(self, user_input: str) -> str:
        """Valuta complessità e seleziona modello ottimale"""
        complexity = self.router.evaluate(user_input)
        
        if self.metrics:
            self.metrics.histogram("complexity_score", complexity)
        
        selected_model = self._select_best_model(complexity, user_input)
        
        if self.verbose and selected_model != self.current_model:
            print(f"\n🔄 Switching to: {self.available_models[selected_model].display_name}")
            print(f"   Complexity: {complexity}/10")
            print(f"   Reason: {self._get_selection_reason(complexity)}")
            
            if self.metrics:
                self.metrics.increment("model_switches")
        
        if self.metrics:
            self.metrics.increment("routing_decisions")
        
        self.current_model = selected_model
        return selected_model
    
    def _select_best_model(self, complexity: int, user_input: str) -> str:
        """
        🤖 ROUTING COMPLETAMENTE QUALITATIVO
        Usa un agent per scegliere il modello migliore basandosi sui profili
        """
        if not self.available_models:
            raise Exception("No models available!")
        
        # 🤖 L'AGENT SCEGLIE basandosi sui profili qualitativi
        selected = self._agent_route_to_best_model(user_input, complexity)
        
        if selected:
            return selected
        
        # Fallback: scegli per complexity range
        suitable = []
        for model_id, config in self.available_models.items():
            min_c, max_c = config.complexity_range
            if min_c <= complexity <= max_c:
                suitable.append((model_id, config))
        
        if not suitable:
            suitable = list(self.available_models.items())
        
        # Ordina per tier
        tier_priority = {"expert": 4, "professional": 3, "competent": 2, "basic": 1, "minimal": 0}
        suitable.sort(key=lambda x: tier_priority.get(x[1].tier, 0), reverse=True)
        
        return suitable[0][0]
    
    def _agent_route_to_best_model(self, user_input: str, complexity: int) -> Optional[str]:
        """🤖 AGENT sceglie il modello migliore basandosi sui PROFILI QUALITATIVI"""
        
        # FIX: Verifica che il router abbia un provider valido
        if not hasattr(self.router, 'provider'):
            if self.verbose:
                print("⚠️ [Routing] Router has no provider, using fallback")
            return None
        
        if not self.router.provider:
            if self.verbose:
                print("⚠️ [Routing] Router provider is None, using fallback")
            return None
        
        # Verifica che il provider sia disponibile
        try:
            if not self.router.provider.is_available():
                if self.verbose:
                    print("⚠️ [Routing] Router provider not available, using fallback")
                return None
        except Exception as e:
            if self.verbose:
                print(f"⚠️ [Routing] Error checking provider: {e}, using fallback")
            return None
        
        # Costruisci la descrizione di tutti i modelli disponibili
        models_description = ""
        model_list = []
        
        # 🆕 Organizza i modelli per tier per aiutare l'agent
        models_by_tier = {'minimal': [], 'basic': [], 'competent': [], 'professional': [], 'expert': []}
        
        for i, (model_id, config) in enumerate(self.available_models.items(), 1):
            tier = config.tier if hasattr(config, 'tier') else 'unknown'
            
            models_description += f"\n{'='*60}\n"
            models_description += f"MODEL {i}: {config.display_name}\n"
            models_description += f"{'='*60}\n"
            models_description += f"Tier: {tier.upper()}\n"
            models_description += f"Speed: {config.speed}\n"
            models_description += f"Complexity range: {config.complexity_range[0]}-{config.complexity_range[1]}\n"
            models_description += f"Size: {config.size_gb:.1f}GB\n"
            
            models_description += f"\nBEST FOR:\n"
            for use_case in config.best_for[:5]:
                models_description += f"  ✓ {use_case}\n"
            
            if config.avoid_for:
                models_description += f"\nAVOID FOR:\n"
                for use_case in config.avoid_for[:3]:
                    models_description += f"  ✗ {use_case}\n"
            
            if config.personality:
                models_description += f"\nPersonality: {config.personality[:100]}\n"
            
            model_list.append(model_id)
            
            # Aggiungi alla categorizzazione
            if tier in models_by_tier:
                models_by_tier[tier].append(f"Model {i}: {config.display_name} ({config.size_gb:.1f}GB)")
        
        # Se non ci sono modelli, ritorna None
        if not model_list:
            if self.verbose:
                print("⚠️ [Routing] No models available for routing")
            return None
        
        # 🆕 Crea un summary per tier per aiutare l'agent
        tier_summary = "\n📊 MODELS BY EFFICIENCY TIER:\n"
        tier_summary += "================================\n"
        for tier in ['minimal', 'basic', 'competent', 'professional', 'expert']:
            if models_by_tier[tier]:
                tier_summary += f"\n{tier.upper()} (for complexity {self._get_tier_complexity_range(tier)}):\n"
                for model in models_by_tier[tier]:
                    tier_summary += f"  • {model}\n"
        
        # 🤖 PROMPT OTTIMIZZATO PER ROUTING INTELLIGENTE
        routing_prompt = f"""<role>
You are a Model Selection Expert. Your task: match COGNITIVE COMPLEXITY of the request to MODEL CAPABILITY.
</role>

<fundamental_principle>
Every request has an INTRINSIC COGNITIVE LOAD - the mental effort needed to produce a quality answer.
Your job: Estimate this load, then select the model with matching cognitive capability.
</fundamental_principle>

<user_request>
"{user_input}" ← SOLO QUESTO, non la storia
Estimated complexity score: {complexity}/10
</user_request>

<cognitive_load_framework>
Analyze the request across these universal dimensions:

1. INFORMATION PROCESSING DEPTH
   └─ Surface (recall/repeat) vs Deep (analyze/synthesize/create)

2. KNOWLEDGE DOMAIN LEVEL  
   └─ Common (everyone knows) vs Specialized (experts know) vs Cutting-edge (researchers know)

3. REASONING COMPLEXITY
   └─ Zero-step (direct) vs Multi-step (sequential) vs Parallel (simultaneous considerations)

4. OUTPUT REQUIREMENTS
   └─ Simple (word/phrase) vs Structured (paragraph/code) vs Rigorous (proof/formal)

5. ABSTRACTION LEVEL
   └─ Concrete (specific facts) vs Abstract (concepts/patterns/principles)

6. CREATIVE/GENERATIVE DEMAND
   └─ Reproduce (existing) vs Adapt (modify) vs Innovate (create new)
</cognitive_load_framework>

<capability_tiers>
Match cognitive load to model tier:

MINIMAL CAPABILITY (Basic models):
• Information: Surface-level recall
• Knowledge: Common, everyday concepts  
• Reasoning: Zero or single-step
• Output: Simple, direct responses
• Abstraction: Concrete, specific
• Generation: Pure reproduction
→ Complexity: 1-3/10

MODERATE CAPABILITY (Professional models):
• Information: Moderate depth analysis
• Knowledge: Technical but standard
• Reasoning: Multi-step sequential
• Output: Structured, organized
• Abstraction: Mix concrete and conceptual
• Generation: Adaptation of known patterns
→ Complexity: 4-7/10

MAXIMUM CAPABILITY (Expert models):
• Information: Deep synthesis and creation
• Knowledge: Specialized or cutting-edge
• Reasoning: Complex parallel reasoning
• Output: Rigorous, formal, innovative
• Abstraction: High-level principles
• Generation: Novel solutions/proofs
→ Complexity: 8-10/10
</capability_tiers>

<universal_indicators>
Indicators of MINIMAL load (use smallest models):
• Request seeks single fact/definition
• Expected output is 1-5 words
• No reasoning chain required
• Common knowledge domain
• Pure recall/recognition task

Indicators of MAXIMUM load (use largest models):
• Request requires PROVING something
• Request requires DESIGNING/ARCHITECTING something
• Request requires DERIVING from first principles
• Request uses specialized academic/research terminology
• Request demands mathematical/logical rigor
• Request asks to CREATE something novel
• Multiple domains must be synthesized
</universal_indicators>

{tier_summary}

<available_models_detail>
{models_description}
</available_models_detail>

<selection_methodology>
Step 1: DECOMPOSE the request
- What TYPE of mental operation is required?
- What DEPTH of processing is needed?
- What KNOWLEDGE level is assumed?

Step 2: MAP to cognitive load
- Is this a recall task? → Low load
- Is this an explanation task? → Medium load  
- Is this a creation/proof task? → High load

Step 3: MATCH to model capability
- Low load → Basic tier sufficient
- Medium load → Professional tier needed
- High load → Expert tier required

Step 4: SELECT specific model
- Within tier, choose based on:
  * For low load: smallest/fastest
  * For medium load: balanced
  * For high load: most capable

Step 5: SANITY CHECK
- "Would a smaller tier FAIL at this task?"
  * If NO → use smaller tier
  * If YES → current selection correct
- "Does this tier have the KNOWLEDGE needed?"
  * If NO → go up one tier
  * If YES → selection correct
</selection_methodology>

<meta_reasoning>
Ask yourself these universal questions:

Q1: "What would happen if I gave this to the smallest model?"
└─ If answer would be PERFECT → use smallest
└─ If answer would be WRONG → need larger
└─ If answer would be INCOMPLETE → need larger

Q2: "What makes this request HARD?"
└─ Nothing → Basic tier
└─ Technical knowledge → Professional tier  
└─ Theoretical rigor → Expert tier

Q3: "What is the user REALLY asking for?"
└─ A fact → Basic
└─ An explanation → Professional
└─ A proof/design/innovation → Expert

Q4: "Could this request appear in..."
└─ Casual conversation → Basic
└─ Professional work → Professional
└─ Academic research → Expert
</meta_reasoning>

<output_format>
Reason through:
1. What cognitive operations does this require?
2. What tier has those capabilities?
3. Which specific model in that tier?

Then output:
SELECTED: [number]
REASON: [explain the cognitive match]
</output_format>

ANALYZE THE COGNITIVE LOAD AND SELECT:"""

        try:
            # L'agent sceglie con temperature ancora più bassa per consistency
            response = self.router.provider.chat(
                messages=[
                    {
                        "role": "system", 
                        "content": """You are an expert Model Router. 
            
Your goal: Select the model that will give the BEST QUALITY answer for the given request.

Rules:
1. Match task complexity to model capability
2. For simple tasks (greetings, definitions): Basic models are PERFECT
3. For technical tasks (code, analysis): Professional models needed
4. For expert tasks (proofs, CUDA, architecture): ONLY expert models work
5. When in doubt about complexity: choose the MORE capable model

Focus on answer QUALITY, not efficiency."""
                    },
                    {
                        "role": "user", 
                        "content": routing_prompt
                    }
                ],
                model="mistral",  # O il tuo modello di routing preferito
                temperature=0.05,  # Molto bassa per decisioni consistenti
                max_tokens=800
            )
            
            # Gestisci la risposta che potrebbe essere un dict o una stringa
            if isinstance(response, dict):
                response_text = response.get('message', {}).get('content', '')
            else:
                response_text = str(response)
            
            # 🆕 Debug: mostra il ragionamento dell'agent
            if self.verbose and len(response_text) > 0:
                # Estrai il ragionamento iniziale
                lines = response_text.split('\n')
                for line in lines[:5]:  # Prime 5 righe del ragionamento
                    if line.strip():
                        print(f"   🤔 {line.strip()}")
            
            # Estrai la scelta
            import re
            
            # Cerca "SELECTED: X"
            patterns = [
                r'SELECTED:\s*\[?[Mm]odel\s*(\d+)\]?',  # [Model 7] o Model 7
                r'SELECTED:\s*(\d+)',                     # Solo numero
                r'[Mm]odel\s*(\d+).*(?:selected|choose|pick)', # Model 7 selected/chosen
                r'(?:select|choose|pick).*[Mm]odel\s*(\d+)',   # I select Model 7
            ]

            match = None
            for pattern in patterns:
                match = re.search(pattern, response_text, re.IGNORECASE)
                if match:
                    break
            if match:
                selected_num = int(match.group(1))
                if 1 <= selected_num <= len(model_list):
                    selected_model_id = model_list[selected_num - 1]
                    
                    # Estrai il reasoning
                    reason_match = re.search(r'REASON:\s*(.+?)(?:\n|$)', response_text, re.IGNORECASE | re.DOTALL)
                    reason = reason_match.group(1).strip() if reason_match else "Agent selected"
                    
                    if self.verbose:
                        selected_config = self.available_models[selected_model_id]
                        print(f"\n🎯 [Agent Routing Decision]")
                        print(f"   Selected: {selected_config.display_name} ({selected_config.size_gb:.1f}GB)")
                        print(f"   Tier: {selected_config.tier}")
                        print(f"   Reason: {reason[:150]}")
                        
                        # 🆕 Avviso se sembra overengineered
                        if complexity <= 2 and selected_config.tier in ['professional', 'expert']:
                            print(f"   ⚠️ Warning: Expert model for simple request?")
                    
                    return selected_model_id
                else:
                    if self.verbose:
                        print(f"⚠️ [Routing] Agent selected invalid model number: {selected_num}")
            else:
                if self.verbose:
                    print(f"⚠️ [Routing] Agent didn't select clearly")
                    if len(response_text) > 0:
                        print(f"   Response preview: {response_text[:200]}...")
            
            return None
            
        except Exception as e:
            if self.verbose:
                print(f"❌ [Routing] Agent error: {str(e)}")
                import traceback
                traceback.print_exc()
            return None

    def _get_tier_complexity_range(self, tier: str) -> str:
        """Helper per ottenere il range di complessità per tier"""
        ranges = {
            'minimal': '1-2',
            'basic': '2-4', 
            'competent': '4-6',
            'professional': '6-8',
            'expert': '8-10'
        }
        return ranges.get(tier, '?-?')
    
    def _fallback_chat(self, user_input: str, exclude: str) -> str:
        """Fallback su altro modello se il principale fallisce"""
        fallback_models = [m for m in self.available_models.keys() if m != exclude]
        
        if not fallback_models:
            return "Sorry, no models are currently available."
        
        fallback = fallback_models[0]
        config = self.available_models[fallback]
        provider = self.providers[config.provider]
        
        if self.verbose:
            print(f"   Trying fallback: {config.display_name}")
        
        if self.logger:
            self.logger.info(f"Using fallback model: {config.display_name}")
        
        try:
            response = provider.chat(
                messages=self.context.get_messages_for_api(),
                model=config.model_id
            )
            
            self.context.add_message("assistant", response, fallback)
            return response
            
        except:
            return "I'm having trouble connecting to the models. Please try again."
    
    def _update_stats(self, model: str, config: ModelConfig, input_len: int, 
                     output_len: int, time: float, cache_hit: bool = False):
        """Aggiorna statistiche"""
        self.stats["total_messages"] += 1
        
        input_tokens = input_len / 4
        output_tokens = output_len / 4
        cost = (input_tokens * config.cost_per_1k_input + 
               output_tokens * config.cost_per_1k_output) / 1000
        self.stats["total_cost"] += cost
        
        if model not in self.stats["models_used"]:
            self.stats["models_used"][model] = 0
        self.stats["models_used"][model] += 1
        
        if config.provider not in self.stats["provider_distribution"]:
            self.stats["provider_distribution"][config.provider] = 0
        self.stats["provider_distribution"][config.provider] += 1
    
    def _print_response_info(self, config: ModelConfig, elapsed: float):
        """Stampa info sulla risposta"""
        cost_str = "FREE" if config.cost_per_1k_input == 0 else f"~${self.stats['total_cost']:.6f} total"
        print(f"   ⚡ {elapsed:.2f}s | {cost_str} | Context: {self.context.get_summary()}")
    
    def get_stats(self) -> Dict:
        """Ritorna statistiche sessione"""
        stats = {
            **self.stats,
            "context_summary": self.context.get_summary(),
            "current_model": self.current_model,
            "available_models": len(self.available_models)
        }
        
        # Add production stats if enabled
        if self.enable_production:
            if self.cache:
                stats["cache"] = self.cache.get_stats()
            if self.metrics:
                stats["metrics"] = self.metrics.get_summary()
            if self.rate_limiter:
                stats["rate_limits"] = self.rate_limiter.get_limits_info()
            if self.db:
                stats["session"] = self.db.get_session_stats(self.session_id)
        
        return stats
    
    def clear_context(self):
        """Pulisce il contesto mantenendo le stats"""
        self.context = ConversationContext()
        if self.verbose:
            print("\n🗑️  Context cleared! Starting fresh conversation.")
    
    def save_conversation(self, filepath: str = None):
        """Salva conversazione su file"""
        if not filepath:
            filepath = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "messages": self.context.messages,
            "stats": self.stats,
            "model_history": self.context.model_history
        }
        
        if self.enable_production and self.db:
            data["session_data"] = self.db.get_session_stats(self.session_id)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        if self.verbose:
            print(f"\n💾 Conversation saved to {filepath}")
    
    def get_analytics(self, hours: int = 24) -> Dict:
        """Get production analytics"""
        if not self.enable_production or not self.db:
            return {}
        
        return {
            "model_performance": self.db.get_model_analytics(hours),
            "error_analysis": self.db.get_error_analysis(hours),
            "cache_stats": self.cache.get_stats() if self.cache else {},
            "metrics_summary": self.metrics.get_summary() if self.metrics else {}
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.enable_production:
            if self.db:
                self.db.update_session(self.session_id, end_time=True)
                self.db.cleanup_old_data(days=30)
            
            if self.logger:
                self.logger.info("Session ended", extra={
                    "session_id": self.session_id,
                    "total_messages": self.stats["total_messages"],
                    "total_cost": self.stats["total_cost"]
                })

# ====================
# LLM-USE V2 WITH AUTO-DISCOVERY
# ====================

class LLMUSEV2(LlmUse):
    """llm-use con auto-discovery, profiling qualitativo e production features"""
    
    def __init__(self, verbose: bool = True, auto_discover: bool = True, 
                 enable_production: bool = True):
        # Initialize base class with production features
        super().__init__(verbose=False, enable_production=enable_production)
        
        self.verbose = verbose
        self.auto_discover = auto_discover
        
        # SALVA I PROFILI QUALITATIVI
        self.discovered_profiles = {}
        
        # Discover models
        if auto_discover:
            self._discover_new_models()
        
        if verbose:
            self._print_initialization()
    
    def _discover_new_models(self):
        """Scopre e aggiunge nuovi modelli al catalogo"""
        if self.verbose:
            print("\n🔬 AUTO-DISCOVERY MODE")
            print("="*50)
        
        discoverer = ModelAutoDiscovery(verbose=self.verbose)
        
        discovered = discoverer.discover_and_benchmark_ollama_models()
        
        for model_id, config in discovered.items():
            self.available_models[model_id] = config
            
            # SALVA IL PROFILO
            if model_id in discoverer.benchmarks:
                self.discovered_profiles[model_id] = discoverer.benchmarks[model_id]
        
        if self.verbose and discovered:
            print(f"\n✅ Added {len(discovered)} Ollama models to catalog")
            print("="*50)
    
    def refresh_models(self):
        """Forza re-discovery dei modelli"""
        if self.verbose:
            print("\n🔄 Refreshing model catalog...")
        
        discoverer = ModelAutoDiscovery(verbose=self.verbose)
        discoverer.benchmarks = {}
        discoverer._save_benchmarks()
        
        self._discover_new_models()

# ====================
# INTERACTIVE INTERFACE
# ====================

def interactive_chat():
    """Chat interattiva con routing automatico e production features"""
    
    print("\n" + "🚀"*35)
    print("  LLM-USE PRODUCTION - INTELLIGENT ROUTING")
    print("  Production monitoring and caching enabled")
    print("🚀"*35 + "\n")
    
    # Setup with production features
    print("Initializing production system with auto-discovery...")
    router = LLMUSEV2(verbose=True, auto_discover=True, enable_production=True)
    
    if not router.available_models:
        print("\n❌ No models available! Please:")
        print("   • Set API keys (export OPENAI_API_KEY=...)")
        print("   • Or install Ollama models (ollama pull mistral)")
        return
    
    print("\n📝 Commands:")
    print("  /stats    - Show session statistics")
    print("  /analytics - Show production analytics")
    print("  /clear    - Clear conversation context")
    print("  /save     - Save conversation")
    print("  /models   - List all available models")
    print("  /refresh  - Re-discover models")
    print("  /quit     - Exit\n")
    
    print("💬 Start chatting! I'll automatically choose the best model for each task.\n")
    
    try:
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == '/quit':
                    print("\n👋 Goodbye!")
                    break
                
                elif user_input.lower() == '/stats':
                    stats = router.get_stats()
                    print("\n📊 SESSION STATS:")
                    print(f"   Messages: {stats['total_messages']}")
                    print(f"   Total cost: ${stats['total_cost']:.6f}")
                    print(f"   Models used: {len(stats['models_used'])}")
                    print(f"   {stats['context_summary']}")
                    
                    if 'cache' in stats:
                        cache_stats = stats['cache']
                        print(f"\n   Cache: {cache_stats['hit_rate']*100:.1f}% hit rate")
                        print(f"   Size: {cache_stats['current_size_mb']:.1f}/{cache_stats['max_size_mb']}MB")
                    
                    if 'metrics' in stats:
                        metrics = stats['metrics']
                        print(f"\n   Avg latency: {metrics['avg_latency_ms']:.0f}ms")
                        print(f"   Error rate: {metrics['error_rate']*100:.1f}%")
                    
                    if stats['models_used']:
                        print("\n   Model distribution:")
                        for model, count in sorted(stats['models_used'].items(), 
                                                  key=lambda x: x[1], reverse=True):
                            if model in router.available_models:
                                print(f"     • {router.available_models[model].display_name}: {count}x")
                    continue
                
                elif user_input.lower() == '/analytics':
                    analytics = router.get_analytics(hours=24)
                    
                    if analytics:
                        print("\n📈 PRODUCTION ANALYTICS (24h):")
                        
                        if 'model_performance' in analytics:
                            print("\n   Model Performance:")
                            for model_data in analytics['model_performance'][:5]:
                                print(f"     • {model_data['model']}: "
                                     f"{model_data['total_calls']} calls, "
                                     f"{model_data['avg_latency']:.0f}ms avg, "
                                     f"{model_data['success_rate']:.1f}% success")
                        
                        if 'error_analysis' in analytics:
                            errors = analytics['error_analysis']
                            if errors:
                                print(f"\n   Recent Errors:")
                                for error in errors[:3]:
                                    print(f"     • {error['model']}: {error['error_message'][:50]}... "
                                         f"({error['error_count']}x)")
                    continue
                
                elif user_input.lower() == '/clear':
                    router.clear_context()
                    continue
                
                elif user_input.lower() == '/save':
                    router.save_conversation()
                    continue
                
                elif user_input.lower() == '/models':
                    print("\n📦 AVAILABLE MODELS:")
                    
                    by_provider = {}
                    for name, config in router.available_models.items():
                        if config.provider not in by_provider:
                            by_provider[config.provider] = []
                        by_provider[config.provider].append((name, config))
                    
                    for provider, models in sorted(by_provider.items()):
                        print(f"\n  {provider.upper()} ({len(models)} models):")
                        for name, config in sorted(models, key=lambda x: -x[1].quality_score):
                            cost = "FREE" if config.cost_per_1k_input == 0 else f"${config.cost_per_1k_input}/1K"
                            size_str = f"{config.size_gb:.1f}GB" if hasattr(config, 'size_gb') and config.size_gb > 0 else "unknown"
                            print(f"    • {config.display_name}")
                            print(f"      Quality: {config.quality_score}/10 | Speed: {config.speed} | Size: {size_str} | Cost: {cost}")
                            print(f"      Best for: {', '.join(config.best_for[:3])}")
                    continue
                
                elif user_input.lower() == '/refresh':
                    router.refresh_models()
                    continue
                
                # Normal chat
                print("\n🤖 Assistant: ", end="")
                response = router.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted! Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                if router.logger:
                    router.logger.error(f"Chat error: {e}")
    
    finally:
        # Cleanup on exit
        if hasattr(router, 'cleanup'):
            router.cleanup()

# ====================
# MAIN ENTRY POINT
# ====================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "test":
            # Test mode - simplified for production
            print("Running production tests...")
            router = LLMUSEV2(verbose=True, auto_discover=True, enable_production=True)
            
            test_inputs = [
                "Hi!",
                "What's 2+2?",
                "Explain quantum computing",
                "Write a Python web scraper"
            ]
            
            for test_input in test_inputs:
                print(f"\nTest: {test_input}")
                response = router.chat(test_input)
                print(f"Response: {response[:100]}...")
            
            # Print analytics
            print("\n" + "="*70)
            print("TEST ANALYTICS:")
            print(json.dumps(router.get_stats(), indent=2))
            
            router.cleanup()
            
        elif command == "benchmark":
            discoverer = ModelAutoDiscovery(verbose=True)
            discoverer.discover_and_benchmark_ollama_models(force_benchmark=True)
            
        elif command == "analytics":
            # Show stored analytics
            db = ProductionDB()
            analytics = {
                "model_performance": db.get_model_analytics(24),
                "error_analysis": db.get_error_analysis(24)
            }
            print(json.dumps(analytics, indent=2, default=str))
            
        elif command == "help":
            print("\nUsage:")
            print("  python llm-use-production.py          # Interactive chat")
            print("  python llm-use-production.py test     # Run tests")
            print("  python llm-use-production.py benchmark# Benchmark models")
            print("  python llm-use-production.py analytics# View analytics")
            print("  python llm-use-production.py help     # Show this help")
        else:
            print(f"Unknown command: {command}")
    else:
        interactive_chat()
