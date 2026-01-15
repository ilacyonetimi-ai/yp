#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaTniX AI Utils - Utility functions and helpers
"""

import os
import sys
import json
import logging
import time
import datetime
import hashlib
import base64
import mimetypes
import re
import uuid
import subprocess
import threading
import queue
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import zipfile
import tarfile
import rarfile
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import schedule
import psutil
import GPUtil
import requests
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename

# Google Drive integration
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseDownload
import io

@dataclass
class SystemInfo:
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    gpu_info: List[Dict[str, Any]]
    network_info: Dict[str, Any]
    uptime: str

@dataclass
class TaskResult:
    task_id: str
    status: str
    result: Any
    error: Optional[str]
    start_time: datetime.datetime
    end_time: Optional[datetime.datetime]

class FileManager:
    """Advanced file management utilities"""
    
    def __init__(self, base_path: str = "/tmp"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self.supported_formats = {
            'text': ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv'],
            'image': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp', '.tiff'],
            'audio': ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'],
            'video': ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'],
            'document': ['.pdf', '.docx', '.xlsx', '.pptx', '.odt', '.ods', '.odp'],
            'archive': ['.zip', '.rar', '.tar.gz', '.7z', '.tar', '.gz']
        }
        
    def get_file_type(self, file_path: str) -> str:
        """Determine file type"""
        ext = Path(file_path).suffix.lower()
        for file_type, extensions in self.supported_formats.items():
            if ext in extensions:
                return file_type
        return 'unknown'
        
    def secure_filename(self, filename: str) -> str:
        """Generate secure filename"""
        return secure_filename(filename)
        
    def create_backup(self, file_path: str, backup_dir: str = None) -> str:
        """Create backup of file"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        if not backup_dir:
            backup_dir = self.base_path / 'backups'
        else:
            backup_dir = Path(backup_dir)
            
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        return str(backup_path)
        
    def batch_rename(self, directory: str, pattern: str, replacement: str) -> List[str]:
        """Batch rename files in directory"""
        directory = Path(directory)
        renamed_files = []
        
        for file_path in directory.iterdir():
            if file_path.is_file():
                new_name = re.sub(pattern, replacement, file_path.name)
                new_path = file_path.parent / new_name
                
                if new_path != file_path:
                    file_path.rename(new_path)
                    renamed_files.append(str(new_path))
                    
        return renamed_files
        
    def find_duplicates(self, directory: str) -> Dict[str, List[str]]:
        """Find duplicate files"""
        directory = Path(directory)
        file_hashes = {}
        duplicates = {}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                file_hash = self._calculate_file_hash(file_path)
                
                if file_hash in file_hashes:
                    if file_hash not in duplicates:
                        duplicates[file_hash] = [file_hashes[file_hash]]
                    duplicates[file_hash].append(str(file_path))
                else:
                    file_hashes[file_hash] = str(file_path)
                    
        return duplicates
        
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def organize_files(self, source_dir: str, target_dir: str, organize_by: str = 'type'):
        """Organize files by type, date, or size"""
        source_dir = Path(source_dir)
        target_dir = Path(target_dir)
        target_dir.mkdir(exist_ok=True)
        
        organized_files = []
        
        for file_path in source_dir.rglob('*'):
            if file_path.is_file():
                if organize_by == 'type':
                    file_type = self.get_file_type(file_path)
                    sub_dir = target_dir / file_type
                elif organize_by == 'date':
                    date_str = datetime.datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m")
                    sub_dir = target_dir / date_str
                elif organize_by == 'size':
                    size = file_path.stat().st_size
                    if size < 1024 * 1024:  # < 1MB
                        sub_dir = target_dir / 'small'
                    elif size < 1024 * 1024 * 100:  # < 100MB
                        sub_dir = target_dir / 'medium'
                    else:
                        sub_dir = target_dir / 'large'
                else:
                    sub_dir = target_dir / 'others'
                    
                sub_dir.mkdir(exist_ok=True)
                new_path = sub_dir / file_path.name
                
                # Handle name conflicts
                counter = 1
                while new_path.exists():
                    stem = file_path.stem
                    suffix = file_path.suffix
                    new_path = sub_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                    
                shutil.move(str(file_path), str(new_path))
                organized_files.append(str(new_path))
                
        return organized_files

class ImageProcessor:
    """Advanced image processing utilities"""
    
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp', '.tiff']
        
    def resize_image(self, image_path: str, output_path: str, width: int, height: int, maintain_aspect: bool = True) -> str:
        """Resize image"""
        with Image.open(image_path) as img:
            if maintain_aspect:
                img.thumbnail((width, height), Image.Resampling.LANCZOS)
            else:
                img = img.resize((width, height), Image.Resampling.LANCZOS)
                
            img.save(output_path)
            return output_path
            
    def convert_format(self, image_path: str, output_path: str, target_format: str) -> str:
        """Convert image format"""
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'LA'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background
                
            img.save(output_path, format=target_format.upper())
            return output_path
            
    def apply_filters(self, image_path: str, output_path: str, filters: List[str]) -> str:
        """Apply filters to image"""
        with Image.open(image_path) as img:
            for filter_name in filters:
                if filter_name == 'blur':
                    img = img.filter(ImageFilter.BLUR)
                elif filter_name == 'sharpen':
                    img = img.filter(ImageFilter.SHARPEN)
                elif filter_name == 'edge_enhance':
                    img = img.filter(ImageFilter.EDGE_ENHANCE)
                elif filter_name == 'brightness':
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(1.2)
                elif filter_name == 'contrast':
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(1.2)
                elif filter_name == 'saturation':
                    enhancer = ImageEnhance.Color(img)
                    img = enhancer.enhance(1.2)
                    
            img.save(output_path)
            return output_path
            
    def create_thumbnail(self, image_path: str, output_path: str, size: tuple = (150, 150)) -> str:
        """Create thumbnail"""
        with Image.open(image_path) as img:
            img.thumbnail(size, Image.Resampling.LANCZOS)
            img.save(output_path)
            return output_path
            
    def batch_process_images(self, directory: str, operation: str, **kwargs) -> List[str]:
        """Batch process images in directory"""
        directory = Path(directory)
        processed_files = []
        
        for image_path in directory.iterdir():
            if image_path.suffix.lower() in self.supported_formats:
                output_path = image_path.parent / f"processed_{image_path.name}"
                
                if operation == 'resize':
                    self.resize_image(str(image_path), str(output_path), kwargs.get('width', 800), kwargs.get('height', 600))
                elif operation == 'convert':
                    self.convert_format(str(image_path), str(output_path), kwargs.get('format', 'PNG'))
                elif operation == 'filters':
                    self.apply_filters(str(image_path), str(output_path), kwargs.get('filters', []))
                elif operation == 'thumbnail':
                    self.create_thumbnail(str(image_path), str(output_path), kwargs.get('size', (150, 150)))
                    
                processed_files.append(str(output_path))
                
        return processed_files

class SystemMonitor:
    """System monitoring utilities"""
    
    def __init__(self):
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks = []
        
    def get_system_info(self) -> SystemInfo:
        """Get comprehensive system information"""
        # CPU info
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory info
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk info
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # GPU info
        gpu_info = []
        try:
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_info.append({
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_used': gpu.memoryUsed,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                    'load': gpu.load * 100
                })
        except:
            gpu_info = [{'error': 'GPU info not available'}]
            
        # Network info
        network = psutil.net_io_counters()
        network_info = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        # Uptime
        boot_time = datetime.datetime.fromtimestamp(psutil.boot_time())
        uptime = str(datetime.datetime.now() - boot_time)
        
        return SystemInfo(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage=disk_usage,
            gpu_info=gpu_info,
            network_info=network_info,
            uptime=uptime
        )
        
    def start_monitoring(self, interval: int = 60):
        """Start system monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self, interval: int):
        """Monitoring loop"""
        while self.monitoring:
            try:
                system_info = self.get_system_info()
                
                # Call registered callbacks
                for callback in self.callbacks:
                    callback(system_info)
                    
                time.sleep(interval)
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(interval)
                
    def add_callback(self, callback: Callable[[SystemInfo], None]):
        """Add monitoring callback"""
        self.callbacks.append(callback)
        
    def remove_callback(self, callback: Callable[[SystemInfo], None]):
        """Remove monitoring callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

class TaskScheduler:
    """Advanced task scheduling"""
    
    def __init__(self):
        self.tasks = {}
        self.running = False
        self.scheduler_thread = None
        
    def add_task(self, task_id: str, func: Callable, schedule_pattern: str, *args, **kwargs):
        """Add scheduled task"""
        if schedule_pattern.startswith('every '):
            parts = schedule_pattern.split()
            if len(parts) >= 2:
                interval = int(parts[1])
                unit = parts[2] if len(parts) > 2 else 'seconds'
                
                if unit.startswith('second'):
                    schedule.every(interval).seconds.do(func, *args, **kwargs)
                elif unit.startswith('minute'):
                    schedule.every(interval).minutes.do(func, *args, **kwargs)
                elif unit.startswith('hour'):
                    schedule.every(interval).hours.do(func, *args, **kwargs)
                elif unit.startswith('day'):
                    schedule.every(interval).days.do(func, *args, **kwargs)
                    
        self.tasks[task_id] = {
            'func': func,
            'schedule': schedule_pattern,
            'args': args,
            'kwargs': kwargs
        }
        
    def remove_task(self, task_id: str):
        """Remove scheduled task"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            schedule.clear(task_id)
            
    def start(self):
        """Start scheduler"""
        if self.running:
            return
            
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
    def stop(self):
        """Stop scheduler"""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
            
    def _scheduler_loop(self):
        """Scheduler loop"""
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                logging.error(f"Scheduler error: {e}")
                time.sleep(1)

class GoogleDriveManager:
    """Google Drive integration"""
    
    def __init__(self, credentials_path: str = None, token_path: str = 'token.json'):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.scopes = ['https://www.googleapis.com/auth/drive']
        self.service = None
        self._authenticate()
        
    def _authenticate(self):
        """Authenticate with Google Drive"""
        creds = None
        
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.scopes)
            
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not self.credentials_path:
                    raise ValueError("Credentials path required for initial authentication")
                    
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.scopes
                )
                creds = flow.run_local_server(port=0)
                
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())
                
        self.service = build('drive', 'v3', credentials=creds)
        
    def upload_file(self, file_path: str, folder_id: str = None) -> Dict[str, Any]:
        """Upload file to Google Drive"""
        file_path = Path(file_path)
        
        file_metadata = {
            'name': file_path.name
        }
        
        if folder_id:
            file_metadata['parents'] = [folder_id]
            
        media = MediaIoBaseUpload(
            io.BytesIO(file_path.read_bytes()),
            mimetype=mimetypes.guess_type(str(file_path))[0],
            resumable=True
        )
        
        file = self.service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,name,size,mimeType,createdTime'
        ).execute()
        
        return file
        
    def download_file(self, file_id: str, output_path: str) -> str:
        """Download file from Google Drive"""
        request = self.service.files().get_media(fileId=file_id)
        
        with open(output_path, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            
            while done is False:
                status, done = downloader.next_chunk()
                
        return output_path
        
    def list_files(self, folder_id: str = None, query: str = None) -> List[Dict[str, Any]]:
        """List files in Google Drive"""
        q = f"'{folder_id}' in parents" if folder_id else None
        
        if query:
            q = f"{q} and {query}" if q else query
            
        results = self.service.files().list(
            q=q,
            pageSize=1000,
            fields="nextPageToken, files(id, name, size, mimeType, createdTime, modifiedTime)"
        ).execute()
        
        return results.get('files', [])
        
    def create_folder(self, folder_name: str, parent_folder_id: str = None) -> Dict[str, Any]:
        """Create folder in Google Drive"""
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        
        if parent_folder_id:
            file_metadata['parents'] = [parent_folder_id]
            
        folder = self.service.files().create(
            body=file_metadata,
            fields='id,name,mimeType,createdTime'
        ).execute()
        
        return folder
        
    def delete_file(self, file_id: str) -> bool:
        """Delete file from Google Drive"""
        try:
            self.service.files().delete(fileId=file_id).execute()
            return True
        except:
            return False

class EmailNotifier:
    """Email notification utilities"""
    
    def __init__(self, smtp_server: str, smtp_port: int, username: str, password: str):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        
    def send_email(self, to_email: str, subject: str, body: str, attachments: List[str] = None) -> bool:
        """Send email with optional attachments"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.username
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            if attachments:
                for file_path in attachments:
                    with open(file_path, 'rb') as attachment:
                        part = MIMEBase('application', 'octet-stream')
                        part.set_payload(attachment.read())
                        
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {os.path.basename(file_path)}'
                    )
                    msg.attach(part)
                    
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.username, self.password)
            text = msg.as_string()
            server.sendmail(self.username, to_email, text)
            server.quit()
            
            return True
        except Exception as e:
            logging.error(f"Email sending error: {e}")
            return False

class DataAnalyzer:
    """Data analysis utilities"""
    
    def __init__(self):
        self.plots_dir = Path('plots')
        self.plots_dir.mkdir(exist_ok=True)
        
    def analyze_csv(self, file_path: str) -> Dict[str, Any]:
        """Analyze CSV file"""
        df = pd.read_csv(file_path)
        
        analysis = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'describe': df.describe().to_dict(),
            'null_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        
        return analysis
        
    def create_visualization(self, data: Union[str, pd.DataFrame], chart_type: str, **kwargs) -> str:
        """Create data visualization"""
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data
            
        plt.figure(figsize=kwargs.get('figsize', (10, 6)))
        
        if chart_type == 'line':
            df.plot(kind='line', **kwargs)
        elif chart_type == 'bar':
            df.plot(kind='bar', **kwargs)
        elif chart_type == 'hist':
            df.plot(kind='hist', **kwargs)
        elif chart_type == 'scatter':
            df.plot(kind='scatter', **kwargs)
        elif chart_type == 'box':
            df.plot(kind='box', **kwargs)
        elif chart_type == 'heatmap':
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        else:
            raise ValueError(f"Unsupported chart type: {chart_type}")
            
        output_path = self.plots_dir / f"chart_{int(time.time())}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(output_path)
        
    def generate_report(self, data: Union[str, pd.DataFrame], output_path: str) -> str:
        """Generate data analysis report"""
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data
            
        report = f"""
# Data Analysis Report

## Basic Information
- Shape: {df.shape}
- Columns: {list(df.columns)}
- Memory Usage: {df.memory_usage(deep=True).sum()} bytes

## Statistical Summary
{df.describe().to_string()}

## Missing Values
{df.isnull().sum().to_string()}

## Data Types
{df.dtypes.to_string()}

## Correlation Matrix
{df.corr().to_string()}
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        return output_path

class SecurityUtils:
    """Security utilities"""
    
    @staticmethod
    def generate_password_hash(password: str) -> str:
        """Generate password hash"""
        import bcrypt
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password"""
        import bcrypt
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        
    @staticmethod
    def generate_api_key(length: int = 32) -> str:
        """Generate API key"""
        import secrets
        return secrets.token_urlsafe(length)
        
    @staticmethod
    def encrypt_data(data: str, key: str) -> str:
        """Encrypt data"""
        from cryptography.fernet import Fernet
        f = Fernet(key.encode())
        return f.encrypt(data.encode()).decode()
        
    @staticmethod
    def decrypt_data(encrypted_data: str, key: str) -> str:
        """Decrypt data"""
        from cryptography.fernet import Fernet
        f = Fernet(key.encode())
        return f.decrypt(encrypted_data.encode()).decode()
        
    @staticmethod
    def validate_file_type(file_path: str, allowed_types: List[str]) -> bool:
        """Validate file type"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in allowed_types
        
    @staticmethod
    def scan_file_for_malware(file_path: str) -> Dict[str, Any]:
        """Basic malware scan"""
        # This is a placeholder - in production, use real antivirus APIs
        suspicious_patterns = [
            b'eval(',
            b'exec(',
            b'system(',
            b'shell_exec(',
            b'passthru(',
            b'base64_decode('
        ]
        
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                
            suspicious_found = []
            for pattern in suspicious_patterns:
                if pattern in content:
                    suspicious_found.append(pattern.decode())
                    
            return {
                'file': file_path,
                'suspicious_patterns': suspicious_found,
                'risk_level': 'high' if suspicious_found else 'low',
                'scan_time': datetime.datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'file': file_path,
                'error': str(e),
                'risk_level': 'unknown'
            }

# Utility functions
def format_bytes(bytes_value: int) -> str:
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"

def format_duration(seconds: float) -> str:
    """Format duration to human readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def download_file(url: str, output_path: str, chunk_size: int = 8192) -> str:
    """Download file from URL"""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                
    return output_path

def extract_archive(archive_path: str, extract_to: str) -> List[str]:
    """Extract archive file"""
    archive_path = Path(archive_path)
    extract_to = Path(extract_to)
    extract_to.mkdir(exist_ok=True)
    
    extracted_files = []
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            extracted_files = zip_ref.namelist()
    elif archive_path.suffix in ['.tar', '.gz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
            extracted_files = tar_ref.getnames()
    elif archive_path.suffix == '.rar':
        with rarfile.RarFile(archive_path, 'r') as rar_ref:
            rar_ref.extractall(extract_to)
            extracted_files = rar_ref.namelist()
            
    return [str(extract_to / file) for file in extracted_files]

def create_archive(files: List[str], archive_path: str, format: str = 'zip') -> str:
    """Create archive from files"""
    if format == 'zip':
        with zipfile.ZipFile(archive_path, 'w') as zip_ref:
            for file in files:
                zip_ref.write(file, os.path.basename(file))
    elif format == 'tar.gz':
        with tarfile.open(archive_path, 'w:gz') as tar_ref:
            for file in files:
                tar_ref.add(file, arcname=os.path.basename(file))
                
    return archive_path

# Global instances
file_manager = FileManager()
image_processor = ImageProcessor()
system_monitor = SystemMonitor()
task_scheduler = TaskScheduler()
data_analyzer = DataAnalyzer()
security_utils = SecurityUtils()
