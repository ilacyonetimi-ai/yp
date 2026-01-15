#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaTniX AI - Süper Asistan Yapay Zekası
Tam kapsamlı Türkçe yapay zeka asistanı
"""

import os
import sys
import json
import logging
import asyncio
import threading
import subprocess
import tempfile
import shutil
import zipfile
import tarfile
import rarfile
import time
import datetime
import hashlib
import base64
import mimetypes
import re
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import pickle

# Core AI and ML libraries
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import speech_recognition as sr
import pyttsx3
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pdfplumber
import docx
import openpyxl
from pydub import AudioSegment
import whisper

# Web and API libraries
import requests
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import bs4
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import feedparser
import yfinance as yf

# Security and authentication
import bcrypt
import jwt
from cryptography.fernet import Fernet
import hashlib

# File processing and utilities
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename

# Google Drive integration
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

# Configuration
@dataclass
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'matnix-super-ai-secret-key-2024')
    ADMIN_USERNAME = 'admin'
    ADMIN_PASSWORD = '7819'
    API_KEY = os.environ.get('API_KEY', 'matnix-api-key-2024')
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    ALLOWED_EXTENSIONS = {
        'txt', 'py', 'js', 'json', 'csv', 'md', 'pdf', 'docx', 'xlsx',
        'png', 'jpg', 'jpeg', 'svg', 'mp3', 'wav', 'mp4', 'zip', 'rar', 'tar.gz', '7z'
    }
    MODEL_NAME = 'dbmdz/bert-base-turkish-cased'
    WHISPER_MODEL = 'base'
    TTS_VOICE_ID = 'tr-TR'
    GOOGLE_DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']
    SANDBOX_DIR = '/tmp/matnix_sandbox'
    LOG_LEVEL = logging.INFO
    MAX_CONCURRENT_TASKS = 10

class MaTniXAI:
    def __init__(self):
        self.config = Config()
        self.setup_logging()
        self.setup_directories()
        self.setup_database()
        self.load_models()
        self.setup_security()
        self.chat_history = []
        self.user_memory = {}
        self.running_tasks = {}
        self.executor = ThreadPoolExecutor(max_workers=self.config.MAX_CONCURRENT_TASKS)
        
    def setup_logging(self):
        logging.basicConfig(
            level=self.config.LOG_LEVEL,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('matnix.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('MaTniX-AI')
        
    def setup_directories(self):
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.sandbox_dir = Path(self.config.SANDBOX_DIR)
        self.sandbox_dir.mkdir(exist_ok=True)
        self.memory_dir = self.base_dir / 'memory'
        self.memory_dir.mkdir(exist_ok=True)
        self.projects_dir = self.base_dir / 'projects'
        self.projects_dir.mkdir(exist_ok=True)
        
    def setup_database(self):
        self.db_path = self.base_dir / 'matnix.db'
        self.conn = sqlite3.connect(str(self.db_path))
        self.cursor = self.conn.cursor()
        self.create_tables()
        
    def create_tables(self):
        tables = [
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                message TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        ]
        
        for table in tables:
            self.cursor.execute(table)
        self.conn.commit()
        
    def load_models(self):
        try:
            self.logger.info("Loading Turkish language model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.MODEL_NAME)
            self.text_model = AutoModelForCausalLM.from_pretrained(self.config.MODEL_NAME)
            self.nlp_pipeline = pipeline('text-generation', model=self.text_model, tokenizer=self.tokenizer)
            
            self.logger.info("Loading Whisper model for speech recognition...")
            self.whisper_model = whisper.load_model(self.config.WHISPER_MODEL)
            
            self.logger.info("Initializing text-to-speech...")
            self.tts_engine = pyttsx3.init()
            voices = self.tts_engine.getProperty('voices')
            for voice in voices:
                if 'turkish' in voice.languages[0].lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
                    
            self.logger.info("Loading computer vision models...")
            self.ocr_reader = None  # Will be loaded on demand
            
            self.logger.info("All models loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
            
    def setup_security(self):
        self.cipher_suite = Fernet(Fernet.generate_key())
        self.create_admin_user()
        
    def create_admin_user(self):
        try:
            password_hash = bcrypt.hashpw(self.config.ADMIN_PASSWORD.encode('utf-8'), bcrypt.gensalt())
            self.cursor.execute(
                "INSERT OR IGNORE INTO users (username, password_hash) VALUES (?, ?)",
                (self.config.ADMIN_USERNAME, password_hash.decode('utf-8'))
            )
            self.conn.commit()
        except Exception as e:
            self.logger.error(f"Error creating admin user: {str(e)}")
            
    def authenticate_user(self, username: str, password: str) -> bool:
        try:
            self.cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
            result = self.cursor.fetchone()
            if result:
                stored_hash = result[0].encode('utf-8')
                return bcrypt.checkpw(password.encode('utf-8'), stored_hash)
            return False
        except Exception as e:
            self.logger.error(f"Authentication error: {str(e)}")
            return False
            
    def process_natural_language(self, text: str) -> Dict[str, Any]:
        """Process natural language input and determine intent"""
        try:
            # Simple intent recognition
            text_lower = text.lower()
            
            intent_patterns = {
                'file_operation': ['dosya', 'dosyayı', 'dosyaları', 'aç', 'kapat', 'sil', 'düzenle', 'kaydet'],
                'web_search': ['ara', 'bul', 'internet', 'web', 'google', 'arama'],
                'code_generation': ['kod', 'python', 'javascript', 'yaz', 'oluştur', 'geliştir'],
                'voice_command': ['konuş', 'ses', 'dinle', 'söyle', 'anlat'],
                'memory_operation': ['bellek', 'hatırla', 'kaydet', 'unutma'],
                'system_command': ['sistem', 'komut', 'çalıştır', 'terminal'],
                'project_creation': ['proje', 'uygulama', 'web', 'mobil', 'masaüstü'],
                'multimedia': ['resim', 'görüntü', 'video', 'müzik', 'ses'],
                'translation': ['çevir', 'tercüme', 'ingilizce', 'türkçe'],
                'summary': ['özet', 'topla', 'kısaca'],
                'calendar': ['takvim', 'ajanda', 'randevu', 'hatırlatıcı'],
                'health': ['sağlık', 'hastalık', 'ilaç', 'doktor'],
                'finance': ['borsa', 'kripto', 'fiyat', 'para', 'yatırım']
            }
            
            detected_intents = []
            for intent, patterns in intent_patterns.items():
                if any(pattern in text_lower for pattern in patterns):
                    detected_intents.append(intent)
                    
            # Generate response using the language model
            response = self.generate_response(text)
            
            return {
                'intents': detected_intents,
                'response': response,
                'confidence': len(detected_intents) / len(intent_patterns),
                'entities': self.extract_entities(text)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing natural language: {str(e)}")
            return {'error': str(e)}
            
    def generate_response(self, prompt: str) -> str:
        """Generate response using the loaded language model"""
        try:
            # Add context from chat history
            context = "\n".join([f"User: {h['user']}\nAI: {h['ai']}" for h in self.chat_history[-5:]])
            full_prompt = f"{context}\nUser: {prompt}\nAI:"
            
            # Generate response
            responses = self.nlp_pipeline(
                full_prompt,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = responses[0]['generated_text'].split('AI:')[-1].strip()
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."
            
    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract entities from text"""
        entities = []
        
        # Simple entity extraction patterns
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+90|0)?\s*\d{3}\s*\d{3}\s*\d{2}\s*\d{2}\b',
            'url': r'https?://[^\s]+',
            'date': r'\d{1,2}[./-]\d{1,2}[./-]\d{2,4}',
            'time': r'\d{1,2}[:]\d{2}'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for match in matches:
                entities.append({'type': entity_type, 'value': match})
                
        return entities
        
    def process_voice_command(self, audio_file_path: str = None) -> str:
        """Process voice command using speech recognition"""
        try:
            if audio_file_path and os.path.exists(audio_file_path):
                # Use Whisper for file-based recognition
                result = self.whisper_model.transcribe(audio_file_path)
                return result['text']
            else:
                # Use microphone for real-time recognition
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    print("Dinliyorum...")
                    recognizer.adjust_for_ambient_noise(source)
                    audio = recognizer.listen(source)
                    
                try:
                    text = recognizer.recognize_google(audio, language='tr-TR')
                    return text
                except sr.UnknownValueError:
                    return "Ses anlaşılamadı"
                except sr.RequestError:
                    return "Ses tanıma servisi çalışmıyor"
                    
        except Exception as e:
            self.logger.error(f"Error processing voice command: {str(e)}")
            return f"Hata: {str(e)}"
            
    def text_to_speech(self, text: str, output_file: str = None) -> str:
        """Convert text to speech"""
        try:
            if output_file:
                self.tts_engine.save_to_file(text, output_file)
                self.tts_engine.runAndWait()
                return output_file
            else:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return "Ses çalındı"
        except Exception as e:
            self.logger.error(f"Error in text-to-speech: {str(e)}")
            return f"Hata: {str(e)}"
            
    def process_file(self, file_path: str, operation: str = 'read') -> Dict[str, Any]:
        """Process various file types"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {'error': 'Dosya bulunamadı'}
                
            file_ext = file_path.suffix.lower()
            
            if operation == 'read':
                if file_ext == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    return {'content': content, 'type': 'text'}
                    
                elif file_ext == '.pdf':
                    with pdfplumber.open(file_path) as pdf:
                        content = ""
                        for page in pdf.pages:
                            content += page.extract_text() or ""
                    return {'content': content, 'type': 'pdf'}
                    
                elif file_ext == '.docx':
                    doc = docx.Document(file_path)
                    content = "\n".join([para.text for para in doc.paragraphs])
                    return {'content': content, 'type': 'docx'}
                    
                elif file_ext == '.xlsx':
                    df = pd.read_excel(file_path)
                    content = df.to_string()
                    return {'content': content, 'type': 'excel'}
                    
                elif file_ext in ['.png', '.jpg', '.jpeg']:
                    image = Image.open(file_path)
                    # OCR processing would go here
                    return {'content': f'Görüntü boyutu: {image.size}', 'type': 'image'}
                    
                elif file_ext in ['.mp3', '.wav']:
                    audio = AudioSegment.from_file(file_path)
                    return {'content': f'Ses süresi: {len(audio)/1000} saniye', 'type': 'audio'}
                    
                else:
                    return {'error': 'Desteklenmeyen dosya formatı'}
                    
            elif operation == 'info':
                stat = file_path.stat()
                return {
                    'size': stat.st_size,
                    'modified': datetime.datetime.fromtimestamp(stat.st_mtime),
                    'type': file_ext,
                    'name': file_path.name
                }
                
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}")
            return {'error': str(e)}
            
    def execute_command(self, command: str, safe_mode: bool = True) -> Dict[str, Any]:
        """Execute system commands with safety checks"""
        try:
            dangerous_commands = [
                'rm -rf', 'del /f /s /q', 'format', 'fdisk', 'mkfs',
                'shutdown', 'reboot', 'halt', 'poweroff',
                'chmod 777', 'chown', 'passwd', 'su', 'sudo'
            ]
            
            if safe_mode and any(dangerous in command.lower() for dangerous in dangerous_commands):
                return {
                    'error': 'Güvenlik uyarısı: Tehlikeli komut tespit edildi',
                    'command': command,
                    'safe_mode': True
                }
                
            # Execute in sandbox directory
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.sandbox_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                'command': command,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'success': result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {'error': 'Komut zaman aşımına uğradı'}
        except Exception as e:
            self.logger.error(f"Error executing command: {str(e)}")
            return {'error': str(e)}
            
    def web_search(self, query: str, engine: str = 'duckduckgo') -> List[Dict[str, str]]:
        """Perform web search"""
        try:
            if engine == 'duckduckgo':
                url = f"https://duckduckgo.com/html/?q={query}"
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                soup = bs4.BeautifulSoup(response.text, 'html.parser')
                
                results = []
                for result in soup.find_all('a', class_='result__a')[:10]:
                    results.append({
                        'title': result.get_text(),
                        'url': result.get('href')
                    })
                return results
                
        except Exception as e:
            self.logger.error(f"Error in web search: {str(e)}")
            return []
            
    def create_project(self, project_type: str, project_name: str) -> Dict[str, Any]:
        """Create various types of projects"""
        try:
            project_path = self.projects_dir / project_name
            project_path.mkdir(exist_ok=True)
            
            if project_type == 'flask':
                self.create_flask_project(project_path)
            elif project_type == 'react':
                self.create_react_project(project_path)
            elif project_type == 'python':
                self.create_python_project(project_path)
            elif project_type == 'nextjs':
                self.create_nextjs_project(project_path)
            else:
                return {'error': 'Desteklenmeyen proje tipi'}
                
            return {
                'success': True,
                'project_path': str(project_path),
                'project_type': project_type,
                'instructions': f'Proje oluşturuldu: {project_path}'
            }
            
        except Exception as e:
            self.logger.error(f"Error creating project: {str(e)}")
            return {'error': str(e)}
            
    def create_flask_project(self, path: Path):
        """Create Flask project structure"""
        (path / 'app.py').write_text('''
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
''')
        
        (path / 'templates').mkdir()
        (path / 'templates' / 'index.html').write_text('''
<!DOCTYPE html>
<html>
<head>
    <title>Flask App</title>
</head>
<body>
    <h1>Merhaba Dünya!</h1>
</body>
</html>
''')
        
        (path / 'requirements.txt').write_text('flask\n')
        (path / 'README.md').write_text(f'# {path.name}\n\nFlask projesi.')
        
    def create_react_project(self, path: Path):
        """Create React project structure"""
        (path / 'package.json').write_text(f'''
{{
  "name": "{path.name}",
  "version": "0.1.0",
  "private": true,
  "dependencies": {{
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
  }},
  "scripts": {{
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  }}
}}
''')
        
        (path / 'src').mkdir()
        (path / 'public').mkdir()
        
        (path / 'src' / 'App.js').write_text('''
import React from 'react';

function App() {
  return (
    <div className="App">
      <h1>Merhaba Dünya!</h1>
    </div>
  );
}

export default App;
''')
        
        (path / 'src' / 'index.js').write_text('''
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
''')
        
    def create_python_project(self, path: Path):
        """Create Python project structure"""
        (path / 'main.py').write_text('''
def main():
    print("Merhaba Dünya!")

if __name__ == "__main__":
    main()
''')
        
        (path / 'requirements.txt').write_text('')
        (path / 'README.md').write_text(f'# {path.name}\n\nPython projesi.')
        
    def create_nextjs_project(self, path: Path):
        """Create Next.js project structure"""
        (path / 'package.json').write_text(f'''
{{
  "name": "{path.name}",
  "version": "0.1.0",
  "private": true,
  "scripts": {{
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  }},
  "dependencies": {{
    "next": "13.0.0",
    "react": "18.2.0",
    "react-dom": "18.2.0"
  }}
}}
''')
        
        (path / 'pages').mkdir()
        (path / 'pages' / 'index.js').write_text('''
export default function Home() {
  return (
    <div>
      <h1>Merhaba Dünya!</h1>
    </div>
  );
}
''')
        
    def save_to_memory(self, key: str, value: str, user_id: int = 1, file_path: str = None):
        """Save data to memory"""
        try:
            self.cursor.execute(
                "INSERT INTO memory (user_id, key, value, file_path) VALUES (?, ?, ?, ?)",
                (user_id, key, value, file_path)
            )
            self.conn.commit()
            return {'success': True, 'message': 'Belleğe kaydedildi'}
        except Exception as e:
            self.logger.error(f"Error saving to memory: {str(e)}")
            return {'error': str(e)}
            
    def get_from_memory(self, key: str = None, user_id: int = 1) -> List[Dict]:
        """Retrieve data from memory"""
        try:
            if key:
                self.cursor.execute(
                    "SELECT * FROM memory WHERE user_id = ? AND key = ?",
                    (user_id, key)
                )
            else:
                self.cursor.execute(
                    "SELECT * FROM memory WHERE user_id = ?",
                    (user_id,)
                )
            
            columns = [desc[0] for desc in self.cursor.description]
            results = []
            for row in self.cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
        except Exception as e:
            self.logger.error(f"Error retrieving from memory: {str(e)}")
            return []
            
    def generate_code(self, prompt: str, language: str = 'python') -> str:
        """Generate code based on prompt"""
        try:
            code_prompt = f"Write {language} code for: {prompt}"
            response = self.generate_response(code_prompt)
            
            # Extract code block from response
            code_match = re.search(r'```(?:python|javascript|java|cpp|html|css)?\n(.*?)\n```', response, re.DOTALL)
            if code_match:
                return code_match.group(1)
            else:
                return response
                
        except Exception as e:
            self.logger.error(f"Error generating code: {str(e)}")
            return f"// Hata: {str(e)}"
            
    def translate_text(self, text: str, target_lang: str = 'en') -> str:
        """Translate text"""
        try:
            # Simple translation using the language model
            if target_lang == 'en':
                prompt = f"Translate to English: {text}"
            elif target_lang == 'tr':
                prompt = f"Translate to Turkish: {text}"
            else:
                prompt = f"Translate to {target_lang}: {text}"
                
            return self.generate_response(prompt)
            
        except Exception as e:
            self.logger.error(f"Error translating text: {str(e)}")
            return f"Çeviri hatası: {str(e)}"
            
    def summarize_text(self, text: str) -> str:
        """Summarize text"""
        try:
            prompt = f"Summarize this text: {text}"
            return self.generate_response(prompt)
        except Exception as e:
            self.logger.error(f"Error summarizing text: {str(e)}")
            return f"Özetleme hatası: {str(e)}"
            
    def get_health_info(self, query: str) -> str:
        """Get health information"""
        try:
            prompt = f"Provide health information about: {query}. Always include a disclaimer that this is not medical advice."
            return self.generate_response(prompt)
        except Exception as e:
            self.logger.error(f"Error getting health info: {str(e)}")
            return f"Sağlık bilgisi hatası: {str(e)}"
            
    def get_finance_data(self, symbol: str) -> Dict[str, Any]:
        """Get financial data"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'symbol': symbol,
                'name': info.get('shortName', ''),
                'price': info.get('regularMarketPrice', 0),
                'change': info.get('regularMarketChange', 0),
                'change_percent': info.get('regularMarketChangePercent', 0)
            }
        except Exception as e:
            self.logger.error(f"Error getting finance data: {str(e)}")
            return {'error': str(e)}
            
    def compress_files(self, files: List[str], output_path: str, format: str = 'zip') -> str:
        """Compress files"""
        try:
            if format == 'zip':
                with zipfile.ZipFile(output_path, 'w') as zipf:
                    for file in files:
                        zipf.write(file, os.path.basename(file))
            elif format == 'tar.gz':
                with tarfile.open(output_path, 'w:gz') as tarf:
                    for file in files:
                        tarf.add(file, arcname=os.path.basename(file))
                        
            return output_path
        except Exception as e:
            self.logger.error(f"Error compressing files: {str(e)}")
            return f"Sıkıştırma hatası: {str(e)}"
            
    def python_to_exe(self, py_file: str, output_dir: str = None) -> str:
        """Convert Python file to executable"""
        try:
            if not output_dir:
                output_dir = os.path.dirname(py_file)
                
            cmd = f"pyinstaller --onefile --distpath {output_dir} {py_file}"
            result = self.execute_command(cmd, safe_mode=False)
            
            if result['success']:
                exe_name = os.path.splitext(os.path.basename(py_file))[0] + '.exe'
                exe_path = os.path.join(output_dir, 'dist', exe_name)
                return exe_path
            else:
                return f"Dönüştürme hatası: {result.get('stderr', '')}"
                
        except Exception as e:
            self.logger.error(f"Error converting to exe: {str(e)}")
            return f"Dönüştürme hatası: {str(e)}"
            
    def add_new_feature(self, feature_description: str, code: str) -> Dict[str, Any]:
        """Add new feature to AI"""
        try:
            feature_name = f"feature_{int(time.time())}"
            feature_file = self.base_dir / f"{feature_name}.py"
            
            feature_file.write_text(f'''
# Auto-generated feature: {feature_description}
{code}

def execute_feature(*args, **kwargs):
    """Execute the new feature"""
    pass
''')
            
            return {
                'success': True,
                'feature_name': feature_name,
                'file_path': str(feature_file),
                'message': 'Yeni özellik eklendi'
            }
            
        except Exception as e:
            self.logger.error(f"Error adding new feature: {str(e)}")
            return {'error': str(e)}
            
    def run(self):
        """Main run loop"""
        print("🤖 MaTniX AI - Süper Asistan Hazır!")
        print("💬 Komutlarınızı bekliyorum...")
        
        while True:
            try:
                user_input = input("\nSiz: ").strip()
                
                if user_input.lower() in ['çık', 'exit', 'quit']:
                    print("👋 Görüşürüz!")
                    break
                    
                if user_input.lower() == 'başlat':
                    print("🚀 MaTniX AI zaten çalışıyor!")
                    continue
                    
                # Process the input
                result = self.process_natural_language(user_input)
                
                if 'error' in result:
                    print(f"❌ Hata: {result['error']}")
                    continue
                    
                print(f"\n🤖 MaTniX: {result['response']}")
                
                # Save to chat history
                self.chat_history.append({
                    'user': user_input,
                    'ai': result['response'],
                    'timestamp': datetime.datetime.now()
                })
                
                # Execute based on detected intents
                if 'voice_command' in result['intents']:
                    voice_result = self.process_voice_command()
                    print(f"🎤 Ses komutu: {voice_result}")
                    
                elif 'file_operation' in result['intents']:
                    # Handle file operations
                    pass
                    
                elif 'web_search' in result['intents']:
                    # Extract search query and perform search
                    search_results = self.web_search(user_input)
                    for result in search_results[:3]:
                        print(f"🔍 {result['title']}")
                        print(f"   {result['url']}")
                        
            except KeyboardInterrupt:
                print("\n👋 Görüşürüz!")
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {str(e)}")
                print(f"❌ Bir hata oluştu: {str(e)}")

if __name__ == "__main__":
    ai = MaTniXAI()
    ai.run()
