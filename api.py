#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaTniX AI API - REST API endpoints for external integration
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import os
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta
from functools import wraps
import jwt
from werkzeug.utils import secure_filename
import tempfile
import base64
from pathlib import Path

from main import MaTniXAI

app = Flask(__name__)
CORS(app)

# Initialize AI
ai = MaTniXAI()

# Configuration
app.config['SECRET_KEY'] = ai.config.SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = ai.config.MAX_FILE_SIZE

# Active sessions
active_sessions = {}

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token required'}), 401
        
        try:
            if token.startswith('Bearer '):
                token = token[7:]
            
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
            current_user = data['username']
            
            if current_user != ai.config.ADMIN_USERNAME:
                return jsonify({'error': 'Invalid token'}), 401
                
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
            
        return f(current_user, *args, **kwargs)
    return decorated

def create_session(username):
    """Create a new session"""
    session_id = str(uuid.uuid4())
    token = jwt.encode({
        'username': username,
        'session_id': session_id,
        'exp': datetime.utcnow() + timedelta(hours=24)
    }, app.config['SECRET_KEY'])
    
    active_sessions[session_id] = {
        'username': username,
        'created_at': datetime.now(),
        'last_activity': datetime.now()
    }
    
    return token

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Authenticate user"""
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'error': 'Username and password required'}), 400
            
        if ai.authenticate_user(username, password):
            token = create_session(username)
            return jsonify({
                'success': True,
                'token': token,
                'username': username,
                'message': 'Login successful'
            })
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/logout', methods=['POST'])
@token_required
def logout(current_user):
    """Logout user"""
    try:
        token = request.headers.get('Authorization')
        if token.startswith('Bearer '):
            token = token[7:]
            
        data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        session_id = data.get('session_id')
        
        if session_id in active_sessions:
            del active_sessions[session_id]
            
        return jsonify({'success': True, 'message': 'Logout successful'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
@token_required
def chat(current_user):
    """Chat with AI"""
    try:
        data = request.get_json()
        message = data.get('message')
        
        if not message:
            return jsonify({'error': 'Message required'}), 400
            
        # Process message
        result = ai.process_natural_language(message)
        
        if 'error' in result:
            return jsonify({'error': result['error']}), 500
            
        # Save to chat history
        ai.cursor.execute(
            "INSERT INTO chat_history (user_id, message, response) VALUES (?, ?, ?)",
            (1, message, result['response'])
        )
        ai.conn.commit()
        
        return jsonify({
            'response': result['response'],
            'intents': result.get('intents', []),
            'confidence': result.get('confidence', 0),
            'entities': result.get('entities', [])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice/process', methods=['POST'])
@token_required
def process_voice(current_user):
    """Process voice input"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Audio file required'}), 400
            
        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            
            # Process audio
            text = ai.process_voice_command(tmp.name)
            
            # Clean up
            os.unlink(tmp.name)
            
            return jsonify({'text': text})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice/synthesize', methods=['POST'])
@token_required
def synthesize_voice(current_user):
    """Convert text to speech"""
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'Text required'}), 400
            
        # Generate speech
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
            result = ai.text_to_speech(text, tmp.name)
            
            if os.path.exists(tmp.name):
                return send_file(tmp.name, as_attachment=True, download_name='speech.mp3')
            else:
                return jsonify({'error': 'Speech synthesis failed'}), 500
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/upload', methods=['POST'])
@token_required
def upload_file(current_user):
    """Upload and process file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'File required'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        filename = secure_filename(file.filename)
        file_path = Path(ai.sandbox_dir) / filename
        
        # Save file
        file.save(str(file_path))
        
        # Process file
        operation = request.form.get('operation', 'read')
        result = ai.process_file(str(file_path), operation)
        
        return jsonify({
            'filename': filename,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/<filename>')
@token_required
def download_file(current_user, filename):
    """Download file"""
    try:
        file_path = Path(ai.sandbox_dir) / filename
        if file_path.exists():
            return send_file(str(file_path), as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files', methods=['GET'])
@token_required
def list_files(current_user):
    """List files in sandbox"""
    try:
        files = []
        for file_path in ai.sandbox_dir.iterdir():
            if file_path.is_file():
                stat = file_path.stat()
                files.append({
                    'name': file_path.name,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
                
        return jsonify({'files': files})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/command/execute', methods=['POST'])
@token_required
def execute_command(current_user):
    """Execute system command"""
    try:
        data = request.get_json()
        command = data.get('command')
        safe_mode = data.get('safe_mode', True)
        
        if not command:
            return jsonify({'error': 'Command required'}), 400
            
        result = ai.execute_command(command, safe_mode)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/web/search', methods=['POST'])
@token_required
def web_search(current_user):
    """Perform web search"""
    try:
        data = request.get_json()
        query = data.get('query')
        engine = data.get('engine', 'duckduckgo')
        
        if not query:
            return jsonify({'error': 'Query required'}), 400
            
        results = ai.web_search(query, engine)
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/code/generate', methods=['POST'])
@token_required
def generate_code(current_user):
    """Generate code"""
    try:
        data = request.get_json()
        prompt = data.get('prompt')
        language = data.get('language', 'python')
        
        if not prompt:
            return jsonify({'error': 'Prompt required'}), 400
            
        code = ai.generate_code(prompt, language)
        
        return jsonify({'code': code, 'language': language})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/create', methods=['POST'])
@token_required
def create_project(current_user):
    """Create new project"""
    try:
        data = request.get_json()
        project_type = data.get('type')
        project_name = data.get('name')
        
        if not project_type or not project_name:
            return jsonify({'error': 'Project type and name required'}), 400
            
        result = ai.create_project(project_type, project_name)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/save', methods=['POST'])
@token_required
def save_to_memory(current_user):
    """Save data to memory"""
    try:
        data = request.get_json()
        key = data.get('key')
        value = data.get('value')
        file_path = data.get('file_path')
        
        if not key or not value:
            return jsonify({'error': 'Key and value required'}), 400
            
        result = ai.save_to_memory(key, value, 1, file_path)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/retrieve', methods=['GET'])
@token_required
def get_from_memory(current_user):
    """Retrieve data from memory"""
    try:
        key = request.args.get('key')
        results = ai.get_from_memory(key, 1)
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/translate', methods=['POST'])
@token_required
def translate_text(current_user):
    """Translate text"""
    try:
        data = request.get_json()
        text = data.get('text')
        target_lang = data.get('target_lang', 'en')
        
        if not text:
            return jsonify({'error': 'Text required'}), 400
            
        result = ai.translate_text(text, target_lang)
        
        return jsonify({'translated_text': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
@token_required
def summarize_text(current_user):
    """Summarize text"""
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({'error': 'Text required'}), 400
            
        result = ai.summarize_text(text)
        
        return jsonify({'summary': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health/info', methods=['POST'])
@token_required
def get_health_info(current_user):
    """Get health information"""
    try:
        data = request.get_json()
        query = data.get('query')
        
        if not query:
            return jsonify({'error': 'Query required'}), 400
            
        result = ai.get_health_info(query)
        
        return jsonify({'info': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/finance/data', methods=['GET'])
@token_required
def get_finance_data(current_user):
    """Get financial data"""
    try:
        symbol = request.args.get('symbol')
        
        if not symbol:
            return jsonify({'error': 'Symbol required'}), 400
            
        result = ai.get_finance_data(symbol)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/compress', methods=['POST'])
@token_required
def compress_files(current_user):
    """Compress files"""
    try:
        data = request.get_json()
        files = data.get('files')
        output_path = data.get('output_path')
        format = data.get('format', 'zip')
        
        if not files or not output_path:
            return jsonify({'error': 'Files and output path required'}), 400
            
        result = ai.compress_files(files, output_path, format)
        
        return jsonify({'compressed_file': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/python/to-exe', methods=['POST'])
@token_required
def python_to_exe(current_user):
    """Convert Python to executable"""
    try:
        data = request.get_json()
        py_file = data.get('py_file')
        output_dir = data.get('output_dir')
        
        if not py_file:
            return jsonify({'error': 'Python file required'}), 400
            
        result = ai.python_to_exe(py_file, output_dir)
        
        return jsonify({'exe_file': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/features/add', methods=['POST'])
@token_required
def add_feature(current_user):
    """Add new feature"""
    try:
        data = request.get_json()
        feature_description = data.get('description')
        code = data.get('code')
        
        if not feature_description or not code:
            return jsonify({'error': 'Description and code required'}), 400
            
        result = ai.add_new_feature(feature_description, code)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get AI status"""
    return jsonify({
        'status': 'running',
        'version': '1.0.0',
        'features': [
            'Natural Language Processing',
            'Voice Recognition & Synthesis',
            'File Operations',
            'Web Search',
            'Code Generation',
            'Project Creation',
            'Memory Management',
            'Translation',
            'Health Information',
            'Financial Data',
            'Security & Authentication'
        ],
        'endpoints': [
            '/api/auth/login',
            '/api/chat',
            '/api/voice/process',
            '/api/voice/synthesize',
            '/api/files/upload',
            '/api/command/execute',
            '/api/web/search',
            '/api/code/generate',
            '/api/project/create',
            '/api/memory/save',
            '/api/translate',
            '/api/summarize',
            '/api/health/info',
            '/api/finance/data'
        ]
    })

@app.route('/api/docs', methods=['GET'])
def api_docs():
    """API documentation"""
    return jsonify({
        'title': 'MaTniX AI API',
        'version': '1.0.0',
        'description': 'Comprehensive AI assistant API',
        'authentication': {
            'type': 'JWT Bearer Token',
            'login_endpoint': '/api/auth/login',
            'credentials': {
                'username': 'admin',
                'password': '7819'
            }
        },
        'endpoints': {
            'chat': {
                'method': 'POST',
                'path': '/api/chat',
                'description': 'Chat with AI assistant',
                'parameters': {
                    'message': 'string (required)'
                }
            },
            'voice_process': {
                'method': 'POST',
                'path': '/api/voice/process',
                'description': 'Process voice input',
                'parameters': {
                    'audio': 'file (required)'
                }
            },
            'voice_synthesize': {
                'method': 'POST',
                'path': '/api/voice/synthesize',
                'description': 'Convert text to speech',
                'parameters': {
                    'text': 'string (required)'
                }
            },
            'file_upload': {
                'method': 'POST',
                'path': '/api/files/upload',
                'description': 'Upload and process files',
                'parameters': {
                    'file': 'file (required)',
                    'operation': 'string (optional)'
                }
            }
        }
    })

# HTML template for main page
@app.route('/templates/index.html')
def index_template():
    return '''
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MaTniX AI - Süper Asistan</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5em;
        }
        .auth-section {
            margin-bottom: 30px;
        }
        .chat-section {
            display: none;
        }
        input, button, textarea {
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            border: none;
            border-radius: 5px;
            font-size: 16px;
        }
        button {
            background: #4CAF50;
            color: white;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #45a049;
        }
        .chat-messages {
            height: 300px;
            overflow-y: auto;
            background: rgba(0,0,0,0.2);
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 10px;
        }
        .user-message {
            background: rgba(255,255,255,0.2);
            text-align: right;
        }
        .ai-message {
            background: rgba(0,0,0,0.3);
        }
        .features {
            margin-top: 30px;
        }
        .feature-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        .feature-card {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 MaTniX AI</h1>
        
        <div class="auth-section" id="authSection">
            <h2>Giriş Yap</h2>
            <input type="text" id="username" placeholder="Kullanıcı Adı" value="admin">
            <input type="password" id="password" placeholder="Şifre" value="7819">
            <button onclick="login()">Giriş Yap</button>
        </div>
        
        <div class="chat-section" id="chatSection">
            <h2>💬 Sohbet</h2>
            <div class="chat-messages" id="chatMessages"></div>
            <textarea id="messageInput" placeholder="Mesajınızı yazın..." rows="3"></textarea>
            <button onclick="sendMessage()">Gönder</button>
            <button onclick="startVoiceRecording()">🎤 Sesli Mesaj</button>
            <button onclick="logout()">Çıkış Yap</button>
        </div>
        
        <div class="features">
            <h2>✨ Özellikler</h2>
            <div class="feature-grid">
                <div class="feature-card">🧠 Doğal Dil İşleme</div>
                <div class="feature-card">🎤 Sesli Komutlar</div>
                <div class="feature-card">📁 Dosya Yönetimi</div>
                <div class="feature-card">🌐 Web Arama</div>
                <div class="feature-card">💻 Kod Üretimi</div>
                <div class="feature-card">🏗️ Proje Oluşturma</div>
                <div class="feature-card">🧠 Bellek Yönetimi</div>
                <div class="feature-card">🌍 Çeviri</div>
                <div class="feature-card">🏥 Sağlık Bilgisi</div>
                <div class="feature-card">💰 Finans Veri</div>
                <div class="feature-card">🔒 Güvenlik</div>
                <div class="feature-card">⚡ Hızlı İşlem</div>
            </div>
        </div>
    </div>

    <script>
        let authToken = '';
        
        async function login() {
            const username = document.getElementById('username').value;
            const password = document.getElementById('password').value;
            
            try {
                const response = await fetch('/api/auth/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ username, password })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    authToken = data.token;
                    document.getElementById('authSection').style.display = 'none';
                    document.getElementById('chatSection').style.display = 'block';
                    addMessage('AI', '🚀 MaTniX AI hazır! Size nasıl yardımcı olabilirim?');
                } else {
                    alert('Giriş başarısız: ' + data.error);
                }
            } catch (error) {
                alert('Hata: ' + error.message);
            }
        }
        
        async function sendMessage() {
            const message = document.getElementById('messageInput').value;
            if (!message.trim()) return;
            
            addMessage('User', message);
            document.getElementById('messageInput').value = '';
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${authToken}`
                    },
                    body: JSON.stringify({ message })
                });
                
                const data = await response.json();
                addMessage('AI', data.response);
            } catch (error) {
                addMessage('AI', 'Hata: ' + error.message);
            }
        }
        
        function addMessage(sender, message) {
            const chatMessages = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender.toLowerCase()}-message`;
            messageDiv.innerHTML = `<strong>${sender}:</strong> ${message}`;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
        
        async function startVoiceRecording() {
            addMessage('AI', '🎤 Ses kaydı başlatıldı...');
            // Voice recording implementation would go here
            addMessage('AI', 'Sesli komut özelliği tarayıcı desteği gerektirir.');
        }
        
        async function logout() {
            try {
                await fetch('/api/auth/logout', {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${authToken}`
                    }
                });
            } catch (error) {
                console.error('Logout error:', error);
            }
            
            authToken = '';
            document.getElementById('authSection').style.display = 'block';
            document.getElementById('chatSection').style.display = 'none';
        }
        
        // Enter key to send message
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    # Create templates directory
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Save index.html
    with open(templates_dir / 'index.html', 'w', encoding='utf-8') as f:
        f.write(index_template())
    
    # Get port from environment variable (Render.com requirement)
    port = int(os.environ.get('PORT', 5000))
    
    print("🚀 MaTniX AI API Başlatılıyor...")
    print(f"📊 Port: {port}")
    print("📊 API Dokümantasyonu: /api/docs")
    print("🌐 Web Arayüzü: /")
    print("🔑 Giriş Bilgileri: admin / 7819")
    
    app.run(host='0.0.0.0', port=port, debug=False)

