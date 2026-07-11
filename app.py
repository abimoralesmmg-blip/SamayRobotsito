import os
import cv2
import numpy as np
import json
import time
import re
import base64
import io
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from ultralytics import YOLO
from PIL import Image
from collections import defaultdict

# ============================================================
# 1. CONFIGURACIÓN E INICIALIZACIÓN
# ============================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'samaydent-secret-key-2026'

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# ---------- CONFIGURACIÓN DE BASE DE DATOS ----------
# Si existe la variable de entorno DATABASE_URL (en Railway), usamos PostgreSQL
# Si no, usamos SQLite local (para desarrollo)
database_url = os.environ.get('DATABASE_URL')
if database_url:
    # Railway proporciona DATABASE_URL con 'postgres://', pero SQLAlchemy requiere 'postgresql://'
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url.replace("postgres://", "postgresql://", 1)
else:
    # Modo local: creamos la carpeta database si no existe
    os.makedirs(os.path.join(BASE_DIR, 'database'), exist_ok=True)
    DB_PATH = os.path.join(BASE_DIR, "database", "samaydent.db")
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# -----------------------------------------------------

CORS(app)
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ============================================================
# 2. MAPEO FDI Y NOMBRES DE DIENTES
# ============================================================
MAPEO_FDI_POR_ID = {
    0: "11", 1: "12", 2: "13", 3: "14", 4: "15", 5: "16", 6: "17", 7: "18",
    8: "21", 9: "22", 10: "23", 11: "24", 12: "25", 13: "26", 14: "27", 15: "28",
    16: "31", 17: "32", 18: "33", 19: "34", 20: "35", 21: "36", 22: "37", 23: "38",
    24: "41", 25: "42", 26: "43", 27: "44", 28: "45", 29: "46", 30: "47", 31: "48"
}
INV_MAPEO_FDI = {v: k for k, v in MAPEO_FDI_POR_ID.items()}

NOMBRES_DIENTES = {
    1: "Incisivo central",
    2: "Incisivo lateral",
    3: "Canino",
    4: "Primer premolar",
    5: "Segundo premolar",
    6: "Primer molar",
    7: "Segundo molar",
    8: "Tercer molar"
}

# ============================================================
# 3. CARGA DE MODELOS
# ============================================================
MODELS = {}
def load_models():
    global MODELS
    model_files = {'nomenclatura': 'modelo_nomenclatura_master.pt', 'patologias': 'modelo_patologias_master.pt'}
    for name, filename in model_files.items():
        path = os.path.join(BASE_DIR, 'models', filename)
        if os.path.exists(path):
            MODELS[name] = YOLO(path)
            print(f"✅ Modelo '{name}' cargado desde {path}")
        else:
            print(f"⚠️ Modelo '{name}' NO encontrado en {path}")

# ============================================================
# 4. BASE DE DATOS Y USUARIOS
# ============================================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class DiagnosticHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    model_used = db.Column(db.String(50))
    result_data = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ============================================================
# 5. MIGRACIÓN DE COLUMNA 'name'
# ============================================================
def migrar_base_datos():
    # Solo ejecutar en modo SQLite (local)
    if not os.environ.get('DATABASE_URL'):
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(user)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'name' not in columns:
            cursor.execute("ALTER TABLE user ADD COLUMN name VARCHAR(100) DEFAULT ''")
            conn.commit()
            print("✅ Columna 'name' añadida a la tabla user.")
        conn.close()
# ============================================================
# 6. CORRECCIÓN DE CUADRANTES (solo primer dígito)
# ============================================================
def corregir_cuadrantes_por_posicion(dientes, img_width, img_height):
    if not dientes:
        return dientes

    centro_x = img_width / 2
    centro_y = img_height / 2

    for d in dientes:
        codigo_actual = d['codigo']
        if len(codigo_actual) != 2:
            continue
        try:
            numero_diente = int(codigo_actual[1])
            cuadrante_actual = int(codigo_actual[0])
        except ValueError:
            continue
        if numero_diente < 1 or numero_diente > 8:
            continue

        box = d['box']
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2

        if cy < centro_y:
            if cx < centro_x:
                cuadrante_real = 1
            else:
                cuadrante_real = 2
        else:
            if cx < centro_x:
                cuadrante_real = 4
            else:
                cuadrante_real = 3

        if cuadrante_real != cuadrante_actual:
            codigo_corregido = f"{cuadrante_real}{numero_diente}"
            if codigo_corregido in INV_MAPEO_FDI:
                print(f"🔧 Corrigiendo cuadrante: {codigo_actual} -> {codigo_corregido}")
                d['codigo'] = codigo_corregido
                d['clase_id'] = INV_MAPEO_FDI[codigo_corregido]

    return dientes

# ============================================================
# 7. ASIGNAR NOMBRE AL DIENTE (basado en el número, sin modificar)
# ============================================================
def asignar_nombres_dientes(dientes):
    for d in dientes:
        codigo = d['codigo']
        if len(codigo) == 2:
            try:
                num = int(codigo[1])
                d['nombre'] = NOMBRES_DIENTES.get(num, f"Diente {num}")
            except ValueError:
                d['nombre'] = "Diente"
        else:
            d['nombre'] = "Diente"
    return dientes

# ============================================================
# 8. PROCESAMIENTO DE IMÁGENES
# ============================================================
def extraer_codigo_fdi(clase_id):
    return MAPEO_FDI_POR_ID.get(clase_id, str(clase_id))

def procesar_unificado(image_data):
    try:
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_array = np.array(img_pil)
        h, w = img_array.shape[:2]

        # 1. DETECCIÓN DE DIENTES
        dientes = []
        model_d = MODELS.get('nomenclatura')
        if model_d:
            res = model_d.predict(img_array, conf=0.35, verbose=False)[0]
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                clase_id = int(box.cls[0])
                dientes.append({
                    'codigo': extraer_codigo_fdi(clase_id),
                    'box': [x1, y1, x2, y2],
                    'clase_id': clase_id
                })

        # 2. CORREGIR CUADRANTES (solo primer dígito)
        dientes = corregir_cuadrantes_por_posicion(dientes, w, h)

        # 3. ASIGNAR NOMBRES (basados en el número del diente, que no se modifica)
        dientes = asignar_nombres_dientes(dientes)

        # 4. DETECCIÓN DE PATOLOGÍAS
        patologias = []
        model_p = MODELS.get('patologias')
        if model_p:
            res_p = model_p.predict(img_array, conf=0.20, iou=0.4, augment=True, verbose=False)[0]
            for box in res_p.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                clase_id = int(box.cls[0])
                nombre = model_p.names[clase_id]
                if not nombre or nombre.strip() == '':
                    nombre = 'Patología'
                fdi = None
                cx, cy = (x1+x2)/2, (y1+y2)/2
                for d in dientes:
                    if d['box'][0] <= cx <= d['box'][2] and d['box'][1] <= cy <= d['box'][3]:
                        fdi = d['codigo']
                        break
                patologias.append({
                    'nombre_traducido': nombre,
                    'box': [x1, y1, x2, y2],
                    'codigo_fdi': fdi,
                    'clase_id': clase_id
                })

        # 5. CONSTRUIR HALLAZGOS
        hallazgos = []
        por_fdi = defaultdict(list)
        nombre_por_fdi = {d['codigo']: d.get('nombre', 'Diente') for d in dientes}

        for p in patologias:
            if p['codigo_fdi']:
                por_fdi[p['codigo_fdi']].append(p['nombre_traducido'])
            else:
                hallazgos.append({'tipo': 'global', 'texto': p['nombre_traducido'], 'patologias': [p['nombre_traducido']]})

        for fdi, pats in por_fdi.items():
            unique_pats = []
            for p in pats:
                if p not in unique_pats:
                    unique_pats.append(p)
            nombre_diente = nombre_por_fdi.get(fdi, 'Diente')
            hallazgos.append({
                'tipo': 'diente',
                'fdi': fdi,
                'nombre': nombre_diente,
                'patologias': unique_pats,
                'texto': f"Pieza {fdi} ({nombre_diente}): {', '.join(unique_pats)}"
            })

        return {
            'imagen_original': base64.b64encode(img_bytes).decode('utf-8'),
            'dientes': dientes,
            'patologias': patologias,
            'hallazgos': hallazgos
        }
    except Exception as e:
        print(f"❌ Error en procesar_unificado: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================================
# 9. RUTAS
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
@login_required
def predict():
    data = request.json
    image_data = data.get('image')
    model_type = data.get('model_type', 'nomenclatura')
    if not image_data:
        return jsonify({'success': False, 'error': 'No se recibió imagen'}), 400

    res = procesar_unificado(image_data)

    if not res:
        return jsonify({'success': False, 'error': 'Error procesando la imagen'}), 500

    historial = DiagnosticHistory(
        user_id=current_user.id,
        model_used=model_type,
        result_data=json.dumps(res)
    )
    db.session.add(historial)
    db.session.commit()

    res['success'] = True
    return jsonify(res)

@app.route('/api/register', methods=['POST'])
def register():
    data = request.json
    name = data.get('name', '').strip()
    email = data.get('email', '').strip()
    password = data.get('password', '')
    confirm = data.get('confirm', '')

    if not name or not email or not password:
        return jsonify({'success': False, 'error': 'Todos los campos son obligatorios'}), 400

    if len(password) < 6:
        return jsonify({'success': False, 'error': 'La contraseña debe tener al menos 6 caracteres'}), 400

    if password != confirm:
        return jsonify({'success': False, 'error': 'Las contraseñas no coinciden'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'success': False, 'error': 'El correo ya está registrado'}), 400

    user = User(name=name, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({'success': True, 'message': 'Usuario registrado correctamente'})

@app.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    user = User.query.filter_by(email=email).first()
    if user and user.check_password(password):
        login_user(user)
        return jsonify({
            'success': True,
            'user': {
                'id': user.id,
                'email': user.email,
                'name': user.name
            }
        })

    return jsonify({'success': False, 'error': 'Credenciales incorrectas'}), 401

@app.route('/api/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'success': True})

@app.route('/api/current-user', methods=['GET'])
def current_user_api():
    if current_user.is_authenticated:
        return jsonify({
            'success': True,
            'user': {
                'id': current_user.id,
                'email': current_user.email,
                'name': current_user.name
            }
        })
    return jsonify({'success': False}), 401

@app.route('/api/change-password', methods=['POST'])
@login_required
def change_password():
    data = request.json
    current = data.get('current', '')
    new_pass = data.get('new', '')
    confirm = data.get('confirm', '')

    if not current or not new_pass or not confirm:
        return jsonify({'success': False, 'error': 'Todos los campos son obligatorios'}), 400

    if len(new_pass) < 6:
        return jsonify({'success': False, 'error': 'La nueva contraseña debe tener al menos 6 caracteres'}), 400

    if new_pass != confirm:
        return jsonify({'success': False, 'error': 'Las contraseñas no coinciden'}), 400

    user = current_user
    if not user.check_password(current):
        return jsonify({'success': False, 'error': 'Contraseña actual incorrecta'}), 401

    user.set_password(new_pass)
    db.session.commit()
    return jsonify({'success': True, 'message': 'Contraseña actualizada correctamente'})

@app.route('/api/history', methods=['GET'])
@login_required
def get_history():
    historial = DiagnosticHistory.query.filter_by(user_id=current_user.id).order_by(DiagnosticHistory.created_at.desc()).limit(50).all()
    return jsonify({
        'success': True,
        'history': [{
            'id': h.id,
            'model_used': h.model_used,
            'result_data': json.loads(h.result_data) if h.result_data else {},
            'created_at': h.created_at.isoformat()
        } for h in historial]
    })

# ============================================================
# 10. INICIALIZACIÓN
# ============================================================
# ============================================================
# 10. INICIALIZACIÓN GLOBAL (Para producción y local)
# ============================================================
# Al estar fuera del 'if __name__', Gunicorn sí ejecutará esto al arrancar
with app.app_context():
    db.create_all()
    migrar_base_datos()

load_models()

if __name__ == '__main__':
    print("="*50)
    print("🦷 SamayDent IA - Servidor v12.18 (Modelos en Producción)")
    print("🌐 http://127.0.0.1:5000")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)