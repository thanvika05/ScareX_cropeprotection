import os
import time
import cv2
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_file
from src.core.logger import logger
from src.logging.db_logger import db_logger
from src.core.config_manager import ConfigManager
from src.vision.tomato_analyzer import TomatoAnalyzer
from fpdf import FPDF

app = Flask(__name__)
config = ConfigManager()

# Global analyzer instance
analyzer = TomatoAnalyzer()
video_source = "dataset/videos/tomato_farm.mp4"

def generate_video_stream():
    """Generator function to yield video frames for the web stream."""
    while True:
        if analyzer.running:
            frame = analyzer.get_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Video ended
                save_session()
                time.sleep(1)
        else:
            time.sleep(0.1)

def save_session():
    duration = time.time() - analyzer.start_time
    total = sum(analyzer.session_stats.values())
    
    full_pct = (analyzer.session_stats['fully_ripened'] / total * 100) if total > 0 else 0
    priority = "LOW"
    if full_pct > 50:
        priority = "HIGH"
    elif full_pct > 20:
        priority = "MEDIUM"
        
    if total > 0:
        db_logger.log_event(
            event_type="MONITORING_SESSION_END",
            species="Tomato",
            confidence=1.0,
            alarm_played="None",
            system_status=f"Duration: {duration:.1f}s, Total: {total}",
            detection_source="DASHBOARD",
            duration=duration,
            green_count=analyzer.session_stats['green'],
            half_ripened_count=analyzer.session_stats['half_ripened'],
            fully_ripened_count=analyzer.session_stats['fully_ripened'],
            harvest_priority=priority
        )

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/start_webcam', methods=['POST'])
def start_webcam():
    success = analyzer.start(0)
    return jsonify({"success": success})

@app.route('/api/start_video', methods=['POST'])
def start_video():
    success = analyzer.start(video_source)
    return jsonify({"success": success})

@app.route('/api/stop', methods=['POST'])
def stop_analysis():
    analyzer.stop()
    save_session()
    return jsonify({"success": True})

@app.route('/api/status')
def system_status():
    stats = analyzer.session_stats
    total = sum(stats.values())
    
    green_pct = (stats['green'] / total * 100) if total > 0 else 0
    half_pct = (stats['half_ripened'] / total * 100) if total > 0 else 0
    full_pct = (stats['fully_ripened'] / total * 100) if total > 0 else 0
    
    priority = "LOW"
    if full_pct > 50:
        priority = "HIGH"
    elif full_pct > 20:
        priority = "MEDIUM"
        
    duration = time.time() - analyzer.start_time if analyzer.running else 0
        
    return jsonify({
        "running": analyzer.running,
        "total": total,
        "green": stats['green'],
        "half_ripened": stats['half_ripened'],
        "fully_ripened": stats['fully_ripened'],
        "green_pct": round(green_pct, 1),
        "half_pct": round(half_pct, 1),
        "full_pct": round(full_pct, 1),
        "priority": priority,
        "duration": round(duration, 1)
    })

@app.route('/api/report/pdf')
def generate_pdf():
    stats = analyzer.session_stats
    total = sum(stats.values())
    
    green_pct = (stats['green'] / total * 100) if total > 0 else 0
    half_pct = (stats['half_ripened'] / total * 100) if total > 0 else 0
    full_pct = (stats['fully_ripened'] / total * 100) if total > 0 else 0
    
    priority = "LOW"
    if full_pct > 50:
        priority = "HIGH"
    elif full_pct > 20:
        priority = "MEDIUM"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="SCareX", ln=True, align='C')
    pdf.cell(200, 10, txt="Daily Tomato Crop Monitoring Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.cell(200, 10, txt="Video: tomato_farm.mp4 / Webcam", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Field Summary:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Detected tomatoes: {total}", ln=True)
    pdf.cell(200, 10, txt=f"Green: {stats['green']}", ln=True)
    pdf.cell(200, 10, txt=f"Half-ripened: {stats['half_ripened']}", ln=True)
    pdf.cell(200, 10, txt=f"Fully-ripened: {stats['fully_ripened']}", ln=True)
    
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Maturity Distribution:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Green percentage: {green_pct:.1f}%", ln=True)
    pdf.cell(200, 10, txt=f"Half-ripened percentage: {half_pct:.1f}%", ln=True)
    pdf.cell(200, 10, txt=f"Fully-ripened percentage: {full_pct:.1f}%", ln=True)
    
    pdf.ln(10)
    pdf.cell(200, 10, txt="Observation:", ln=True)
    pdf.cell(200, 10, txt="Percentages are based on tomatoes detected in the camera/video view.", ln=True)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt=f"Harvest Priority: {priority}", ln=True)
    
    pdf_path = "scarex_report.pdf"
    pdf.output(pdf_path)
    return send_file(os.path.abspath(pdf_path), as_attachment=True)

def start_flask_app(fusion, vision, audio):
    # Dummy to maintain compatibility if called from main.py
    port = config.get("flask_port", 5000)
    logger.info(f"Starting Flask Dashboard on port {port}...")
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False), daemon=True).start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
