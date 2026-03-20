from nicegui import ui, events
from ultralytics import YOLO
import cv2
import numpy as np
import base64
from pathlib import Path
import os
import threading
import time
import asyncio
import inspect

# --- Configuration ---
# Look for models in the current directory and its parent (where yolov8n.pt might be)
ROOT_DIR = Path(__file__).parent.resolve()
PARENT_DIR = ROOT_DIR.parent.resolve()

# --- State ---
class AppState:
    def __init__(self):
        self._model = None
        self.model_path = ""
        self.current_image_bytes = None
        self.annotated_image_b64 = None
        self.busy = False
        self.results = None
        self.detected_objects = []

    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
        print(f"DEBUG: Model set to {getattr(value, 'model_name', 'Loaded')}")

state = AppState()

# --- Helpers ---
def extract_detections(result) -> list[dict[str, float | int | str]]:
    detections = []

    if not result or result.boxes is None:
        return detections

    names = getattr(result, 'names', {}) or {}

    for index, box in enumerate(result.boxes, 1):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        label = names.get(cls, f"class_{cls}") if isinstance(names, dict) else f"class_{cls}"
        detections.append({
            'index': index,
            'class_id': cls,
            'label': label,
            'confidence': conf,
        })

    return detections

def get_available_models():
    # Gather .pt files from current, parent, and runs directory
    models = list(ROOT_DIR.glob("*.pt"))
    models += list(PARENT_DIR.glob("*.pt"))
    models += list(ROOT_DIR.glob("runs/detect/*/weights/*.pt"))
    
    # Standard models that YOLO knows how to download
    standard_models = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt', 'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt']
    
    # Filter unique and return relative paths or just names
    all_models = sorted(list(set([str(m) for m in models] + standard_models)))
    print(f"DEBUG: Found models: {all_models}")
    return all_models

def load_model(path: str):
    if not path: return
    state.busy = True
    print(f"DEBUG: Loading model from {path}")
    try:
        state.model = YOLO(path)
        state.model_path = path
        ui.notify(f"Model yüklendi: {Path(path).name}", type='positive', position='top')
    except Exception as e:
        print(f"DEBUG: Model load error: {e}")
        ui.notify(f"Model yükleme hatası: {e}", type='negative', position='top')
    finally:
        state.busy = False

def run_detection():
    if not state.model:
        ui.notify("Lütfen önce bir model seçin!", type='warning', position='top')
        return
    if not state.current_image_bytes:
        ui.notify("Lütfen bir görüntü yükleyin!", type='warning', position='top')
        return

    state.busy = True
    progress = ui.download_progress if hasattr(ui, 'download_progress') else None
    
    try:
        # Convert bytes to cv2 image
        nparr = np.frombuffer(state.current_image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run YOLO
        results = state.model.predict(img, conf=0.3)
        state.results = results[0]
        state.detected_objects = extract_detections(state.results)
        
        # Process image for display
        annotated_frame = results[0].plot()
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        state.annotated_image_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Update UI element
        img_display.set_source(f"data:image/jpeg;base64,{state.annotated_image_b64}")
        
        ui.notify(f"Tespit tamamlandı: {len(state.detected_objects)} nesne bulundu", type='positive', position='top')
    except Exception as e:
        state.detected_objects = []
        ui.notify(f"Tespit hatası: {e}", type='negative', position='top')
    finally:
        state.busy = False

# --- UI Layout ---

@ui.page('/')
def index():
    global img_display
    
    ui.colors(primary='#3b82f6', secondary='#64748b', accent='#10b981', dark='#0f172a')
    ui.dark_mode().enable()

    # Style definitions
    card_style = 'bg-[#1e293b] border border-slate-700 shadow-xl rounded-xl'

    with ui.row().classes('w-full h-full p-6 gap-6 no-wrap overflow-auto'):
        
        # --- LEFT SIDEBAR ---
        with ui.column().classes('w-96 flex-none gap-6'):
            
            # Model Card
            with ui.card().classes(card_style + ' w-full p-5'):
                ui.label('MODEL YÖNETİMİ').classes('text-xs font-bold text-slate-500 mb-2 uppercase tracking-widest')
                models = get_available_models()
                m_select = ui.select(models, label='Aktif Model', on_change=lambda e: load_model(e.value)).classes('w-full')
                if models and not state.model_path:
                    m_select.value = models[0]
                    load_model(models[0])
                
                with ui.row().classes('mt-4 items-center gap-2 text-slate-400'):
                    ui.icon('info', size='16px')
                    ui.label('YOLOv8/v11 desteklenir.').classes('text-[11px] font-medium')

            # Prediction Controls
            with ui.card().classes(card_style + ' w-full p-5 flex-grow'):
                ui.label('İŞLEM MERKEZİ').classes('text-xs font-bold text-slate-500 mb-4 uppercase tracking-widest')
                
                upload_label = ui.label('Henüz görüntü seçilmedi').classes('text-[10px] text-slate-500 mb-2 italic')

                async def handle_upload(e: events.UploadEventArguments):
                    print(f"DEBUG: Upload triggered. Event type: {type(e)}")
                    print(f"DEBUG: Event attributes: {dir(e)}")
                    try:
                        # For NiceGUI 1.x/2.x/3.x
                        file_obj = getattr(e, 'file', None)
                        if file_obj:
                            # In newer versions read() is a coroutine
                            read_attr = getattr(file_obj, 'read')
                            if asyncio.iscoroutinefunction(read_attr) or inspect.iscoroutine(read_attr):
                                state.current_image_bytes = await read_attr()
                            else:
                                state.current_image_bytes = read_attr()
                            
                            name = getattr(file_obj, 'name', 'unknown')
                        elif hasattr(e, 'content'):
                            content_attr = getattr(e.content, 'read')
                            if asyncio.iscoroutinefunction(content_attr) or inspect.iscoroutine(content_attr):
                                state.current_image_bytes = await content_attr()
                            else:
                                state.current_image_bytes = content_attr()
                            name = getattr(e, 'name', 'unknown')
                        else:
                            raise AttributeError("Upload event has no 'file' or 'content' attribute")
                        
                        print(f"DEBUG: Read {len(state.current_image_bytes)} bytes from {name}")
                        
                        b64 = base64.b64encode(state.current_image_bytes).decode('utf-8')
                        img_display.set_source(f"data:image/jpeg;base64,{b64}")
                        upload_label.set_text(f"Yüklendi: {name}")
                        state.results = None
                        state.detected_objects = []
                        ui.notify(f"Görüntü yüklendi: {name}", position='top', type='info')
                    except Exception as ex:
                        print(f"DEBUG: Upload error: {ex}")
                        ui.notify(f"Yükleme hatası: {ex}", type='negative')

                ui.upload(on_upload=handle_upload, label='GÖRÜNTÜ SEÇ', auto_upload=True).classes('w-full border-dashed border-2 border-slate-700 bg-slate-800/20 rounded-lg overflow-hidden').props('flat color=primary icon=cloud_upload')
                
                ui.button('ANALİZ BAŞLAT', icon='psychology', on_click=run_detection).classes('w-full mt-6 py-4 rounded-xl shadow-lg shadow-primary/20').props('color=primary unelevated')
                
                ui.separator().classes('my-6 opacity-30')
                
                ui.label('İSTATİSTİKLER').classes('text-xs font-bold text-slate-500 mb-2 uppercase tracking-widest')
                with ui.row().classes('w-full justify-between items-center p-3 bg-slate-800/40 rounded-lg'):
                    ui.label('Tespit Edilen Nesne:').classes('text-slate-300 text-sm')
                    stats_label = ui.label('0').classes('text-primary font-black text-xl')
                
                def refresh_stats():
                    stats_label.set_text(str(len(state.detected_objects)))
                ui.timer(0.5, refresh_stats)

        # --- MAIN DISPLAY AREA ---
        with ui.column().classes('flex-grow gap-6'):
            
            # Viewport
            with ui.card().classes(card_style + ' w-full flex-grow p-0 overflow-hidden relative flex items-center justify-center bg-black/40'):
                ui.label('VİZYON ÇIKTISI').classes('absolute top-4 left-4 z-10 text-[10px] font-bold text-white bg-primary/80 px-3 py-1 rounded-full uppercase tracking-tighter shadow-lg')
                img_display = ui.image().classes('max-w-full max-h-full object-contain')
                
                # Overlay when busy
                with ui.column().classes('absolute inset-0 bg-dark/60 backdrop-blur-sm items-center justify-center z-20').bind_visibility_from(state, 'busy'):
                    ui.spinner('gears', size='64px', color='primary')
                    ui.label('ANALİZ EDİLİYOR...').classes('mt-4 text-primary font-bold tracking-widest')

            # Metrics / List
            with ui.card().classes(card_style + ' w-full h-64 p-5 overflow-hidden'):
                ui.label('NESNE DETAYLARI').classes('text-xs font-bold text-slate-500 mb-4 uppercase tracking-widest')
                
                with ui.scroll_area().classes('w-full h-full pr-4'):
                    log_container = ui.column().classes('w-full gap-3')
                    
                    def refresh_details():
                        log_container.clear()

                        if state.detected_objects:
                            with log_container:
                                detected_names = ', '.join(item['label'] for item in state.detected_objects)
                                with ui.card().classes('w-full p-3 bg-emerald-500/10 border border-emerald-500/20 rounded-lg'):
                                    ui.label('Tespit Edilen Nesneler').classes('text-[11px] font-bold uppercase tracking-widest text-emerald-300')
                                    ui.label(detected_names).classes('text-sm text-slate-100 font-medium break-words')

                                for item in state.detected_objects:
                                    label = item['label']
                                    conf = item['confidence']
                                    index = item['index']
                                    class_id = item['class_id']
                                    with ui.row().classes('w-full items-center gap-3 p-3 bg-slate-800/50 rounded-lg border border-slate-700/50 hover:bg-slate-800 transition-all'):
                                        ui.label(f'{index}.').classes('text-slate-500 font-bold min-w-[1.5rem]')
                                        with ui.column().classes('gap-0'):
                                            ui.label(label).classes('font-bold text-slate-200 text-sm')
                                            ui.label(f'Sınıf ID: {class_id}').classes('text-[11px] text-slate-500')
                                        ui.label(f'%{conf*100:.1f}').classes('ml-auto bg-primary/20 text-primary px-3 py-1 rounded-full text-xs font-bold')
                        elif state.results:
                            with log_container:
                                ui.label('Nesne bulunamadı...').classes('text-slate-500 italic text-sm text-center w-full mt-4')
                    
                    ui.timer(1.0, refresh_details)

# --- Entry Point ---
if __name__ in {"__main__", "__mp_main__"}:
    ui.run(
        title='TKNFST ModernUI',
        port=8080,
        dark=True,
        reload=True,
        favicon='🚀'
    )