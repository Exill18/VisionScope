import os
import glob
import json
import hashlib
import numpy as np
import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

# --- Configuration ---
IMAGE_DIR = 'data/images/train'
LABEL_DIR = 'data/labels/train'
HISTORY_FILE = 'labeled_images.json'
CLASSES = [
    "class_0", "class_1", "class_2", "class_3",
    "class_4", "class_5", "class_6", "class_7"
]
ENABLE_PRELABEL = True
PRELABEL_MODEL = 'yolov8n.pt'
PRELABEL_CONF_THRESHOLD = 0.25

class ImageLabelerGUI(tk.Tk):
    """
    A GUI application for labeling objects in images for YOLO training.
    """
    def __init__(self):
        super().__init__()
        self.title("VisionScope Labeling Tool")
        self.geometry("1200x800")

        # --- Internal State ---
        self.image_paths = []
        self.current_image_index = -1
        self.original_img = None
        self.tk_img = None
        self.boxes = []
        self.current_class_index = 0

        # Drawing/Panning/Zooming state
        self.drawing = False
        self.start_point = (0, 0)
        self.zoom_scale = 1.0
        self.view_offset = np.array([0.0, 0.0])
        self.is_panning = False
        self.pan_start = (0, 0)

        # --- Model and History ---
        self.yolo_model = YOLO(PRELABEL_MODEL) if ENABLE_PRELABEL else None
        self.history_hashes = self._load_history()
        os.makedirs(LABEL_DIR, exist_ok=True)

        # --- GUI Setup ---
        self._create_widgets()
        self._bind_events()

        # --- Load Data ---
        self.load_image_paths()
        self.load_next_image()

    def _create_widgets(self):
        """Creates and lays out the GUI widgets."""
        # Main layout
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        # Canvas for image display
        self.canvas = tk.Canvas(main_frame, bg="gray20", cursor="cross")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Right-side control panel
        control_panel = ttk.Frame(main_frame, width=250)
        control_panel.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        control_panel.grid_propagate(False) # Prevent resizing

        # Class selection buttons
        class_frame = ttk.LabelFrame(control_panel, text="Classes")
        class_frame.pack(fill=tk.X, pady=10)
        self.class_buttons = []
        for i, class_name in enumerate(CLASSES):
            btn = ttk.Button(class_frame, text=f"({i}) {class_name}", command=lambda i=i: self.set_class(i))
            btn.pack(fill=tk.X, padx=5, pady=2)
            self.class_buttons.append(btn)
        
        # Action buttons
        action_frame = ttk.LabelFrame(control_panel, text="Actions")
        action_frame.pack(fill=tk.X, pady=10)
        ttk.Button(action_frame, text="Undo (Z)", command=self.undo_last_box).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(action_frame, text="Reset View (R)", command=self.reset_view).pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(action_frame, text="Save & Next (Enter)", command=self.save_and_next).pack(fill=tk.X, padx=5, pady=5)
        
        # Status Bar
        self.status_var = tk.StringVar(value="Loading...")
        status_bar = ttk.Label(self, textvariable=self.status_var, anchor=tk.W, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_events(self):
        """Binds mouse and keyboard events."""
        # Canvas mouse events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel) # Windows/macOS
        self.canvas.bind("<Button-4>", self.on_mouse_wheel) # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel) # Linux scroll down
        self.canvas.bind("<ButtonPress-2>", self.on_pan_press) # Middle mouse button
        self.canvas.bind("<B2-Motion>", self.on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_pan_release)

        # Global key events
        self.bind("<KeyPress>", self.on_key_press)
        
        # Handle window resize
        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())


    def load_image_paths(self):
        """Finds all images in the directory that haven't been labeled yet."""
        all_paths = sorted(
            glob.glob(os.path.join(IMAGE_DIR, "*.jpg")) +
            glob.glob(os.path.join(IMAGE_DIR, "*.jpeg")) +
            glob.glob(os.path.join(IMAGE_DIR, "*.png"))
        )
        self.image_paths = [p for p in all_paths if self.get_file_hash(p) not in self.history_hashes]
        if not self.image_paths:
            messagebox.showinfo("Done!", "All images have been labeled.")
            self.quit()

    def load_next_image(self):
        """Loads the next available image and prepares it for labeling."""
        self.current_image_index += 1
        if self.current_image_index >= len(self.image_paths):
            messagebox.showinfo("Done!", "All images have been labeled.")
            self.quit()
            return

        img_path = self.image_paths[self.current_image_index]
        self.original_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        self.boxes = []
        self.reset_view()
        
        filename = os.path.basename(img_path)
        self.status_var.set(f"Labeling: {filename}")

        if self.yolo_model:
            self._run_prelabeling()

        self.redraw_canvas()
        self.update_class_highlight()

    def redraw_canvas(self):
        """Clears and redraws the entire canvas with the image and boxes."""
        if self.original_img is None:
            return
        
        self.canvas.delete("all")

        # Get the current view (zoomed and panned)
        view = self._get_view()
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(view))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        # Draw existing boxes, transforming their coordinates to the view
        for (x, y, w, h, cls_idx) in self.boxes:
            x1_view, y1_view = self._transform_coords_to_view((x, y))
            x2_view, y2_view = self._transform_coords_to_view((x + w, y + h))
            
            # Only draw if the box is at least partially visible
            if x2_view > 0 and y2_view > 0 and x1_view < self.canvas.winfo_width() and y1_view < self.canvas.winfo_height():
                self.canvas.create_rectangle(x1_view, y1_view, x2_view, y2_view, outline="cyan", width=2, tags="box")
                self.canvas.create_text(x1_view, y1_view - 5, text=CLASSES[cls_idx], fill="cyan", anchor=tk.SW, tags="box")
        
        # Live drawing preview
        if self.drawing:
            x1, y1 = self.start_point
            x2, y2 = self.end_point_view
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="lime green", width=2, dash=(4, 4))
            
    def _get_view(self):
        """Crops and resizes the original image based on zoom and pan."""
        img_h, img_w = self.original_img.shape[:2]
        canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()

        # Define the portion of the original image to show
        view_w_orig = int(img_w / self.zoom_scale)
        view_h_orig = int(img_h / self.zoom_scale)

        # Clamp offset to prevent panning outside the image
        self.view_offset[0] = np.clip(self.view_offset[0], 0, img_w - view_w_orig)
        self.view_offset[1] = np.clip(self.view_offset[1], 0, img_h - view_h_orig)

        x_start, y_start = self.view_offset.astype(int)
        crop = self.original_img[y_start : y_start + view_h_orig, x_start : x_start + view_w_orig]

        return cv2.resize(crop, (canvas_w, canvas_h), interpolation=cv2.INTER_AREA)

    # --- Event Handlers ---
    def on_key_press(self, event):
        """Handles global keyboard shortcuts."""
        if '0' <= event.keysym <= '9':
            class_idx = int(event.keysym)
            if class_idx < len(CLASSES):
                self.set_class(class_idx)
        elif event.keysym.lower() == 'z':
            self.undo_last_box()
        elif event.keysym.lower() == 'r':
            self.reset_view()
        elif event.keysym == 'Return':
            self.save_and_next()

    def on_mouse_press(self, event):
        self.drawing = True
        self.start_point = (event.x, event.y)
        self.end_point_view = (event.x, event.y)

    def on_mouse_drag(self, event):
        if self.drawing:
            self.end_point_view = (event.x, event.y)
            self.redraw_canvas()

    def on_mouse_release(self, event):
        if not self.drawing: return
        self.drawing = False
        
        start_img_coords = self._transform_coords_to_image(self.start_point)
        end_img_coords = self._transform_coords_to_image((event.x, event.y))
        
        x1, y1 = start_img_coords
        x2, y2 = end_img_coords
        
        # Ensure box has a positive area
        if abs(x1 - x2) > 1 and abs(y1 - y2) > 1:
            box = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1), self.current_class_index]
            self.boxes.append(box)
        
        self.redraw_canvas()

    def on_mouse_wheel(self, event):
        """Handles zooming with the mouse wheel."""
        img_coords_before_zoom = self._transform_coords_to_image((event.x, event.y))

        if event.num == 5 or event.delta < 0: # Scroll down
            self.zoom_scale /= 1.1
        if event.num == 4 or event.delta > 0: # Scroll up
            self.zoom_scale *= 1.1
        
        self.zoom_scale = np.clip(self.zoom_scale, 1.0, 10.0)

        # Adjust pan to keep the point under the cursor stationary
        img_coords_after_zoom = self._transform_coords_to_image((event.x, event.y))
        self.view_offset += np.array(img_coords_before_zoom) - np.array(img_coords_after_zoom)

        self.redraw_canvas()

    def on_pan_press(self, event):
        self.is_panning = True
        self.pan_start = (event.x, event.y)
        self.canvas.config(cursor="fleur")

    def on_pan_drag(self, event):
        if self.is_panning:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.pan_start = (event.x, event.y)
            
            # The distance moved in the view needs to be scaled back to original image coordinates
            self.view_offset -= np.array([dx, dy]) * (self.original_img.shape[1] / self.zoom_scale / self.canvas.winfo_width())
            self.redraw_canvas()

    def on_pan_release(self, event):
        self.is_panning = False
        self.canvas.config(cursor="cross")

    # --- Actions ---
    def set_class(self, class_idx):
        self.current_class_index = class_idx
        self.update_class_highlight()

    def undo_last_box(self):
        if self.boxes:
            self.boxes.pop()
            self.redraw_canvas()

    def reset_view(self):
        self.zoom_scale = 1.0
        self.view_offset = np.array([0.0, 0.0])
        self.redraw_canvas()

    def save_and_next(self):
        """Saves labels for the current image and loads the next one."""
        img_path = self.image_paths[self.current_image_index]
        filename = os.path.basename(img_path)
        label_path = os.path.join(LABEL_DIR, f"{os.path.splitext(filename)[0]}.txt")
        file_hash = self.get_file_hash(img_path)

        img_h, img_w = self.original_img.shape[:2]
        with open(label_path, 'w') as f:
            for (x, y, w, h, cls) in self.boxes:
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
        
        self.history_hashes.add(file_hash)
        self._save_history()
        self.load_next_image()

    # --- Helpers ---
    def _run_prelabeling(self):
        """Uses YOLO to suggest initial bounding boxes."""
        results = self.yolo_model(self.original_img, conf=PRELABEL_CONF_THRESHOLD, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_idx = int(box.cls[0])
            if cls_idx < len(CLASSES):
                self.boxes.append([x1, y1, x2 - x1, y2 - y1, cls_idx])

    def _transform_coords_to_image(self, view_coords):
        """Converts coordinates from the canvas view to the original image."""
        vx, vy = view_coords
        
        # Scale factor from view to original image crop
        scale = (self.original_img.shape[0] / self.zoom_scale) / self.canvas.winfo_height()
        
        # Coordinates within the original image
        img_x = self.view_offset[0] + vx * scale
        img_y = self.view_offset[1] + vy * scale
        
        return int(img_x), int(img_y)

    def _transform_coords_to_view(self, img_coords):
        """Converts coordinates from the original image to the canvas view."""
        ix, iy = img_coords
        
        # Scale factor from original image crop to view
        scale = self.canvas.winfo_height() / (self.original_img.shape[0] / self.zoom_scale)

        view_x = (ix - self.view_offset[0]) * scale
        view_y = (iy - self.view_offset[1]) * scale
        
        return int(view_x), int(view_y)
    
    def update_class_highlight(self):
        """Updates the visual style of class buttons to show the active one."""
        for i, btn in enumerate(self.class_buttons):
            style = "Accent.TButton" if i == self.current_class_index else "TButton"
            btn.config(style=style)

    def _load_history(self):
        """Loads the set of hashes of already labeled images."""
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                # In the old version, this was a list of dicts. We just need the hashes.
                history_data = json.load(f)
                return {entry['hash'] for entry in history_data}
        return set()

    def _save_history(self):
        """Saves the history file. For compatibility, we save in the old format."""
        # This is inefficient but maintains compatibility with the old script's format.
        # A better approach would be to just save the set of hashes.
        dummy_filename = "labeled"
        history_list = [{"filename": dummy_filename, "hash": h} for h in self.history_hashes]
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history_list, f, indent=2)

    @staticmethod
    def get_file_hash(filepath):
        """Calculates the SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


if __name__ == "__main__":
    app = ImageLabelerGUI()
    
    # Add a simple theme and an accent color for the selected button
    style = ttk.Style(app)
    style.theme_use('clam') # Other options: 'alt', 'default', 'classic'
    style.configure("Accent.TButton", foreground="white", background="navy")

    app.mainloop()
